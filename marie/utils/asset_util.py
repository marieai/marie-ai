import functools
import hashlib
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

from PIL import Image

from marie.common.file_io import get_cache_dir
from marie.logging_core.predefined import default_logger as logger
from marie.utils.image_utils import hash_frames_fast
from marie.utils.utils import ensure_exists


def avoid_concurrent_lock_wrapper(func: Callable) -> Callable:
    """
    Wrap the function around a File Lock to ensure that the function is run by a single replica on the same machine.

    :param func: The function to decorate
    :return: The wrapped function
    """

    @functools.wraps(func)
    def arg_wrapper(*args, **kwargs):
        from marie.importer import ImportExtensions

        with ImportExtensions(
            required=False,
            help_text='FileLock is needed to guarantee non-concurrent execution of the function.',
        ):
            import filelock

            locks_root = Path.home().joinpath('.locks')
            locks_root.mkdir(parents=True, exist_ok=True)
            lock_file = locks_root.joinpath(f'{func.__name__}.lock')

            file_lock = filelock.FileLock(lock_file, timeout=-1)

        with file_lock:
            return func(*args, **kwargs)

    return arg_wrapper


@avoid_concurrent_lock_wrapper
def prepare_asset_directory(
    frames,
    local_path: str,
    ref_id: str,
    ref_type: str,
    logger,
    restore_dirs: list[str] | None = None,
):
    """
    Prepares the asset directory by creating the required subdirectories and processing input files.

    :param frames: List of frames (numpy arrays) to process.
    :param local_path: Local file path of the downloaded S3 file.
    :param ref_id: Unique identifier for the asset reference.
    :param ref_type: Type of the reference for the asset.
    :param logger: Logger instance to handle logging.
    :param restore_dirs: Optional list of directory names to download from S3
        (e.g. ``["agent-output"]``).  Downloaded after metadata is fetched.
    :return: Tuple containing root asset directory path, frames directory path, and metadata file path.
    :raises ValueError: If the local_path parameter is None.
    """

    if local_path is None:
        logger.error("The 'local_path' parameter is None. Unable to proceed.")
        raise ValueError("The 'local_path' parameter cannot be None.")

    root_asset_dir = create_working_dir(frames, ref_id=ref_id, ref_type=ref_type)
    frames_dir = os.path.join(root_asset_dir, "frames")
    ensure_exists(frames_dir)

    existing_files = (
        sorted(os.listdir(frames_dir)) if os.path.exists(frames_dir) else []
    )

    burst_dir = os.path.join(root_asset_dir, "burst")
    if not existing_files and os.path.exists(burst_dir):
        burst_files = sorted([f for f in os.listdir(burst_dir) if f.endswith('.tif')])
        if burst_files:
            for idx, file in enumerate(burst_files):
                src = os.path.join(burst_dir, file)
                dst = os.path.join(frames_dir, f"{idx + 1:05}.png")
                shutil.copy2(src, dst)
            existing_files = sorted(os.listdir(frames_dir))

    if existing_files:
        existing_frames = [
            os.path.join(frames_dir, file)
            for file in existing_files
            if file.endswith('.png')
        ]
        valid_frames = len(existing_frames) == len(frames) and all(
            os.path.isfile(os.path.join(frames_dir, f"{idx + 1:05}.png"))
            for idx in range(len(frames))
        )

        if valid_frames:
            logger.info(
                f"Frames already exist in '{frames_dir}' and match the expected format. Skipping further processing."
            )
            metadata_file = os.path.join(root_asset_dir, f"{ref_id}.meta.json")
            return root_asset_dir, frames_dir, metadata_file

    # Copy local file to the target path in the asset directory
    target_path = os.path.join(root_asset_dir, ref_id)
    if not os.path.exists(target_path):
        shutil.copy2(local_path, target_path)
        with open(target_path, 'a') as f:
            f.flush()
            os.fsync(f.fileno())
        logger.info(f"Copied file from '{local_path}' to '{target_path}'.")

    logger.info(f"Root asset directory created: '{root_asset_dir}'")

    for idx, frame in enumerate(frames):
        frame_path = os.path.join(frames_dir, f"{idx + 1:05}.png")
        try:
            Image.fromarray(frame).save(frame_path)
            logger.debug(f"Frame {idx + 1} saved at '{frame_path}'.")

            img = Image.open(frame_path)
            logger.debug(f"Image dimensions: {img.size}")
        except Exception as e:
            logger.error(f"Error while processing frame {idx + 1} - {e}")
            raise

    # Download additional metadata for the asset
    metadata_file = download_asset(
        ref_id=ref_id,
        ref_type=ref_type,
        root_asset_dir=root_asset_dir,
        s3_file_path=f"{ref_id}.meta.json",
        overwrite=True,
    )
    if metadata_file is None:
        raise FileNotFoundError(
            f"Failed to download metadata file '{ref_id}.meta.json' from S3 "
            f"for ref_id={ref_id}, ref_type={ref_type}"
        )
    logger.info(f"Metadata file downloaded and stored at: '{metadata_file}'")
    time.sleep(0.1)  # Ensure file system operations are completed

    # Download additional directories from S3 (e.g. agent-output from upstream nodes)
    if restore_dirs:
        from marie.storage import StorageManager

        s3_root_path = s3_asset_path(ref_id, ref_type)
        connected = StorageManager.ensure_connection("s3://", silence_exceptions=True)
        if connected:
            for dir_name in restore_dirs:
                try:
                    StorageManager.copy_remote(
                        s3_root_path,
                        root_asset_dir,
                        match_wildcard=f"*/{dir_name}/*",
                        overwrite=True,
                        silence_errors=True,
                    )
                except Exception as e:
                    logger.warning(f"Failed to restore '{dir_name}' from S3: {e}")
        else:
            logger.warning("Could not connect to S3 to restore directories")

    # Ensure the metadata file exists and that it is a valid JSON file
    if not os.path.exists(metadata_file):
        logger.error(f"Metadata file '{metadata_file}' does not exist.")
        raise FileNotFoundError(f"Metadata file '{metadata_file}' not found.")

    try:
        with open(metadata_file, 'r') as f:
            metadata = f.read()
            if not metadata.strip().startswith('{'):
                raise ValueError(
                    f"Metadata file '{metadata_file}' is not a valid JSON."
                )
    except Exception as e:
        logger.error(f"Error reading metadata file '{metadata_file}': {e}")
        raise

    return root_asset_dir, frames_dir, metadata_file


def create_working_dir(
    frames: List,
    backup: bool = False,
    ref_id: str = None,
    ref_type: str = None,
    queue_id: str = None,
    job_id: str = None,
) -> str:
    """Create an isolated working directory for a pipeline execution.

    By default the legacy layout (``~/.marie/generators/<frame_checksum>``)
    is used for backwards compatibility.  Set the environment variable
    ``MARIE_LEGACY_WORKING_DIR=false`` to opt-in to the new layout that
    derives the directory hash from frame pixel data **and** request
    identifiers (ref_id, ref_type) so that concurrent jobs never collide
    even when processing identical frames.

    .. deprecated::
        The legacy layout is deprecated and will be removed in the next
        major release.
    """
    frame_checksum = hash_frames_fast(frames=frames)

    # MARIE_LEGACY_WORKING_DIR defaults to True (legacy behaviour).
    # Set MARIE_LEGACY_WORKING_DIR=false to opt-in to the new isolated
    # working directory layout that includes ref_id/ref_type in the hash.
    # NOTE: The legacy layout is deprecated and will be removed in the
    # next major release.
    use_legacy = os.environ.get("MARIE_LEGACY_WORKING_DIR", "true").lower() not in (
        "false",
        "0",
        "no",
    )
    if use_legacy:
        import warnings

        warnings.warn(
            "MARIE_LEGACY_WORKING_DIR is deprecated and will be removed in the "
            "next major release. Set MARIE_LEGACY_WORKING_DIR=false to opt-in "
            "to the new isolated working directory layout.",
            DeprecationWarning,
            stacklevel=2,
        )
        cache_dir = get_cache_dir()
        target_dir = os.path.join(cache_dir, "generators", frame_checksum)
    else:
        md5 = hashlib.md5(frame_checksum.encode("utf-8"))
        for token in (ref_id, ref_type):
            if token:
                md5.update(token.encode("utf-8"))
        combined_checksum = md5.hexdigest()
        generators_dir = os.path.join(get_cache_dir(), "generators")
        os.makedirs(generators_dir, exist_ok=True)
        target_dir = os.path.join(generators_dir, combined_checksum)

    # create backup name by appending a timestamp
    if backup and os.path.exists(target_dir):
        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        shutil.move(target_dir, f"{target_dir}-{ts}")

    return ensure_exists(target_dir)


def split_filename(img_path: str) -> tuple[str, str, str]:
    filename = img_path.split("/")[-1]
    prefix, suffix = os.path.splitext(filename)
    suffix = suffix.lstrip('.')

    return filename, prefix, suffix


def filename_supplier_page(
    filename: str, prefix: str, suffix: str, pagenumber: int
) -> str:
    return f"{prefix}_{pagenumber:05}.{suffix}"


def s3_asset_path(
    ref_id: str, ref_type: str, include_prefix=False, include_filename=False
) -> str:
    """
    Create a path to store the assets for a given ref_id and ref_type
    The path is of the form s3://marie/{ref_type}/{prefix} and can be used between different marie instances

    All paths are lowercased and ref_type is cleaned to avoid path traversal attacks by replacing "/" with "_",

    Following are equivalent:

    .. code-block:: text

        s3://marie/ocr/sample
        s3://marie/OCR/sample
        s3://marie/ocr/SAMPLE
        s3://marie/OCR/SAMPLE


    Example usage:

    .. code-block:: python

        # this will return s3://marie/ocr/sample
        path = s3_asset_path(ref_id="sample.tif", ref_type="ocr")

        # this will return s3://marie/ocr/sample/sample
        path = s3_asset_path(ref_id="sample.tif", ref_type="ocr", include_prefix=True)

        # this will return s3://marie/ocr/sample/SAMple.tif
        path = s3_asset_path(ref_id="SAMple.tif", ref_type="ocr", include_filename=True)

    :param ref_type: type of the reference document
    :param ref_id:  id of the reference document
    :param include_prefix: include the filename prefix in the path(name of the file without extension)
    :param include_filename: include the filename in the path (name of the file with extension)
    :return: s3 path to store the assets
    """
    # prefix and filename need to be exclusive of each other
    assert not (include_prefix and include_filename)

    filename, prefix, suffix = split_filename(ref_id)
    # clean ref_type to avoid path traversal attacks
    ref_type = ref_type.replace("/", "_").lower()
    marie_bucket = os.environ.get("MARIE_S3_BUCKET", "marie")

    ret_path = f"s3://{marie_bucket}/{ref_type.lower()}/{prefix.lower()}"
    if include_prefix:
        ret_path = f"s3://{marie_bucket}/{ref_type.lower()}/{prefix.lower()}/{prefix}"

    if include_filename:
        ret_path = f"s3://{marie_bucket}/{ref_type.lower()}/{prefix.lower()}/{filename}"

    return ret_path


def restore_assets(
    ref_id: str,
    ref_type: str,
    root_asset_dir: str,
    full_restore=False,
    overwrite=False,
) -> Optional[str]:
    """
    Restore assets from primary storage (S3) into root asset directory. This restores
    the assets from the last run of the extract pipeline.

    :param ref_id: document reference id (e.g. filename)
    :param ref_type: document reference type(e.g. document, page, process)
    :param root_asset_dir: root asset directory
    :param full_restore: if True, restore all assets, otherwise only restore subset of assets (clean, results, pdf)
    that are required for the extract pipeline.
    :param overwrite: if True, overwrite existing assets in root asset directory
    :return:
    """
    from marie.storage import StorageManager

    s3_root_path = s3_asset_path(ref_id, ref_type)
    connected = StorageManager.ensure_connection("s3://", silence_exceptions=True)
    if not connected:
        logger.error(f"Error restoring assets : Could not connect to S3")
        return None

    logger.info(f"Restoring assets from {s3_root_path} to {root_asset_dir}")

    if full_restore:
        try:
            StorageManager.copy_remote(
                s3_root_path,
                root_asset_dir,
                match_wildcard="*",
                overwrite=overwrite,
            )
        except Exception as e:
            logger.error(f"Error restoring assets : {e}")
    else:
        dirs_to_restore = ["clean", "results", "pdf"]
        for dir_to_restore in dirs_to_restore:
            try:
                StorageManager.copy_remote(
                    s3_root_path,
                    root_asset_dir,
                    match_wildcard=f"*/{dir_to_restore}/*",
                    overwrite=overwrite,
                )
            except Exception as e:
                logger.error(f"Error restoring assets {dir_to_restore} : {e}")
    return s3_root_path


def store_assets(
    ref_id: str, ref_type: str, root_asset_dir: str, match_wildcard: Optional[str] = "*"
) -> list[str] | None:
    """
    Store assets in primary storage (S3)

    :param ref_id:  document reference id (e.g. filename)
    :param ref_type: document reference type (e.g. document, page, process)
    :param root_asset_dir: root asset directory where all assets are stored
    :param match_wildcard: wildcard to match files to store
    :return:
    """
    from marie.storage import StorageManager

    try:
        s3_asset_base = s3_asset_path(ref_id, ref_type)
        connected = StorageManager.ensure_connection("s3://", silence_exceptions=True)
        if not connected:
            logger.error(f"Error storing assets : Could not connect to S3")
            return [s3_asset_base]

        logger.info(f"store_assets for {ref_id} : {ref_type} : {root_asset_dir}")
        StorageManager.copy_dir(
            root_asset_dir,
            s3_asset_base,
            relative_to_dir=root_asset_dir,
            match_wildcard=match_wildcard,
        )

        return StorageManager.list(s3_asset_base, return_full_path=True)
    except Exception as e:
        logger.error(f"Error storing assets : {e}")


def download_asset(
    ref_id: str,
    ref_type: str,
    root_asset_dir: str,
    s3_file_path: str = "meta.json",
    overwrite=True,
) -> Optional[str]:
    """
    Download assets from primary storage (S3) into root asset directory. This restores
    the assets from the last run of the extract pipeline.

    :param ref_id: document reference id (e.g. filename)
    :param ref_type: document reference type(e.g. document, page, process)
    :param root_asset_dir: root asset directory
    :param s3_file_path: file path in S3
    :param overwrite: if True, overwrite existing assets in root asset directory
    :return:
    """
    from marie.storage import StorageManager

    s3_root_path = s3_asset_path(ref_id, ref_type)
    connected = StorageManager.ensure_connection("s3://", silence_exceptions=True)
    if not connected:
        logger.error(f"Error restoring assets : Could not connect to S3")
        return None

    uri = f"{s3_root_path}/{s3_file_path}"
    logger.info(f"Restoring assets from {uri} to {root_asset_dir}")
    output_file_path = os.path.join(root_asset_dir, s3_file_path)
    success = StorageManager.read_to_file(uri, output_file_path, overwrite=overwrite)
    if success is False:
        logger.error(
            f"Failed to download file '{s3_file_path}' from S3 for ref_id={ref_id}, ref_type={ref_type}"
        )
        return None
    return output_file_path
