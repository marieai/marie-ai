import functools
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

from omegaconf import OmegaConf
from PIL import Image

from marie.common.file_io import get_cache_dir
from marie.logging_core.predefined import default_logger as logger
from marie.pipe.components import download_asset
from marie.utils.image_utils import hash_frames_fast
from marie.utils.utils import ensure_exists

# TODO : This should be moved to a proper package


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
            try:
                print(f'Acquired lock for {func.__name__} with {file_lock}')
                return func(*args, **kwargs)
            finally:
                print(f'Released lock for {func.__name__}')

    return arg_wrapper


@avoid_concurrent_lock_wrapper
def prepare_asset_directory(
    frames, local_path: str, ref_id: str, ref_type: str, logger
):
    """
    Prepares the asset directory by creating the required subdirectories and processing input files.

    :param frames: List of frames (numpy arrays) to process.
    :param local_path: Local file path of the downloaded S3 file.
    :param ref_id: Unique identifier for the asset reference.
    :param ref_type: Type of the reference for the asset.
    :param logger: Logger instance to handle logging.
    :return: Tuple containing root asset directory path, frames directory path, and metadata file path.
    :raises ValueError: If the local_path parameter is None.
    """

    if local_path is None:
        logger.error("The 'local_path' parameter is None. Unable to proceed.")
        raise ValueError("The 'local_path' parameter cannot be None.")

    root_asset_dir = create_working_dir(frames)
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
    logger.info(f"Metadata file downloaded and stored at: '{metadata_file}'")
    time.sleep(0.1)  # Ensure file system operations are completed

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


def create_working_dir(frames: List, backup: bool = False) -> str:
    frame_checksum = hash_frames_fast(frames=frames)
    generators_dir = os.path.join(get_cache_dir(), "generators")
    os.makedirs(generators_dir, exist_ok=True)

    # create backup name by appending a timestamp
    if backup:
        if os.path.exists(os.path.join(generators_dir, frame_checksum)):
            ts = datetime.now().strftime("%Y%m%d%H%M%S")
            shutil.move(
                os.path.join(generators_dir, frame_checksum),
                os.path.join(generators_dir, f"{frame_checksum}-{ts}"),
            )
    return ensure_exists(os.path.join(generators_dir, frame_checksum))


def load_layout_config(base_dir: str, layout_dir: str, debug_config=False) -> OmegaConf:
    # TODO : THIS NEEDS TO BE CONFIGURABLE
    layout_dir = os.path.expanduser(layout_dir)
    base_dir = os.path.expanduser(base_dir)

    # root config
    base_cfg_path = os.path.join(base_dir, "base-config.yml")
    field_cfg_path = os.path.join(base_dir, "field-config.yml")
    # layout config
    layout_cfg_path = os.path.join(layout_dir, "config.yml")
    base_cfg = OmegaConf.load(base_cfg_path)
    field_cfg = OmegaConf.load(field_cfg_path)
    specific_cfg = OmegaConf.load(layout_cfg_path)
    merged_conf = OmegaConf.merge(field_cfg, base_cfg, specific_cfg)

    # Keys can come in few formats
    # - PATIENT NAME
    # - PATIENT    NAME
    # - PATIENT_NAME
    # for that we will normalize spaces into underscores (_)

    merged_conf.grounding = {
        k: [entry.replace(" ", "_") for entry in v]
        for k, v in merged_conf.grounding.items()
    }

    if debug_config:
        with open("merged_config_output.yml", "w") as yaml_file:
            yaml_file.write(OmegaConf.to_yaml(merged_conf))
            print(OmegaConf.to_yaml(merged_conf))

    return merged_conf


def layout_config(root_config_dir: str, layout_id: str) -> OmegaConf:
    base_dir = os.path.expanduser(os.path.join(root_config_dir, "base"))
    layout_dir = os.path.expanduser(
        os.path.join(root_config_dir, f"TID-{layout_id}/annotator")
    )

    logger.info(f"Layout ID : {layout_id}")
    logger.info(f"Base dir : {base_dir}")
    logger.info(f"Layout dir : {layout_dir}")

    cfg = load_layout_config(base_dir, layout_dir)
    logger.debug(f"Layout config : {cfg}")
    return cfg


def setup_table_directories(
    working_dir: str, annotator_name: Optional[str] = None
) -> tuple:
    """Setup required directories for table annotation."""
    output_dir = ensure_exists(os.path.join(working_dir, "agent-output"))
    htables_output_dir = os.path.join(output_dir, "highlighted_tables")
    table_src_dir = os.path.join(output_dir, "tables")
    table_annotated_dir = os.path.join(output_dir, "table_annotated")
    table_annotated_fragments_dir = os.path.join(
        output_dir, "table_annotated", "fragments"
    )

    # specific to the annotator that is being used
    annotator_output_dir = None
    if annotator_name:
        annotator_output_dir = ensure_exists(os.path.join(output_dir, annotator_name))

    ensure_exists(htables_output_dir)
    ensure_exists(table_annotated_dir)
    ensure_exists(table_annotated_fragments_dir)

    return (
        htables_output_dir,
        table_src_dir,
        table_annotated_dir,
        table_annotated_fragments_dir,
        annotator_output_dir,
    )
