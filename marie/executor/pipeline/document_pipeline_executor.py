import os
import warnings
from typing import Optional

import torch
from docarray import DocList

from marie import requests
from marie.api import get_frames_from_docs, parse_parameters
from marie.api.docs import AssetKeyDoc
from marie.executor.extract.util import create_working_dir
from marie.executor.marie_executor import MarieExecutor
from marie.executor.mixin import StorageMixin
from marie.logging_core.logger import MarieLogger
from marie.logging_core.predefined import default_logger as logger
from marie.models.utils import initialize_device_settings, setup_torch_optimizations
from marie.pipe.components import s3_asset_path, store_assets, update_existing_meta
from marie.storage import StorageManager
from marie.utils.json import load_json_file, store_json_object
from marie.utils.network import get_ip_address


class PipelineExecutor(MarieExecutor, StorageMixin):
    """Executor for pipeline document processing"""

    def __init__(
        self,
        name: str = "",
        device: Optional[str] = None,
        num_worker_preprocess: int = 4,
        storage: dict[str, any] = None,
        **kwargs,
    ):
        """
        Initialize the PipelineExecutor.

        :param name: Optional name for the executor instance.
        :param device: Device identifier for computation (e.g., 'cuda', 'cpu').
        :param num_worker_preprocess: Number of preprocessing worker threads.
        :param storage: Storage configuration dictionary (e.g., Postgres settings).
        :param kwargs: Additional keyword arguments passed to MarieExecutor.
        """
        kwargs['storage'] = storage
        super().__init__(**kwargs)
        self.logger = MarieLogger(
            getattr(self.metas, "name", self.__class__.__name__)
        ).logger

        self.logger.info(f"Starting executor : {self.__class__.__name__}")
        self.logger.info(f"Runtime args : {kwargs.get('runtime_args')}")
        self.logger.info(f"Storage config: {storage}")
        self.logger.info(f"Device : {device}")
        self.logger.info(f"Num worker preprocess : {num_worker_preprocess}")
        self.logger.info(f"Kwargs : {kwargs}")

        self.show_error = True  # show prediction errors
        # sometimes we have CUDA/GPU support but want to only use CPU
        use_cuda = True if torch.cuda.is_available() and device == "cuda" else False
        resolved_devices, _ = initialize_device_settings(
            devices=[device], use_cuda=use_cuda, multi_gpu=False
        )
        if len(resolved_devices) > 1:
            logger.warning(
                "Multiple devices are not supported in %s inference, using the first device %s.",
                self.__class__.__name__,
                resolved_devices[0],
            )
        self.device = resolved_devices[0]
        num_threads = max(1, torch.get_num_threads())
        if not self.device.type.startswith("cuda") and (
            "OMP_NUM_THREADS" not in os.environ
            and hasattr(self.runtime_args, "replicas")
        ):
            replicas = getattr(self.runtime_args, "replicas", 1)
            num_threads = max(1, torch.get_num_threads() // replicas)

            if num_threads < 2:
                warnings.warn(
                    f"Too many replicas ({replicas}) vs too few threads {num_threads} may result in "
                    f"sub-optimal performance."
                )

            # NOTE: make sure to set the threads right after the torch import,
            # and `torch.set_num_threads` always take precedence over environment variables `OMP_NUM_THREADS`.
            # For more details, please see https://pytorch.org/docs/stable/generated/torch.set_num_threads.html
            torch.set_num_threads(max(num_threads, 1))
            torch.set_num_interop_threads(1)

        setup_torch_optimizations(num_threads=num_threads)

        instance_name = "not_defined"
        if kwargs is not None:
            instance_name = kwargs.get("runtime_args", {}).get("name", "not_defined")

        self.runtime_info = {
            "name": self.__class__.__name__,
            "instance_name": instance_name,
            "model": "",
            "host": get_ip_address(),
            "workspace": self.workspace,
            "use_cuda": True if self.device.type.startswith("cuda") else False,
        }

        self.storage_enabled = False
        if storage is not None and "psql" in storage:
            sconf = storage["psql"]
            self.setup_storage(sconf.get("enabled", False), sconf)

        connected = StorageManager.ensure_connection("s3://", silence_exceptions=False)
        logger.warning(f"S3 connection status : {connected}")

    @requests(on="/merge/metadata")
    def merge_metadata(
        self, docs: DocList[AssetKeyDoc], parameters: dict, *args, **kwargs
    ):
        """
        Merge metadata from multiple pipeline executions for a document.

        :param docs: DocList containing a single AssetKeyDoc.
        :param parameters: Dictionary of request parameters including payload.
        :returns: Dictionary with merge status, runtime_info, and stored assets.
        :raises ConnectionError: If unable to fetch existing assets.
        """
        job_id, ref_id, ref_type, _, payload = parse_parameters(parameters)
        frames = get_frames_from_docs(docs)
        root_asset_dir = create_working_dir(frames)

        features = payload.get("features", [])
        meta_folders = [
            str(feature.get("name"))
            for feature in features
            if feature.get("type") == "pipeline" and feature.get("name")
        ]

        if not meta_folders:
            self.logger.warning("No pipelines defined in features.")
            return {
                "status": "failed",
                "message": "No pipelines defined in features. Nothing to merge",
            }

        self.logger.info(f"Collecting meta data from pipelines: {meta_folders}")
        s3_root_path = fetch_assets(ref_id, ref_type, root_asset_dir, meta_folders)
        if s3_root_path is None:
            raise ConnectionError("Unable to collect meta data from")

        meta_filename = f"{ref_id}.meta.json"
        meta_path = os.path.join(root_asset_dir, meta_filename)
        metadata = {}
        for meta_folder in meta_folders:
            pipeline_meta_path = os.path.join(
                root_asset_dir, meta_folder, meta_filename
            )
            if not os.path.exists(pipeline_meta_path):
                raise FileNotFoundError(
                    f"Meta data for {meta_folder} not found in remote or local storage."
                )
            pipeline_meta = load_json_file(pipeline_meta_path, True)
            metadata = update_existing_meta(metadata, pipeline_meta)
        metadata["pipeline"] = ",".join(meta_folders)

        self.logger.info(f"Storing merged metadata : {meta_path}")
        store_json_object(metadata, meta_path)
        stored_assets = store_assets(
            ref_id, ref_type, root_asset_dir, match_wildcard="*.meta.json"
        )

        return {
            "status": "success",
            "runtime_info": self.runtime_info,
            "assets": stored_assets,
        }


# TODO: refactor marie.pipe.components.restore_assets to do this job
def fetch_assets(
    ref_id: str,
    ref_type: str,
    root_asset_dir: str,
    dirs_to_fetch: list,
    full_restore=False,
) -> str or None:
    """
    Fetch assets from primary storage (S3) into root asset directory. This pulls the
    assets from the last run of the pipeline.

    :param ref_id: document reference id (e.g. filename)
    :param ref_type: document reference type(e.g. document, page, process)
    :param root_asset_dir: root asset directory
    :param dirs_to_fetch: a subset of dirs to restore
    :param full_restore: if True, restore all assets, otherwise only restore the dirs_to_fetch
    that are required for the extract pipeline.
    :return:
    """
    s3_root_path = s3_asset_path(ref_id, ref_type)
    connected = StorageManager.ensure_connection("s3://", silence_exceptions=True)
    if not connected:
        logger.error(f"Error fetching assets : Could not connect to S3")
        return None

    logger.info(f"Restoring assets from {s3_root_path} to {root_asset_dir}")

    if full_restore:
        try:
            StorageManager.copy_remote(
                s3_root_path,
                root_asset_dir,
                match_wildcard="*",
                overwrite=True,
            )
        except Exception as e:
            logger.error(f"Error fetching all assets : {e}")
    else:
        for dir_to_fetch in dirs_to_fetch:
            try:
                StorageManager.copy_remote(
                    s3_root_path,
                    root_asset_dir,
                    match_wildcard=f"*/{dir_to_fetch}/*",
                    overwrite=True,
                )
            except Exception as e:
                logger.error(f"Error fetching assets from {dir_to_fetch} : {e}")
    return s3_root_path
