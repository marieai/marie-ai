import os
from typing import Any

from docarray import DocList

from marie import requests
from marie.api.docs import AssetKeyDoc
from marie.executor.asset_util import create_working_dir
from marie.executor.marie_executor import MarieExecutor
from marie.executor.request_util import (
    get_frames_from_docs,
    get_payload_features,
    parse_parameters,
)
from marie.logging_core.logger import MarieLogger
from marie.logging_core.predefined import default_logger as logger
from marie.pipe.components import restore_assets, store_assets, update_existing_meta
from marie.storage import StorageManager
from marie.utils.json import load_json_file, store_json_object
from marie.utils.network import get_ip_address


class MetaMergeExecutor(MarieExecutor):
    """Maire Executor for meta file merging"""

    def __init__(
        self,
        name: str = "",
        storage: dict[str, Any] = None,
        **kwargs,
    ):
        """
        Initialize the MetaMergeExecutor.

        :param name: Optional name for the executor instance.
        :param storage: Storage configuration dictionary (e.g., Postgres settings).
        :param kwargs: Additional keyword arguments passed to MarieExecutor.
        """
        kwargs['storage'] = storage
        super().__init__(**kwargs)
        self.logger = MarieLogger(
            getattr(self.metas, "name", self.__class__.__name__)
        ).logger

        self.logger.info(f"Starting executor : {self.__class__.__name__}")
        self.logger.info(f"Storage config: {storage}")
        self.logger.info(f"Kwargs : {kwargs}")

        instance_name = "not_defined"
        if kwargs is not None:
            instance_name = kwargs.get("runtime_args", {}).get("name", "not_defined")

        self.runtime_info = {
            "name": self.__class__.__name__,
            "instance_name": instance_name,
            "model": "",
            "host": get_ip_address(),
            "workspace": self.workspace,
        }

        connected = StorageManager.ensure_connection("s3://", silence_exceptions=False)
        logger.warning(f"S3 connection status : {connected}")

    @requests(on="/merge/metadata")
    def merge_metadata(
        self, docs: DocList[AssetKeyDoc], parameters: dict, *args, **kwargs
    ):
        """
        Handles merging of metadata with the provided document list and parameters. This
        method is designed to collect and consolidate metadata from defined pipeline
        features. It performs the operation by parsing input parameters, retrieving
        necessary frames and assets, and integrating metadata files from specified
        pipelines. The resulting merged metadata is stored and prepared for subsequent
        operations.

        Parameters:
            docs: List of documents containing asset keys required for metadata processing.
            parameters: Dictionary of parameters used to control metadata merging behavior.
            *args: Additional positional arguments for extensibility, if required.
            **kwargs: Additional keyword arguments for extensibility, if required.

        Returns:
            dict: A dictionary containing the status of the metadata merging operation as
            well as runtime information and details of stored assets.

        Raises:
            ConnectionError: If the metadata collection operation fails due to connection
            issues.
            FileNotFoundError: If metadata for a specified pipeline is not found in the
            storage.
        """
        job_id, ref_id, ref_type, _, payload = parse_parameters(parameters)
        frames = get_frames_from_docs(docs)
        root_asset_dir = create_working_dir(frames)

        features = get_payload_features(payload, f_type="pipeline")
        meta_folders = [
            str(feature.get("name")) for feature in features if feature.get("name")
        ]

        if not meta_folders:
            self.logger.warning("No pipelines defined in features.")
            return {
                "status": "failed",
                "message": "No pipelines defined in features. Nothing to merge",
            }

        self.logger.info(f"Collecting meta data from pipelines: {meta_folders}")
        s3_root_path = restore_assets(
            ref_id, ref_type, root_asset_dir, meta_folders, overwrite=True
        )
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
