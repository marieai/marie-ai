import os
import warnings
from typing import Any, Optional

import torch
from docarray import DocList

from marie import requests, safely_encoded
from marie.api import value_from_payload_or_args
from marie.api.docs import AssetKeyDoc, StorageDoc
from marie.boxes import PSMode
from marie.executor.marie_executor import MarieExecutor
from marie.executor.mixin import StorageMixin
from marie.logging_core.logger import MarieLogger
from marie.logging_core.mdc import MDC
from marie.logging_core.predefined import default_logger as logger
from marie.models.utils import (
    initialize_device_settings,
    setup_torch_optimizations,
    torch_gc,
)
from marie.ocr import CoordinateFormat
from marie.pipe.components import s3_asset_path, store_assets, update_existing_meta
from marie.storage import StorageManager
from marie.utils.docs import docs_from_asset, frames_from_docs
from marie.utils.image_utils import ensure_max_page_size, hash_frames_fast
from marie.utils.json import load_json_file, store_json_object
from marie.utils.network import get_ip_address
from marie.utils.utils import ensure_exists


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

    def extract_base_parameters(
        self, parameters: dict
    ) -> tuple[str, str, str, str, dict]:
        """
        Extract common parameters and payload for pipeline execution.

        :param parameters: Dictionary of request parameters.
        :returns: Tuple containing:
            - job_id (str): Unique identifier for the job.
            - ref_id (str): Reference identifier for the document.
            - ref_type (str): Type/category of the document (defaults to "not_defined").
            - queue_id (str): Queue identifier (defaults to "0000-0000-0000-0000").
            - payload (dict): The extracted payload dictionary.
        :rtype: tuple[str, str, str, str, dict]
        :raises ValueError: If `job_id`, `ref_id`, or `payload` is missing.
        """
        if parameters is None or "job_id" not in parameters:
            self.logger.error(f"Job ID is not present in parameters")
            raise ValueError("Job ID is not present in parameters")

        job_id = parameters.get("job_id", "0000-0000-0000-0000")
        MDC.put("request_id", job_id)

        self.logger.info("Parsing Parameters")
        for key, value in parameters.items():
            self.logger.info("The value of {} is {}".format(key, value))

        ref_id = parameters.get("ref_id")
        if ref_id is None:
            raise ValueError("ref_id is not present in parameters")
        ref_type = parameters.get("ref_type", "not_defined")
        queue_id: str = parameters.get("queue_id", "0000-0000-0000-0000")

        payload = parameters.get("payload")
        if payload is None:
            self.logger.error("Empty Payload")
            raise ValueError("Empty Payload")

        return job_id, ref_id, ref_type, queue_id, payload

    def get_frames_from_docs(self, docs: DocList[AssetKeyDoc]):
        """
        Load and preprocess frames from a single document asset.

        :param docs: DocList containing exactly one AssetKeyDoc.
        :raises ValueError: If no or multiple documents are provided.
        :return: List of image frames (e.g., numpy arrays).
        """
        if len(docs) == 0:
            raise ValueError("Expected single document. No documents found")
        if len(docs) > 1:
            raise ValueError("Expected single document. Multiple documents found.")

        doc = docs[0]
        self.logger.debug(
            f"Load documents from specified document asset key: {doc.asset_key}"
        )
        docs = docs_from_asset(doc.asset_key, doc.pages)

        src_frames = frames_from_docs(docs)
        changed, frames = ensure_max_page_size(src_frames)
        if changed:
            self.logger.warning(f"Page size of frames was changed ")
            for i, (s, f) in enumerate(zip(src_frames, frames)):
                self.logger.warning(f"Frame[{i}] changed : {s.shape} -> {f.shape}")

        return frames

    def run_pipeline(
        self, pipeline, docs: DocList[AssetKeyDoc], parameters: dict, *args, **kwargs
    ):
        """
        Execute the provided Pipeline on a single document asset.

        :param pipeline: Pipeline instance to execute.
        :param docs: DocList containing a single AssetKeyDoc.
        :param parameters: Dictionary of request parameters including payload.
        :raises ValueError: If pipeline execution fails.
        :return: Safely encoded response bytes containing status, runtime_info, and metadata.
        """
        job_id, ref_id, ref_type, queue_id, payload = self.extract_base_parameters(
            parameters
        )
        frames = self.get_frames_from_docs(docs)

        # https://github.com/marieai/marie-ai/issues/51
        regions = payload.get("regions", [])
        for region in regions:
            region["id"] = f'{int(region["id"])}'
            region["x"] = int(region["x"])
            region["y"] = int(region["y"])
            region["w"] = int(region["w"])
            region["h"] = int(region["h"])
            region["pageIndex"] = int(region["pageIndex"])

        # due to compatibility issues with other frameworks we allow passing same arguments in the 'args' object
        coordinate_format = CoordinateFormat.from_value(
            value_from_payload_or_args(payload, "format", default="xywh")
        )
        pms_mode = PSMode.from_value(
            value_from_payload_or_args(payload, "mode", default="")
        )

        self.logger.debug(
            "ref_id, ref_type frames , regions , pms_mode, coordinate_format, checksum: "
            f"{ref_id}, {ref_type},  {len(frames)}, {len(regions)}, {pms_mode}, {coordinate_format}"
        )

        self.logger.info("Extracting Runtime Config from features list")
        runtime_conf = {}
        pipeline_names = [
            conf["pipeline"]["name"] for conf in pipeline.pipelines_config
        ]
        for feature in payload.get("features", []):
            if feature.get("type") != "pipeline":
                continue
            name = feature.get("name")
            if name and any(name == p_name for p_name in pipeline_names):
                runtime_conf = feature
        self.logger.debug(f"Resolved Runtime Config: {runtime_conf}")

        try:
            metadata = pipeline.execute(
                ref_id=ref_id,
                ref_type=ref_type,
                frames=frames,
                pms_mode=pms_mode,
                coordinate_format=coordinate_format,
                regions=regions,
                queue_id=queue_id,
                job_id=job_id,
                runtime_conf=runtime_conf,
            )
        except BaseException as error:
            self.logger.error(f"Pipeline error : {error}", exc_info=True)
            torch_gc()
            MDC.remove("request_id")
            raise error

        if metadata is None:
            self.logger.error(f"Metadata is None, this should not happen")
            raise ValueError("Pipeline Execution Error: Metadata is None")

        # NOTE: see todo on self.persist function
        # self.persist(ref_id, ref_type, metadata)

        include_ocr = value_from_payload_or_args(payload, "return_ocr", default=False)
        # strip out ocr results from metadata
        if not include_ocr and "ocr" in metadata:
            del metadata["ocr"]
        del frames
        del regions

        torch_gc()
        MDC.remove("request_id")

        response = {
            "status": "success",
            "runtime_info": self.runtime_info,
            "metadata": metadata,
        }
        converted = safely_encoded(lambda x: x)(response)
        return converted

    # TODO: Persist only what is needed. This function is currently not doing anything on self.store
    def persist(self, ref_id: str, ref_type: str, results: Any) -> None:
        """Persist results"""

        def _tags(index: int, ftype: str, checksum: str):
            return {
                "action": "classifier",
                "index": index,
                "type": ftype,
                "ttl": 48 * 60,
                "checksum": checksum,
                "runtime": self.runtime_info,
            }

        if self.storage_enabled:
            # frame_checksum = hash_frames_fast(frames=[frame])

            docs = DocList[StorageDoc](
                [
                    StorageDoc(
                        content=results,
                        tags=_tags(-1, "metadata", ref_id),
                    )
                ]
            )

            self.store(
                ref_id=ref_id,
                ref_type=ref_type,
                store_mode="content",
                docs=docs,
            )

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
        job_id, ref_id, ref_type, _, payload = self.extract_base_parameters(parameters)
        frames = self.get_frames_from_docs(docs)

        # create local asset directory
        frame_checksum = hash_frames_fast(frames=frames)
        root_asset_dir = ensure_exists(
            os.path.join("/tmp/generators", frame_checksum, job_id)
        )

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
            pipeline_meta = load_json_file(pipeline_meta_path, True)
            metadata = update_existing_meta(metadata, pipeline_meta)
        metadata["pipeline"] = ",".join(meta_folders)
        metadata["pages"] = len(frames)

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
