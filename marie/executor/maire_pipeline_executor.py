import os
import warnings
from typing import Any, Optional

import torch
from docarray import DocList

from marie import safely_encoded
from marie.api import AssetKeyDoc, value_from_payload_or_args
from marie.api.docs import StorageDoc
from marie.boxes import PSMode
from marie.executor.marie_executor import MarieExecutor
from marie.executor.mixin import StorageMixin
from marie.executor.request_util import (
    get_frames_from_docs,
    get_payload_features,
    parse_parameters,
)
from marie.logging_core.logger import MarieLogger
from marie.logging_core.mdc import MDC
from marie.logging_core.predefined import default_logger as logger
from marie.models.utils import (
    initialize_device_settings,
    setup_torch_optimizations,
    torch_gc,
)
from marie.ocr import CoordinateFormat
from marie.storage import StorageManager
from marie.utils.image_utils import hash_frames_fast
from marie.utils.network import get_ip_address


class PipelineExecutor(MarieExecutor, StorageMixin):
    """
    Handles execution of a scalable and configurable pipeline with runtime-specific
    optimizations, dependency setups, and device management.

    This class is designed to handle preprocessing, computational configurations,
    storage solution integrations, and pipeline execution in highly dynamic environments.
    It ensures efficient utilization of available resources while providing
    extensibility for managing complex workflows and settings through parameters.

    Attributes:
        logger: Logger instance used to log runtime information and issues.
        device: Computation device being used for execution, such as 'cpu' or 'cuda'.
        has_cuda: Boolean indicating whether CUDA/GPU is being used.
        runtime_info: Dictionary containing various runtime-related properties such as
            device type, workspace, and other settings.
        storage_enabled: Boolean indicating whether storage mechanisms are enabled.
    """

    def __init__(
        self,
        name: str = "",
        device: Optional[str] = None,
        num_worker_preprocess: int = 4,
        storage: dict[str, Any] = None,
        **kwargs,
    ):
        """
        Initializes the executor with specified configuration and settings.

        Summary:
        The __init__ method sets up the executor with configurable runtime settings,
        including logging, computational device selection, preprocessing workers,
        and optional storage configurations. It ensures that dependencies and
        runtime optimizations are appropriately managed to maximize performance
        based on the provided parameters.

        Resolves errors if CUDA/GPU is unavailable but requested, optimizes
        device and threading configurations for performance, and issues
        warnings for potentially sub-optimal setups.

        Arguments:
            name (str, optional): The name of the executor. Default is an empty string.
            device (Optional[str], optional): Specifies the computation device ('cuda'
                for GPU or 'cpu'). If not provided, defaults to CPU.
            num_worker_preprocess (int, optional): Number of workers to use for
                preprocessing tasks. Defaults to 4.
            storage (dict[str, Any], optional): Storage configuration dictionary
                for managing persistence and hosting settings.
            **kwargs: Additional parameters passed to the base class or used in
                the executor setup.
        """
        kwargs['storage'] = storage
        super().__init__(**kwargs)
        self.logger = MarieLogger(
            getattr(self.metas, "name", self.__class__.__name__)
        ).logger

        self.logger.info(f"Starting executor : {self.__class__.__name__}")
        self.logger.info(f"Runtime args : {kwargs.get('runtime_args')}")
        self.logger.info(f"Storage config: {storage}")
        self.logger.info(f"Device : {device or 'cpu'}")
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

        self.has_cuda = True if self.device.type.startswith("cuda") else False

        self.runtime_info = {
            "name": self.__class__.__name__,
            "instance_name": instance_name,
            "model": "",
            "host": get_ip_address(),
            "workspace": self.workspace,
            "use_cuda": self.has_cuda,
        }

        self.storage_enabled = False
        if storage is not None and "psql" in storage:
            sconf = storage["psql"]
            self.setup_storage(sconf.get("enabled", False), sconf)

        connected = StorageManager.ensure_connection("s3://", silence_exceptions=False)
        logger.warning(f"S3 connection status : {connected}")

    def parse_params_and_execute(
        self,
        docs: DocList[AssetKeyDoc],
        parameters: dict,
        pipeline,
        default_ref_type="extract",
        **kwargs,
    ):
        """
        Executes a pipeline after parsing the provided parameters and updating any missing
        values with defaults. It supports handling additional keyword arguments to allow
        flexibility during execution.

        Parameters:
            docs (DocList[AssetKeyDoc]): The list of documents to process.
            parameters (dict): A dictionary containing the parameters to parse.
            pipeline: The pipeline object to execute after parameter parsing.
            default_ref_type (str): The default reference type to use if none is provided.
            **kwargs: Additional optional keyword arguments for the pipeline execution.

        Returns:
            The result of the pipeline execution with the parsed and updated parameters.
        """
        job_id, ref_id, ref_type, queue_id, payload = parse_parameters(parameters)
        ref_type = default_ref_type if ref_type is None else ref_type
        return self.execute_pipeline(
            docs, pipeline, job_id, ref_id, payload, ref_type, queue_id, **kwargs
        )

    def execute_pipeline(
        self,
        docs: DocList[AssetKeyDoc],
        pipeline,
        job_id: str,
        ref_id: str,
        payload: Any,
        ref_type: str = "extract",
        queue_id: str = "0000-0000-0000-0000",
        **kwargs,
    ):
        """
        Executes a processing pipeline for the provided documents and configuration, handling
        various settings and runtime augmentations.

        Parameters:
            docs (DocList[AssetKeyDoc]): List of documents to process in the pipeline.
            pipeline: The processing pipeline object to apply to the documents.
            job_id (str): Unique identifier for the OCR or processing request.
            ref_id (str): Reference identifier for the processing task or input.
            payload (Any): Data payload containing configuration details like regions, modes, and formats.
            ref_type (str): Type of the reference being processed. Defaults to "extract".
            queue_id (str): Identifier for the execution queue. Defaults to "0000-0000-0000-0000".
            **kwargs: Additional optional keyword arguments passed to the pipeline execution.

        Raises:
            ValueError: If runtime configuration cannot be resolved for the pipeline.

        Returns:
            Any: Encoded response object indicating the status of the operation and optional
            metadata.
        """
        MDC.put("request_id", job_id)
        self.logger.info("Starting Pipeline Request")

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

        include_ocr = value_from_payload_or_args(payload, "return_ocr", default=False)

        runtime_conf = {}
        pipeline_features = get_payload_features(payload, f_type="pipeline")
        if len(pipeline_features) == 1:
            runtime_conf = pipeline_features[0]
        else:
            pipeline_name = getattr(pipeline, "pipeline_name", "default")
            filtered_features = [
                f for f in pipeline_features if f.get("name") == pipeline_name
            ]
            if len(pipeline_features) == 1:
                runtime_conf = filtered_features[0]
            elif len(filtered_features) > 1:
                self.logger.error(
                    f"Unable to distinguish Runtime Config : {filtered_features}"
                )
                raise ValueError(f"Cannot Resolve Pipeline Runtime Config")

        pages = runtime_conf.get("pages", None)
        if isinstance(pages, str):
            pages = sorted({int(n) for n in pages.split(",")})
        elif isinstance(pages, list):
            pages = sorted({int(n) for n in pages})
        elif pages is not None:
            self.logger.warning(f"Unexpected pages attr {pages}, ignoring")
            pages = None

        frames = get_frames_from_docs(docs, pages)
        ref_id = hash_frames_fast(frames) if ref_id is None else ref_id

        self.logger.info(
            "ref_id, ref_type frames , regions , pms_mode, coordinate_format,"
            f" checksum: {ref_id}, {ref_type},  {len(frames)}, {len(regions)}, {pms_mode},"
            f" {coordinate_format}"
        )

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
                **kwargs,
            )

            if metadata is None:
                self.logger.warning(
                    f"Metadata is None, this can happen if no text was found"
                )
                response = {
                    "status": "failed",
                    "runtime_info": self.runtime_info,
                    "metadata": {},
                }
                converted = safely_encoded(lambda x: x)(response)
                return converted

            del frames
            del regions

            self.persist(ref_id, ref_type, metadata)

            # strip out ocr results from metadata
            if not include_ocr and "ocr" in metadata:
                del metadata["ocr"]

            response = {
                "status": "succeeded",
                "runtime_info": self.runtime_info,
                # "metadata": metadata,
            }
            converted = safely_encoded(lambda x: x)(response)
            return converted
        except BaseException as error:
            self.logger.error(f"Extract error : {error}", exc_info=True)
            msg = "inference exception"
            if self.show_error:
                msg = (str(error),)
            return {
                "status": "error",
                "runtime_info": self.runtime_info,
                "error": msg,
            }
        finally:
            torch_gc()
            MDC.remove("request_id")

    def persist(self, ref_id: str, ref_type: str, results: Any) -> None:
        """
        Persists extracted results into storage if storage is enabled.

        The method generates metadata tags and packages the provided results into a
        storage document for persistence. This process is only executed when storage
        is enabled.

        Parameters:
            ref_id (str): A unique identifier reference.
            ref_type (str): The type associated with the reference ID.
            results (Any): The results data to be persisted.

        Returns:
            None
        """

        def _tags(index: int, ftype: str, checksum: str):
            return {
                "action": "extract",
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
