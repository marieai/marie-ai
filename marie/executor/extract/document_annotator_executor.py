from typing import Optional, Union

import torch
from docarray import DocList

from marie import requests
from marie.api.docs import AssetKeyDoc
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
from marie.utils.docs import docs_from_asset, frames_from_docs
from marie.utils.image_utils import ensure_max_page_size
from marie.utils.network import get_ip_address


class DocumentAnnotatorExecutor(MarieExecutor, StorageMixin):
    """Executor for document annotation"""

    def __init__(
        self,
        name: str = "",
        device: Optional[str] = None,
        num_worker_preprocess: int = 4,
        storage: dict[str, any] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.logger = MarieLogger(
            getattr(self.metas, "name", self.__class__.__name__)
        ).logger

        logger.info(f"Starting executor : {self.__class__.__name__}")
        logger.info(f"Runtime args : {kwargs.get('runtime_args')}")
        logger.info(f"Storage config: {storage}")
        logger.info(f"Device : {device}")
        logger.info(f"Num worker preprocess : {num_worker_preprocess}")
        logger.info(f"Kwargs : {kwargs}")

        self.show_error = True  # show prediction errors
        # sometimes we have CUDA/GPU support but want to only use CPU
        resolved_devices, _ = initialize_device_settings(
            devices=[device], use_cuda=True, multi_gpu=False
        )
        if len(resolved_devices) > 1:
            self.logger.warning(
                "Multiple devices are not supported in %s inference, using the first device %s.",
                self.__class__.__name__,
                resolved_devices[0],
            )
        self.device = resolved_devices[0]
        has_cuda = True if self.device.type.startswith("cuda") else False

        setup_torch_optimizations()
        instance_name = "not_defined"
        if kwargs is not None:
            if "runtime_args" in kwargs:
                instance_name = kwargs.get("runtime_args").get("name", "not_defined")

        self.runtime_info = {
            "name": self.__class__.__name__,
            "instance_name": instance_name,
            "model": "",
            "host": get_ip_address(),
            "workspace": self.workspace,
            "use_cuda": has_cuda,
        }

        self.storage_enabled = False
        if storage is not None and "psql" in storage:
            sconf = storage["psql"]
            self.setup_storage(sconf.get("enabled", False), sconf)

    @requests(on="/default")
    def default(self, parameters, **kwargs):
        return {"valid": True}

    @requests(on="/annotator/roi")
    def annotator_roi(
        self, docs: DocList[AssetKeyDoc], parameters: dict, *args, **kwargs
    ):
        """
        Document annotator executor

        EXAMPLE USAGE

            As Executor

            .. code-block:: python

                exec = AnnotatorExecutor()

        :param parameters:
        :param docs: Documents to process
        :param kwargs:
        :return:
        """
        if len(docs) == 0:
            return {"error": "empty payload"}
        if len(docs) > 1:
            return {"error": "expected single document"}

        # load documents from specified document asset key
        doc = docs[0]
        docs = docs_from_asset(doc.asset_key, doc.pages)

        src_frames = frames_from_docs(docs)
        changed, frames = ensure_max_page_size(src_frames, max_page_size=(2500, 3000))

        if changed:
            self.logger.warning(f"Page size of frames was changed ")
            for i, (s, f) in enumerate(zip(src_frames, frames)):
                self.logger.warning(f"Frame[{i}] changed : {s.shape} -> {f.shape}")

        if parameters is None or "job_id" not in parameters:
            self.logger.warning(f"Job ID is not present in parameters")
            raise ValueError("Job ID is not present in parameters")

        job_id = parameters.get("job_id")
        MDC.put("request_id", job_id)

        self.logger.info("Starting Annotator processing request")
        for key, value in parameters.items():
            self.logger.info("The value of {} is {}".format(key, value))

        try:
            response = {
                "status": "success",
                "runtime_info": self.runtime_info,
                "error": None,
            }
            return response
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
