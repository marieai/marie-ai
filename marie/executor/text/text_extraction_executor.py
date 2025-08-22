import os
import time
import warnings
from typing import Any, Optional, Union

import numpy as np
import torch
from docarray import DocList

from marie import Executor, requests, safely_encoded
from marie.api import (
    get_frames_from_docs,
    get_payload_features,
    parse_parameters,
    value_from_payload_or_args,
)
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
from marie.pipe import ExtractPipeline
from marie.storage import StorageManager
from marie.utils.docs import docs_from_asset, frames_from_docs
from marie.utils.image_utils import hash_frames_fast
from marie.utils.network import get_ip_address
from marie.utils.types import strtobool


class TextExtractionExecutor(MarieExecutor, StorageMixin):
    """
    Executor for extracting text.
    Text extraction can either be executed out over the entire image or over selected regions of interests (ROIs)
    aka bounding boxes.
    """

    def __init__(
        self,
        name: str = "",
        device: Optional[str] = None,
        num_worker_preprocess: int = 4,
        storage: dict[str, Any] = None,
        pipeline: dict[str, Any] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
        **kwargs,
    ):
        kwargs['storage'] = storage
        super().__init__(**kwargs)
        self.logger = MarieLogger(
            getattr(self.metas, "name", self.__class__.__name__)
        ).logger

        # # sometimes we have CUDA/GPU support but want to only use CPU
        use_cuda = not strtobool(os.environ.get("MARIE_DISABLE_CUDA", "false"))

        if not device:
            device = torch.device(
                "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
            )
            logger.warning(f"Device is not specified, using {device} as default device")

        logger.info(f"Starting executor : {self.__class__.__name__}")
        logger.info(f"Runtime args : {kwargs.get('runtime_args')}")
        logger.info(f"Storage config: {storage}")
        logger.info(f"Pipeline config: {pipeline}")
        logger.info(f"Device : {device}")
        logger.info(f"use_cuda : {use_cuda}")
        logger.info(f"Num worker preprocess : {num_worker_preprocess}")
        logger.info(f"Kwargs : {kwargs}")

        resolved_devices, _ = initialize_device_settings(
            devices=[device], use_cuda=use_cuda, multi_gpu=False
        )
        if len(resolved_devices) > 1:
            self.logger.warning(
                "Multiple devices are not supported in %s inference, using the first device %s.",
                self.__class__.__name__,
                resolved_devices[0],
            )
        self.device = resolved_devices[0]
        self.has_cuda = True if self.device.type.startswith("cuda") else False

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
        self.show_error = True  # show prediction errors
        # self.pipeline = ExtractPipeline(pipeline_config=pipeline, cuda=has_cuda)

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
            "use_cuda": self.has_cuda,
        }

        self.storage_enabled = False
        if storage is not None and "psql" in storage:
            sconf = storage["psql"]
            self.setup_storage(sconf.get("enabled", False), sconf)

        connected = StorageManager.ensure_connection("s3://", silence_exceptions=False)
        logger.warning(f"S3 connection status : {connected}")

        self.pipeline = ExtractPipeline(pipeline_config=pipeline, cuda=self.has_cuda)

    @requests(on="/document/extract")
    # @safely_encoded # BREAKS WITH docarray 0.39 as it turns this into a LegacyDocument which is not supported
    def extract(self, docs: DocList[AssetKeyDoc], parameters: dict, *args, **kwargs):

        job_id, ref_id, ref_type, queue_id, payload = parse_parameters(parameters)
        frames = get_frames_from_docs(docs)
        ref_id = hash_frames_fast(frames) if ref_id is None else ref_id
        ref_type = "extract" if ref_type is None else ref_type

        MDC.put("request_id", job_id)

        self.logger.info("Starting OCR request")
        for key, value in parameters.items():
            self.logger.info("The value of {} is {}".format(key, value))

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

        self.logger.debug(
            "ref_id, ref_type frames , regions , pms_mode, coordinate_format,"
            f" checksum: {ref_id}, {ref_type},  {len(frames)}, {len(regions)}, {pms_mode},"
            f" {coordinate_format}"
        )

        runtime_conf = {}
        pipeline_features = get_payload_features(payload, f_type="pipeline")
        if len(pipeline_features) == 1:
            runtime_conf = pipeline_features[0]
        else:
            filtered_features = [
                f
                for f in pipeline_features
                if f.get("name") == self.pipeline.pipeline_name
            ]
            if len(pipeline_features) == 1:
                runtime_conf = filtered_features[0]
            elif len(filtered_features) > 1:
                self.logger.error(
                    f"Unable to distinguish Runtime Config : {filtered_features}"
                )
                raise ValueError(f"Cannot Resolve Pipeline Runtime Config")

        try:
            metadata = self.pipeline.execute(
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

    @requests(on="/document/extract/status")
    def status(self, parameters, **kwargs):
        use_cuda = torch.cuda.is_available()
        return {"index": "complete", "use_cuda": use_cuda}

    @requests(on="/document/extract/validate")
    def validate(self, parameters, **kwargs):
        return {"valid": True}

    @requests(on="/default")
    def default(self, parameters, **kwargs):
        return {"valid": True}

    def persist(self, ref_id: str, ref_type: str, results: Any) -> None:
        """Persist results"""

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


class TextExtractionExecutorMock(MarieExecutor):
    def __init__(
        self,
        name: str = "",
        device: Optional[str] = None,
        num_worker_preprocess: int = 4,
        pipeline: Optional[dict[str, any]] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
        **kwargs,
    ):
        """
        :param device: 'cpu' or 'cuda'. Default is None, which auto-detects the device.
        :param num_worker_preprocess: The number of CPU workers to preprocess images and texts. Default is 4.
        :param minibatch_size: The size of the minibatch for preprocessing and encoding. Default is 32. Reduce this
            number if you encounter OOM errors.
        :param dtype: inference data type, if None defaults to torch.float32 if device == 'cpu' else torch.float16.
        """
        super().__init__(**kwargs)

        logger.info(f"Starting mock executor : {time.time()}")
        logger.info(f"Starting executor : {self.__class__.__name__}")
        logger.info(f"Runtime args : {kwargs.get('runtime_args')}")
        logger.info(f"Pipeline config: {pipeline}")
        logger.info(f"Device : {device}")
        logger.info(f"Num worker preprocess : {num_worker_preprocess}")
        logger.info(f"Kwargs : {kwargs}")

        setup_torch_optimizations()

        self.show_error = True  # show prediction errors
        # sometimes we have CUDA/GPU support but want to only use CPU
        use_cuda = torch.cuda.is_available()
        if os.environ.get("MARIE_DISABLE_CUDA"):
            use_cuda = False
        self.logger = MarieLogger(context=self.__class__.__name__)

        if not device:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not use_cuda:
            device = "cpu"
        self.device = device

        self.runtime_info = {
            "name": self.__class__.__name__,
            "instance_name": kwargs.get("runtime_args", {}).get("name", "not_defined"),
            "model": "",
            "host": get_ip_address(),
            "workspace": self.workspace,
            "use_cuda": use_cuda,
            "device": self.device.__str__() if self.device is not None else "",
        }

        logger.info(f"Runtime info: {self.runtime_info}")
        logger.info(f"Pipeline : {pipeline}")
        connected = StorageManager.ensure_connection("s3://", silence_exceptions=False)
        logger.warning(f"S3 connection status : {connected}")

    # @requests(on="/document/status")
    # def status(self, parameters, **kwargs):
    #     use_cuda = torch.cuda.is_available()
    #     print(f"{use_cuda=}")
    #     return {"index": "complete", "use_cuda": use_cuda}
    #
    # @requests(on="/document/validate")
    # def validate(self, parameters, **kwargs):
    #     return {"valid": True}

    @requests(on="/document/extractXXXX")
    # @safely_encoded # BREAKS WITH docarray 0.39
    def extract(self, docs: DocList[AssetKeyDoc], parameters: dict, *args, **kwargs):
        self.logger.info("TEXT-EXTRACT")

        self.logger.info(docs)
        try:
            frames = get_frames_from_docs(docs)
        except Exception as e:
            self.logger.error(f"Error in get_frames_from_docs : {e}")
            return {"error": str(e)}

        self.logger.info(f"Doc Ids: {list(doc.id for doc in docs)}")
        self.logger.info(f"Frame Count: {len(frames)}")

        self.logger.info(parameters)
        job_id, ref_id, ref_type, _, payload = parse_parameters(
            parameters, strict=False
        )
        self.logger.info(
            f"job_id, ref_id, ref_type, payload: {job_id}, {ref_id}, {ref_type}, {payload}"
        )

        time.sleep(1)
        return safely_encoded(lambda x: x)(self.runtime_info)

    @requests(on="/document/extract")
    async def func_extract(
        self,
        docs: DocList[AssetKeyDoc],
        parameters=None,
        *args,
        **kwargs,
    ):
        if parameters is None:
            parameters = {}
        self.logger.info(f"func called : {len(docs)}, {parameters}")
        # randomly throw an error to test the error handling
        import random

        # if random.random() > 0.5:
        #     raise Exception("random error in exec")
        # for doc in docs:
        #     doc.text += " First Exec"
        sec = 3600
        sec = random.randint(1, 5)
        if False:
            # sys.exit()
            raise RuntimeError("Mock error for testing purposes")

        self.logger.info(f"Sleeping for {sec} seconds : ", time.time())
        time.sleep(sec)

        self.logger.info(f"Sleeping for {sec} seconds - done : ", time.time())
        return {
            "parameters": parameters,
            "data": "Data reply",
        }


class FirstExecutor(Executor):
    """Example executor for demonstration."""

    @requests
    def process_one(self, docs: DocList, **kwargs):
        for doc in docs:
            doc.tags["processed_by"] = "FirstExecutor"


class LLMExtractionExecutorMock(MarieExecutor):
    def __init__(
        self,
        name: str = "",
        device: Optional[str] = None,
        num_worker_preprocess: int = 4,
        pipeline: Optional[dict[str, any]] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
        **kwargs,
    ):
        """
        :param device: 'cpu' or 'cuda'. Default is None, which auto-detects the device.
        :param num_worker_preprocess: The number of CPU workers to preprocess images and texts. Default is 4.
        :param minibatch_size: The size of the minibatch for preprocessing and encoding. Default is 32. Reduce this
            number if you encounter OOM errors.
        :param dtype: inference data type, if None defaults to torch.float32 if device == 'cpu' else torch.float16.
        """
        super().__init__(**kwargs)
        logger.info(f"Starting mock executor : {time.time()}")
        self.show_error = True  # show prediction errors
        # sometimes we have CUDA/GPU support but want to only use CPU
        use_cuda = torch.cuda.is_available()
        if os.environ.get("MARIE_DISABLE_CUDA"):
            use_cuda = False
        self.logger = MarieLogger(context=self.__class__.__name__)

        if not device:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not use_cuda:
            device = "cpu"
        self.device = device

        self.runtime_info = {
            "name": self.__class__.__name__,
            "instance_name": kwargs.get("runtime_args", {}).get("name", "not_defined"),
            "model": "",
            "host": get_ip_address(),
            "workspace": self.workspace,
            "use_cuda": use_cuda,
            "device": self.device.__str__() if self.device is not None else "",
        }

        logger.info(f"Runtime info: {self.runtime_info}")
        logger.info(f"Pipeline : {pipeline}")

    # @requests(on="/document/status")
    # def status(self, parameters, **kwargs):
    #     use_cuda = torch.cuda.is_available()
    #     print(f"{use_cuda=}")
    #     return {"index": "complete", "use_cuda": use_cuda}
    #
    # @requests(on="/document/validate")
    # def validate(self, parameters, **kwargs):
    #     return {"valid": True}

    @requests(on="/document/llm-annotate")
    def extract(self, docs: DocList[AssetKeyDoc], parameters: dict, *args, **kwargs):
        print("LLM-TEXT-EXTRACT")
        print(parameters)
        print(docs)

        logger.info(kwargs)
        logger.info(parameters)

        if len(docs) == 0:
            return {"error": "empty payload"}
        if len(docs) > 1:
            return {"error": "expected single document"}

        doc = docs[0]
        # load documents from specified document asset key
        docs = docs_from_asset(doc.asset_key, doc.pages)

        for doc in docs:
            print(doc.id)

        frames = frames_from_docs(docs)
        frame_len = len(frames)

        print(f"{frame_len=}")
        # this value will be stuffed in the  resp.parameters["__results__"] as we are using raw Responses

        if "payload" not in parameters or parameters["payload"] is None:
            return {"error": "empty payload"}
        else:
            payload = parameters["payload"]

        # https://github.com/marieai/marie-ai/issues/51

        regions = payload["regions"] if "regions" in payload else []
        for region in regions:
            region["id"] = int(region["id"])
            region["pageIndex"] = int(region["pageIndex"])

        np_arr = np.array([1, 2, 3])

        out = [
            {"sample": 112, "complex": ["a", "b"]},
            {"sample": 112, "complex": ["a", "b"], "np_arr": np_arr},
        ]

        time.sleep(1)
        # invoke the safely_encoded decorator as a function
        meta = get_ip_address()
        #  DocList / Dict / `None`
        # converted = safely_encoded(lambda x: x)(self.runtime_info)
        # return converted
