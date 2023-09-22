import os
from typing import Dict, Union, Optional, Any

import numpy as np
import psutil
import torch
from docarray import DocumentArray, Document

from marie import Executor, requests, safely_encoded
from marie.api import value_from_payload_or_args
from marie.boxes import PSMode
from marie.executor.mixin import StorageMixin
from marie.logging.logger import MarieLogger
from marie.logging.predefined import default_logger as logger
from marie.models.utils import enable_tf32, openmp_setup
from marie.ocr import CoordinateFormat
from marie.ocr.extract_pipeline import ExtractPipeline
from marie.utils.docs import array_from_docs
from marie.utils.image_utils import hash_frames_fast
from marie.utils.network import get_ip_address


class TextExtractionExecutor(Executor, StorageMixin):
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
        storage: dict[str, any] = None,
        pipeline: dict[str, any] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        logger.info(f"Starting executor : {self.__class__.__name__}")
        logger.info(f"Runtime args : {kwargs.get('runtime_args')}")
        logger.info(f"Storage config: {storage}")
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
        self.pipeline = ExtractPipeline(pipeline_config=pipeline, cuda=use_cuda)

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
            "use_cuda": use_cuda,
        }

        if storage is not None and "psql" in storage:
            sconf = storage["psql"]
            storage_enabled = sconf.get("enabled", False)
            self.setup_storage(storage_enabled, sconf)

    @requests(on="/document/extract")
    @safely_encoded
    def extract(self, docs: DocumentArray, parameters: Dict, *args, **kwargs):
        """Load the image from `uri`, extract text and bounding boxes.
        :param parameters:
        :param docs: Documents to process
        :param kwargs:
        :return:
        """
        self.logger.info("Starting ICR processing request")
        for key, value in parameters.items():
            self.logger.info("The value of {} is {}".format(key, value))

        queue_id: str = parameters.get("queue_id", "0000-0000-0000-0000")

        try:
            if "payload" not in parameters or parameters["payload"] is None:
                return {"error": "empty payload"}
            else:
                payload = parameters["payload"]

            # https://github.com/marieai/marie-ai/issues/51
            regions = payload["regions"] if "regions" in payload else []
            for region in regions:
                region["id"] = f'{int(region["id"])}'
                region["x"] = int(region["x"])
                region["y"] = int(region["y"])
                region["w"] = int(region["w"])
                region["h"] = int(region["h"])
                region["pageIndex"] = int(region["pageIndex"])

            # for testing
            if len(regions) == 0:
                return {"error": "empty regions"}

            # due to compatibility issues with other frameworks we allow passing same arguments in the 'args' object
            coordinate_format = CoordinateFormat.from_value(
                value_from_payload_or_args(payload, "format", default="xywh")
            )
            pms_mode = PSMode.from_value(
                value_from_payload_or_args(payload, "mode", default="")
            )

            # output_format = OutputFormat.from_value(
            #     value_from_payload_or_args(payload, "output", default="json")
            # )

            frames = array_from_docs(docs)
            frame_len = len(frames)

            if parameters:
                for key, value in parameters.items():
                    self.logger.debug("The p-value of {} is {}".format(key, value))
                ref_id = parameters.get("ref_id")
                ref_type = parameters.get("ref_type")
                job_id = parameters.get("job_id", "0000-0000-0000-0000")
            else:
                self.logger.warning(
                    f"REF_ID and REF_TYPE are not present in parameters"
                )
                ref_id = hash_frames_fast(frames)
                ref_type = "extract"
                job_id = "0000-0000-0000-0000"

            self.logger.debug(
                "ref_id, ref_type frames , regions , pms_mode, coordinate_format,"
                f" checksum: {ref_id}, {ref_type},  {frame_len}, {len(regions)}, {pms_mode},"
                f" {coordinate_format}"
            )
            payload_kwargs = {}
            if "args" in payload:
                payload_kwargs["args"] = payload["args"]

            runtime_conf = {}
            if "features" in payload:
                for feature in payload["features"]:
                    if "type" in feature and feature["type"] == "pipeline":
                        runtime_conf = feature

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

            del frames
            del regions

            print('metadata["ocr"]')
            print(metadata)
            self.persist(ref_id, ref_type, metadata)

            # strip out ocr results from metadata
            include_ocr = True
            if not include_ocr and "ocr" in metadata:
                del metadata["ocr"]

            return {
                "status": "succeeded",
                "runtime_info": self.runtime_info,
                "metadata": metadata,
            }
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

    @requests(on="/document/status")
    def status(self, parameters, **kwargs):
        use_cuda = torch.cuda.is_available()
        print(f"{use_cuda=}")
        return {"index": "complete", "use_cuda": use_cuda}

    @requests(on="/document/validate")
    def validate(self, parameters, **kwargs):
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
            }

        if self.storage_enabled:
            # frame_checksum = hash_frames_fast(frames=[frame])
            docs = DocumentArray(
                [
                    Document(
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


class TextExtractionExecutorMock(Executor):
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
        import time

        logger.info(f"Starting mock executor : {time.time()}")
        setup_torch_optimizations()

        self.show_error = True  # show prediction errors
        # sometimes we have CUDA/GPU support but want to only use CPU
        use_cuda = torch.cuda.is_available()
        if os.environ.get("MARIE_DISABLE_CUDA"):
            use_cuda = False
        self.logger = MarieLogger(context=self.__class__.__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.runtime_info = {
            "name": self.__class__.__name__,
            "instance_name": kwargs.get("runtime_args").get("name", "not_defined"),
            "model": "",
            "host": get_ip_address(),
            "workspace": self.workspace,
            "use_cuda": use_cuda,
            "device": self.device.__str__() if self.device is not None else "",
        }

        logger.info(f"Runtime info: {self.runtime_info}")
        logger.info(f"Pipeline : {pipeline}")
        print("TEXT-START")

    @requests(on="/document/status")
    def status(self, parameters, **kwargs):
        use_cuda = torch.cuda.is_available()
        print(f"{use_cuda=}")
        return {"index": "complete", "use_cuda": use_cuda}

    @requests(on="/document/validate")
    def validate(self, parameters, **kwargs):
        return {"valid": True}

    @requests(on="/document/extract")
    @safely_encoded
    def extract(self, parameters, docs: Optional[DocumentArray] = None, **kwargs):
        logger.info(f"Executing extract : {len(docs)}")
        logger.info(kwargs)
        logger.info(parameters)

        import threading
        import time

        if "payload" not in parameters or parameters["payload"] is None:
            return {"error": "empty payload"}
        else:
            payload = parameters["payload"]

        # https://github.com/marieai/marie-ai/issues/51

        regions = payload["regions"] if "regions" in payload else []
        for region in regions:
            region["id"] = int(region["id"])
            region["pageIndex"] = int(region["pageIndex"])

        print("AFTER")
        print(payload)
        time.sleep(5)

        for doc in docs:
            doc.text = f"{doc.text} : >> {threading.get_ident()}"

        np_arr = np.array([1, 2, 3])

        out = [
            {"sample": 112, "complex": ["a", "b"]},
            {"sample": 112, "complex": ["a", "b"], "np_arr": np_arr},
        ]

        meta = get_ip_address()
        return out


def setup_torch_optimizations():
    logger.info(f"Setting up torch optimizations")

    # Optimizations for PyTorch
    core_count = psutil.cpu_count(logical=False)

    torch_versions = torch.__version__.split(".")
    torch_major_version = int(torch_versions[0])
    torch_minor_version = int(torch_versions[1])
    if torch_major_version > 1 or (
        torch_major_version == 1 and torch_minor_version >= 12
    ):
        # Gives a large speedup on Ampere-class GPUs
        torch.set_float32_matmul_precision("high")

    logger.info(f"Setting up TF32")
    enable_tf32()

    logger.info(f"Setting up OpenMP with {core_count} threads")
    openmp_setup(core_count)
    torch.set_num_threads(core_count)

    # Enable oneDNN Graph
    torch.jit.enable_onednn_fusion(True)
