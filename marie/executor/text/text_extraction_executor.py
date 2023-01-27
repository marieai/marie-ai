import os
from typing import Dict, Union, Optional

import numpy as np
import torch
from docarray import DocumentArray
from marie import Executor, requests, safely_encoded
from marie.api import value_from_payload_or_args
from marie.boxes import PSMode

from marie.logging.logger import MarieLogger
from marie.logging.predefined import default_logger
from marie.ocr import DefaultOcrEngine, OutputFormat, CoordinateFormat
from marie.utils.docs import array_from_docs
from marie.utils.network import get_ip_address


class TextExtractionExecutor(Executor):
    """
    Executor for extracting text.
    Text extraction can either be executed out over the entire image or over selected regions of interests (ROIs)
    aka bounding boxes.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.show_error = True  # show prediction errors
        # sometimes we have CUDA/GPU support but want to only use CPU
        use_cuda = torch.cuda.is_available()
        if os.environ.get("MARIE_DISABLE_CUDA"):
            use_cuda = False
        self.logger = MarieLogger(context=self.__class__.__name__)
        self.ocr_engine = DefaultOcrEngine(cuda=use_cuda)
        # self.exec_pipe = ExtractPipeline(cuda=use_cuda)

        self.runtimeinfo = {
            "name": self.__class__.__name__,
            "instance_name": kwargs.get("runtime_args").get("name", "not_defined"),
            "model": "",
            "host": get_ip_address(),
            "workspace": self.workspace,
            "use_cuda": use_cuda,
        }

    @requests(on="/text/status")
    def info(self, **kwargs):
        self.logger.info(f"Self : {self}")
        return {"index": "complete"}

    @requests(on="/text/extract")
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
                region["id"] = int(region["id"])
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
            output_format = OutputFormat.from_value(
                value_from_payload_or_args(payload, "output", default="json")
            )

            frames = array_from_docs(docs)
            frame_len = len(frames)

            self.logger.info(
                "frames , regions , output_format, pms_mode, coordinate_format,"
                f" checksum:  {frame_len}, {len(regions)}, {output_format}, {pms_mode},"
                f" {coordinate_format}"
            )

            results = self.ocr_engine.extract(
                frames, pms_mode, coordinate_format, regions, queue_id
            )
            # store_json_object(results, '/tmp/fragments/results-complex.json')

            print(results)
            return results
        except BaseException as error:
            self.logger.error("Extract error", error)
            if self.show_error:
                return {"error": str(error)}
            else:
                return {"error": "inference exception"}


class ExtractExecutor(Executor):
    def __init__(
        self,
        name: str = '',
        device: Optional[str] = None,
        num_worker_preprocess: int = 4,
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
        self.show_error = True  # show prediction errors
        # sometimes we have CUDA/GPU support but want to only use CPU
        use_cuda = torch.cuda.is_available()
        print(f"{use_cuda=}")
        if os.environ.get("MARIE_DISABLE_CUDA"):
            use_cuda = False
        self.logger = MarieLogger(context=self.__class__.__name__)
        print(f"{use_cuda=}")
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
        print(self.runtime_info)

    @requests(on="/text/status")
    def status(self, parameters, **kwargs):
        use_cuda = torch.cuda.is_available()
        print(f"{use_cuda=}")
        return {"index": "complete", "use_cuda": use_cuda}

    @requests(on="/text/extract")
    @safely_encoded
    def extract(self, parameters, docs: Optional[DocumentArray] = None, **kwargs):
        default_logger.info(f"Executing extract : {len(docs)}")
        default_logger.info(kwargs)
        default_logger.info(parameters)

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

        print('AFTER')
        print(payload)
        time.sleep(1.3)

        for doc in docs:
            doc.text = f"{doc.text} : >> {threading.get_ident()}"

        np_arr = np.array([1, 2, 3])

        out = [
            {"sample": 112, "complex": ["a", "b"]},
            {"sample": 112, "complex": ["a", "b"], "np_arr": np_arr},
        ]

        meta = get_ip_address()
        return out
