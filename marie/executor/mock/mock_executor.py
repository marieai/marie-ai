import os
import time
from typing import Optional, Union

import numpy as np
import torch
from docarray import DocList

from marie import requests
from marie.api.docs import AssetKeyDoc
from marie.executor.marie_executor import MarieExecutor
from marie.logging_core.logger import MarieLogger
from marie.logging_core.predefined import default_logger as logger
from marie.models.utils import (
    setup_torch_optimizations,
)
from marie.storage import StorageManager
from marie.utils.docs import docs_from_asset, frames_from_docs
from marie.utils.network import get_ip_address


class IntegrationExecutorMock(MarieExecutor):
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
        sec = random.randint(3, 6)
        sec = 3

        if False:
            # sys.exit()
            raise RuntimeError("Mock error for testing purposes")

        self.logger.info(f"Sleeping for {sec} seconds : {time.time()}")
        time.sleep(sec)

        self.logger.info(f"Sleeping for {sec} seconds - done : {time.time()}")
        return {
            "parameters": parameters,
            "data": "Data reply",
        }


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
