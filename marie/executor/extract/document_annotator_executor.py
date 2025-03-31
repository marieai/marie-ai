import os
import shutil
from datetime import datetime
from typing import List, Optional, Union

import torch
from docarray import DocList
from PIL import Image

from marie import requests
from marie.api.docs import AssetKeyDoc
from marie.executor.marie_executor import MarieExecutor
from marie.executor.mixin import StorageMixin
from marie.logging_core.logger import MarieLogger
from marie.logging_core.mdc import MDC
from marie.logging_core.predefined import default_logger as logger
from marie.models.utils import torch_gc
from marie.pipe.components import burst_frames, restore_assets
from marie.utils.docs import docs_from_asset, frames_from_docs
from marie.utils.image_utils import ensure_max_page_size, hash_frames_fast
from marie.utils.network import get_ip_address
from marie.utils.utils import ensure_exists


def create_working_dir(frames: List, backup: bool = False) -> str:
    frame_checksum = hash_frames_fast(frames=frames)
    # create backup name by appending a timestamp
    if backup:
        if os.path.exists(os.path.join("/tmp/generators", frame_checksum)):
            ts = datetime.now().strftime("%Y%m%d%H%M%S")
            shutil.move(
                os.path.join("/tmp/generators", frame_checksum),
                os.path.join("/tmp/generators", f"{frame_checksum}-{ts}"),
            )
    root_asset_dir = ensure_exists(os.path.join("/tmp/generators", frame_checksum))
    return root_asset_dir


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
        kwargs['storage'] = storage
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
        }

        self.storage_enabled = False
        if storage is not None and "psql" in storage:
            sconf = storage["psql"]
            self.setup_storage(sconf.get("enabled", False), sconf)

    @requests(on="/default")
    def default(self, parameters, **kwargs):
        raise NotImplementedError(
            'Invalid(/default) endpoint have been called, ensure your config are correct'
        )

    @requests(on="/annotator/llm")
    def annotator_llm(
        self, docs: DocList[AssetKeyDoc], parameters: dict, *args, **kwargs
    ):
        """
        Document annotator executor
        Much of this is hardcoded in here and need to be moved into proper pipeline.

        EXAMPLE USAGE

            As Executor

            .. code-block:: python

                exec = AnnotatorExecutor()

        :param parameters:
        :param docs: Documents to process
        :param kwargs:
        :return:
        """

        print(f'Annotating documents')
        print(f'docs : {docs}')
        print(f'parameters : {parameters}')

        if len(docs) == 0:
            return {"error": "empty payload"}
        if len(docs) > 1:
            return {"error": "expected single document"}

        # load documents from specified document asset key
        doc = docs[0]
        self.logger.info(f"doc.asset_key = {doc.asset_key}")
        docs = docs_from_asset(doc.asset_key, doc.pages)
        frames = frames_from_docs(docs)

        if parameters is None or "job_id" not in parameters:
            self.logger.warning(f"Job ID is not present in parameters")
            raise ValueError("Job ID is not present in parameters")

        job_id = parameters.get("job_id")
        MDC.put("request_id", job_id)

        self.logger.info("Starting Annotator processing request")
        for key, value in parameters.items():
            self.logger.info("The value of {} is {}".format(key, value))

        try:
            ref_id = parameters.get("ref_id")
            ref_type = parameters.get("ref_type")

            payload = parameters.get("payload")
            op_params = payload.get("op_params")
            self.logger.info(f"Executing operation with params : {op_params}")

            root_asset_dir = create_working_dir(frames)
            ensure_exists(os.path.join(root_asset_dir, "frames"))

            self.logger.info(f"root_asset_dir = {root_asset_dir}")
            # Save frames as PNG files as LLM will need this format
            for idx, frame in enumerate(frames):
                frame_path = os.path.join(root_asset_dir, "frames", f"{idx}.png")
                self.logger.info(f"Saving frame {idx + 1} to {frame_path}")
                Image.fromarray(frame).save(frame_path)

            restore_assets(
                ref_id, ref_type, root_asset_dir, full_restore=False, overwrite=True
            )

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
