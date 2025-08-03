import os
import random
import time
from typing import Optional, Union

import torch
from docarray import DocList
from omegaconf import OmegaConf

from marie.api.docs import AssetKeyDoc
from marie.constants import __config_dir__
from marie.executor.extract.util import layout_config, prepare_asset_directory
from marie.executor.marie_executor import MarieExecutor
from marie.executor.mixin import StorageMixin
from marie.extract.annotators.types import AnnotatorClassType
from marie.extract.readers.meta_reader.meta_reader import MetaReader
from marie.extract.structures import UnstructuredDocument
from marie.logging_core.logger import MarieLogger
from marie.logging_core.mdc import MDC
from marie.logging_core.predefined import default_logger as logger
from marie.models.utils import torch_gc
from marie.utils.docs import docs_from_asset, frames_from_docs
from marie.utils.json import load_json_file
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

        self.root_config_dir = os.path.join(__config_dir__, "extract")
        self.logger.info(f"root_config_dir: {self.root_config_dir}")

    # @requests(on="/default")
    # def default(self,
    #         docs: DocList[AssetKeyDoc],
    #         parameters: dict, **kwargs):
    #     print('===================== DEFAULT =====================')
    #     self.logger.warning(f"Default endpoint called")
    #     print(docs)
    #     print(parameters)
    #     print(kwargs)
    #     raise NotImplementedError(
    #         'Invalid(/default) endpoint have been called, ensure your config are correct'
    #     )

    def _setup_request(
        self, docs: DocList[AssetKeyDoc], parameters: dict, *args, **kwargs
    ):
        if len(docs) == 0:
            return {"error": "empty payload"}
        if len(docs) > 1:
            return {"error": "expected single document"}

        if parameters is None or "job_id" not in parameters:
            self.logger.warning(f"Job ID is not present in parameters")
            raise ValueError("Job ID is not present in parameters")

        job_id = parameters.get("job_id")
        MDC.put("request_id", job_id)

        self.logger.info("processing request parameters")
        for key, value in parameters.items():
            self.logger.info("The value of {} is {}".format(key, value))

    async def _process_annotation_request(
        self,
        docs: DocList[AssetKeyDoc],
        parameters: dict,
        annotator_class: AnnotatorClassType,
        *args,
        **kwargs,
    ):
        """
        Common processing method for document annotation requests

        :param docs: Documents to process
        :param parameters: Request parameters
        :param annotator_class: The annotator class to use (LLMAnnotator or LLMTableAnnotator)
        :param args: Additional arguments
        :param kwargs: Additional keyword arguments
        :return: Response dictionary
        """

        if False:
            # Simulate a long-running process with a random chance of cancellation
            self.logger.info(
                "Simulating long-running process with a random chance of cancellation"
            )
            self.logger.info(
                "========================= SLEEPING ========================="
            )
            # if random.random() < 0.2:  # 20% chance to simulate async cancellation
            #     raise asyncio.CancelledError("Simulated async task cancellation")

            sec = random.randint(15, 30)
            # sec = 2
            time.sleep(sec)  # this will trigger
            for i in range(sec):
                # await asyncio.sleep(1)
                # time.sleep(1)
                self.logger.info(f"Sleeping... {i + 1}/{sec} seconds elapsed")

            return {'status': 'success', 'message': 'Documents annotated successfully'}

        self._setup_request(docs, parameters)

        # load documents from specified document asset key
        doc: AssetKeyDoc = docs[0]
        self.logger.info(f"doc.asset_key = {doc.asset_key}")
        docs, local_downloaded_s3_path = docs_from_asset(
            doc.asset_key, doc.pages, return_file_path=True
        )
        frames = frames_from_docs(docs)

        try:
            job_id = parameters.get("job_id")
            ref_id = parameters.get("ref_id")
            ref_type = parameters.get("ref_type")
            payload = parameters.get("payload")
            op_params = payload.get(
                "op_params"
            )  # These are operator parameters (Layout, Config Key, etc.)

            op_key = op_params.get('key')
            op_layout = op_params.get('layout')

            self.logger.info(f"Executing operation with params : {op_params}")
            self.logger.info(f"Extracted op_key: {op_key}")
            self.logger.info(f"Extracted op_layout: {op_layout}")

            cfg = layout_config(self.root_config_dir, op_layout)

            annotator_conf = None
            for annotator in cfg.annotators:
                if annotator == op_key:
                    annotator_conf = cfg.annotators[annotator]
                    # we need to set the name of the annotator as they are keys in conf
                    annotator_conf['name'] = annotator
                    break

            if annotator_conf is None:
                raise ValueError(f"Invalid annotator key: {op_key}")
            # remove any dependencies on OmegaConf to avoid issues with index access
            annotator_conf = OmegaConf.to_container(annotator_conf, resolve=True)

            root_asset_dir, frames_dir, metadata_file = prepare_asset_directory(
                frames=frames,
                local_path=local_downloaded_s3_path,
                ref_id=ref_id,
                ref_type=ref_type,
                logger=self.logger,
            )
            self.logger.info(f"root_asset_dir = {root_asset_dir}")

            # self.logger.info(f"Downloaded assets to {metadata_file}")
            metadata = load_json_file(metadata_file)
            unstructured_meta = {
                'ref_id': ref_id,
                'ref_type': ref_type,
                'job_id': job_id,
                'source_metadata': metadata,
            }

            doc: UnstructuredDocument = MetaReader.from_data(
                frames=frames,
                ocr_meta=metadata["ocr"],
                unstructured_meta=unstructured_meta,
            )
            self.logger.info(f"Doc : {doc}")
            self.logger.info(f"Doc page_count: {doc.page_count}")

            annotator = annotator_class(
                working_dir=root_asset_dir,
                annotator_conf=annotator_conf,
                layout_conf={
                    "layout_id": op_layout,
                },
            )

            await annotator.aannotate(doc, frames)
            del annotator

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
