import time
from pprint import pformat
from typing import Optional, Union

import torch
from docarray import DocList

from marie import requests
from marie.api.docs import AssetKeyDoc
from marie.executor.asset_util import prepare_asset_directory
from marie.executor.extract import DocumentAnnotatorExecutor
from marie.executor.extract.util import layout_config
from marie.extract.registry import component_registry
from marie.extract.registry.loader import initialize_components_from_config
from marie.extract.results.result_parser import parse_results
from marie.logging_core.logger import MarieLogger
from marie.logging_core.predefined import default_logger as logger
from marie.utils.docs import docs_from_asset, frames_from_docs
from marie.utils.json import load_json_file


class DocumentAnnotatorParserExecutor(DocumentAnnotatorExecutor):
    """Executor for extract annotation"""

    def __init__(
        self,
        name: str = "",
        device: Optional[str] = None,
        num_worker_preprocess: int = 4,
        storage: dict[str, any] = None,
        parsers: dict[str, any] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
        **kwargs,
    ):
        kwargs['storage'] = storage
        super().__init__(**kwargs)
        self.logger = MarieLogger(
            getattr(self.metas, "name", self.__class__.__name__)
        ).logger

        result = initialize_components_from_config(parsers)

        logger.info(f"Loaded modules: {result['loaded']}")
        logger.info(f"Failed modules: {result['failed']}")
        logger.info(f"Total parsers available: {result['total_parsers']}")

        reg_parsers = component_registry.list_parsers()
        logger.info("Available parsers:")
        for parser_name in reg_parsers:
            logger.info(f"- {parser_name}")

        info = component_registry.get_registry_info()
        logger.info(f"Parser info:\n{pformat(info)}")
        logger.info(f"Started executor : {self.__class__.__name__}")

    # TODO : this should be moved to a proper pipeline
    @requests(on="/annotator/result-parser")
    def annotator_result_parser(
        self, docs: DocList[AssetKeyDoc], parameters: dict, *args, **kwargs
    ):
        """
        Document result parser executor
        :param docs: Documents to process
        :param parameters: Request parameters
        :param args: Additional arguments
        :param kwargs: Additional keyword arguments
        :return: Response dictionary
        """

        if False:
            print('===================== SLEEPING =====================')
            time.sleep(0)
            return

        self._setup_request(docs, parameters, *args, **kwargs)

        job_id = parameters.get("job_id")
        ref_id = parameters.get("ref_id")
        ref_type = parameters.get("ref_type")
        payload = parameters.get("payload")
        op_params = payload.get(
            "op_params"
        )  # These are operator parameters (Layout, Config Key, etc.)

        op_key = op_params.get('key')
        op_layout = op_params.get('layout')  # used to locate the config files

        self.logger.info(f"Executing operation with params : {op_params}")
        self.logger.info(f"Extracted op_key: {op_key}")
        self.logger.info(f"Extracted op_layout: {op_layout}")

        conf = layout_config(self.root_config_dir, op_layout)

        # load documents from specified document asset key
        doc: AssetKeyDoc = docs[0]
        self.logger.info(f"doc.asset_key = {doc.asset_key}")
        docs, local_downloaded_s3_path = docs_from_asset(
            doc.asset_key, doc.pages, return_file_path=True
        )
        frames = frames_from_docs(docs)

        root_asset_dir, frames_dir, metadata_file = prepare_asset_directory(
            frames=frames,
            local_path=local_downloaded_s3_path,
            ref_id=ref_id,
            ref_type=ref_type,
            logger=self.logger,
        )
        self.logger.info(f"root_asset_dir = {root_asset_dir}")

        metadata = load_json_file(metadata_file)
        unstructured_meta = {
            'ref_id': ref_id,
            'ref_type': ref_type,
            'job_id': job_id,
            'source_metadata': metadata,
        }

        parse_results(root_asset_dir, metadata, conf)
