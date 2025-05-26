import os
import os.path
import time
from typing import Optional, Union

import torch
from docarray import DocList
from grapnel_g5.result_parser import extract_tables, highlight_tables, parse_tables
from omegaconf import OmegaConf

from marie import requests
from marie.api.docs import AssetKeyDoc
from marie.executor.extract import DocumentAnnotatorExecutor
from marie.executor.extract.util import prepare_asset_directory, setup_table_directories
from marie.extract.readers.meta_reader.meta_reader import MetaReader
from marie.extract.structures import UnstructuredDocument
from marie.logging_core.logger import MarieLogger
from marie.logging_core.predefined import default_logger as logger
from marie.utils.docs import docs_from_asset, frames_from_docs
from marie.utils.json import load_json_file


class DocumentAnnotatorTableParserExecutor(DocumentAnnotatorExecutor):
    """Executor for extract tables from annotations that later will be used for extracting OCR data (table fragments)"""

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

        logger.info(f"Started executor : {self.__class__.__name__}")

    @requests(on="/annotator/table-parser")
    async def annotator_table_result_parser(
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
        self.logger.info("Table parser executor")

        if True:
            print('===================== SLEEPING =====================')
            time.sleep(2)
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

        conf = self.layout_config(op_layout)

        # load documents from specified document asset key
        doc: AssetKeyDoc = docs[0]
        self.logger.info(f"doc.asset_key = {doc.asset_key}")
        docs, local_downloaded_s3_path = docs_from_asset(
            doc.asset_key, doc.pages, return_file_path=True
        )
        frames = frames_from_docs(docs)

        ##########
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

        document: UnstructuredDocument = MetaReader.from_data(
            frames=frames, ocr_meta=metadata["ocr"], unstructured_meta=unstructured_meta
        )
        self.logger.info(f"Doc : {document}")
        self.logger.info(f"Doc page_count: {document.page_count}")

        try:
            working_dir = root_asset_dir
            (
                htables_output_dir,
                table_src_dir,
                table_annotated_dir,
                table_annotated_fragments_dir,
                table_output_dir,
            ) = setup_table_directories(working_dir)

            self.logger.info(f"htables_output_dir = {htables_output_dir}")
            self.logger.info(f"table_src_dir = {table_src_dir}")
            self.logger.info(f"table_annotated_dir = {table_annotated_dir}")
            self.logger.info(
                f"table_annotated_fragments_dir = {table_annotated_fragments_dir}"
            )
            self.logger.info(f"table_output_dir = {table_output_dir}")

            conf = OmegaConf.create({"grounding": {"table": []}})
            parse_tables(document, working_dir, src_dir=table_src_dir, conf=conf)
            highlight_tables(document, frames, htables_output_dir)
            extract_tables(
                document, frames, metadata={}, output_dir=table_annotated_dir
            )
            self.logger.info(f"Extracted tables to {table_annotated_dir}")
        except Exception as e:
            self.logger.error(f"Error in table extraction: {e}")
            raise e

        return {
            "status": "success",
            "message": "Table extraction completed successfully.",
        }
