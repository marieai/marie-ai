import os
from typing import Dict, Union, Optional, Any

import torch
from docarray import DocumentArray, Document

from marie import Executor, requests, safely_encoded
from marie.logging.logger import MarieLogger
from marie.overlay.overlay import OverlayProcessor
from marie.timer import Timer
from marie.utils.docs import array_from_docs
from marie.utils.utils import ensure_exists
from marie.executor.storage.PostgreSQLStorage import PostgreSQLStorage


class OverlayExecutor(Executor):
    """Executor for creating text overlays."""

    def __init__(
        self,
        model_name_or_path: Optional[Union[str, os.PathLike]] = None,
        storage_enabled: bool = False,
        storage_conf: Dict[str, str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.show_error = True  # show prediction errors
        self.logger = MarieLogger(
            getattr(self.metas, "name", self.__class__.__name__)
        ).logger
        # sometimes we have CUDA/GPU support but want to only use CPU
        use_cuda = torch.cuda.is_available()
        if os.environ.get("MARIE_DISABLE_CUDA"):
            use_cuda = False
        work_dir = ensure_exists("/tmp/form-segmentation")
        self.overlay_processor = OverlayProcessor(work_dir=work_dir, cuda=use_cuda)
        self.__setup_storage(storage_enabled, storage_conf)

    @requests(on="/overlay/segment")
    @safely_encoded
    def segment(
        self, docs: DocumentArray, parameters: Optional[Dict] = None, *args, **kwargs
    ):
        """
        Segment document

        EXAMPLE USAGE

            As Executor

            .. code-block:: python

                filename = img_path.split("/")[-1].replace(".png", "")
                checksum = hash_file(img_path)
                docs = docs_from_file(img_path)

                parameters = {
                    "ref_id": filename,
                    "ref_type": "filename",
                    "checksum": checksum,
                    "img_path": img_path,
                }

                results = executor.segment(docs, parameters)


        :param parameters:
        :param docs: Documents to process
        :param kwargs:
        :return:
        """
        self.logger.info("Starting segment request")

        if parameters:
            for key, value in parameters.items():
                self.logger.info("The p-value of {} is {}".format(key, value))
            filename = parameters.get("ref_id")
        else:
            filename = "GREG"

        storage_meta = {"ref_id": filename, "ref_type": "filename"}
        print(f"storage_meta = {storage_meta}")

        try:
            frames = array_from_docs(docs)
            print(f'frames = {len(frames)}')
            results = []
            for i, frame in enumerate(frames):
                try:
                    doc_id = f"doc_{i}"
                    real, mask, blended = self.overlay_processor.segment_frame(
                        doc_id, frame
                    )
                    dd = DocumentArray([Document(tensor=mask)])
                    self.__store(ref_id=filename, ref_type="filename", docs=dd)
                except Exception as e:
                    self.logger.warning("Unable to segment document")

            return results
        except BaseException as error:
            self.logger.error("Extract error", error)
            if self.show_error:
                return {"error": str(error)}
            else:
                return {"error": "inference exception"}

    def __setup_storage(self, storage_enabled, storage_conf: Dict[str, str]):
        self.storage = None
        if storage_enabled:
            try:
                self.storage = PostgreSQLStorage(
                    hostname=storage_conf["hostname"],
                    port=int(storage_conf["port"]),
                    username=storage_conf["username"],
                    password=storage_conf["password"],
                    database=storage_conf["database"],
                    table="overlay_indexer",
                )
            except Exception as e:
                self.logger.warning("Storage config not set", exc_info=1)

    @Timer(text="stored in {:.4f} seconds")
    def __store(self, ref_id: int, ref_type: str, docs: DocumentArray) -> None:
        """Store results"""
        try:
            if self.storage is not None:
                self.storage.add(docs, {"ref_id": ref_id, "ref_type": ref_type})
        except Exception as e:
            self.logger.error("Unable to store document")
            print(e)
