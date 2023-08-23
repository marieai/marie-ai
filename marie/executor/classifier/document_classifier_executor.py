import os
from typing import Dict, Union, Optional

import numpy as np
import torch
from docarray import DocumentArray, Document

from marie import Executor, requests, safely_encoded
from marie.components import TransformersDocumentClassifier
from marie.executor.mixin import StorageMixin
from marie.logging.logger import MarieLogger
from marie.utils.docs import array_from_docs
from marie.utils.image_utils import (
    hash_frames_fast,
    convert_to_bytes,
)
from marie.utils.network import get_ip_address


class DocumentClassificationExecutor(Executor, StorageMixin):
    """Executor for document classification"""

    def __init__(
        self,
        model_name_or_path: Optional[Union[str, os.PathLike, list]] = None,
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

        # model_name_or_path="marie/layoutlmv3-document-classification"
        self.classifiers = dict()

        if isinstance(model_name_or_path, list):
            self.logger.info(f"Using ensemble of models: {model_name_or_path}")
            for model in model_name_or_path:
                self.logger.info(f"Loading model: {model}")
                self.classifiers[model] = TransformersDocumentClassifier(
                    model_name_or_path=model
                )
        else:
            self.classifiers[model_name_or_path] = TransformersDocumentClassifier(
                model_name_or_path=model_name_or_path
            )

        self.logger.info(f"Storage enabled: {storage_enabled}")
        self.setup_storage(storage_enabled, storage_conf)

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

    @requests(on="/document/classify")
    @safely_encoded
    def classify(
        self, docs: DocumentArray, parameters: Optional[Dict] = None, *args, **kwargs
    ):
        """
        Document classification

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

                results = executor.classify(docs, parameters)


        :param parameters:
        :param docs: Documents to process
        :param kwargs:
        :return:
        """
        self.logger.info("Starting classification request")

        try:
            frames = array_from_docs(docs)
            self.logger.info(f"Processing total frames : {len(frames)}")

            if parameters:
                for key, value in parameters.items():
                    self.logger.info("The p-value of {} is {}".format(key, value))
                ref_id = parameters.get("ref_id")
                ref_type = parameters.get("ref_type")
            else:
                self.logger.warning(
                    f"REF_ID and REF_TYPE are not present in parameters"
                )
                ref_id = hash_frames_fast(frames)
                ref_type = "checksum_frames"

            results = []
            for i, frame in enumerate(frames):
                try:
                    doc_id = f"doc_{i}"

                    self.persist(ref_id, ref_type, i, frame, None)
                except Exception as e:
                    self.logger.warning(f"Unable to segment document : {e}")

            return results
        except BaseException as error:
            self.logger.error("Extract error", error)
            if self.show_error:
                return {"error": str(error)}
            else:
                return {"error": "inference exception"}

    def persist(
        self,
        ref_id: str,
        ref_type: str,
        index: int,
        frame: np.ndarray,
        real: np.ndarray,
    ) -> None:
        def _tags(tag_index: int, ftype: str, checksum: str):
            return {
                "action": "classifier",
                "index": tag_index,
                "type": ftype,
                "ttl": 48 * 60,
                "checksum": checksum,
                "runtime": self.runtime_info,
            }

        if self.storage_enabled:
            frame_checksum = hash_frames_fast(frames=[frame])
            docs = DocumentArray(
                [
                    Document(
                        blob=convert_to_bytes(real),
                        tags=_tags(index, "classification", frame_checksum),
                    ),
                ]
            )

            self.store(
                ref_id=ref_id,
                ref_type=ref_type,
                store_mode="blob",
                docs=docs,
            )
