import os
from abc import abstractmethod
from typing import Optional, List

from docarray import DocumentArray

from marie.base_handler import BaseHandler
from marie.constants import __model_path__
from marie.logging.logger import MarieLogger


class BaseDocumentClassifier(BaseHandler):
    def __init__(
        self,
        work_dir: str,
        models_dir: str = os.path.join(__model_path__),
        cuda: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.models_dir = os.path.join(models_dir, "classifier")
        self.cuda = cuda
        self.work_dir = work_dir
        self.initialized = False
        self.logger = MarieLogger(self.__class__.__name__).logger

    @abstractmethod
    def predict(
        self,
        documents: DocumentArray,
        words: Optional[List[str]] = None,
        boxes: Optional[List[List[int]]] = None,
        batch_size: Optional[int] = None,
    ) -> DocumentArray:
        pass

    def run(
        self,
        documents: DocumentArray,
        words: Optional[List[str]] = None,
        boxes: Optional[List[List[int]]] = None,
        batch_size: Optional[int] = None,
    ):
        if documents:
            results = self.predict(
                documents=documents, words=words, boxes=boxes, batch_size=batch_size
            )
        else:
            results = DocumentArray()

        document_id = [document.id for document in documents]

        output = {"documents": results}
        self.logger.info(f"Classified documents with IDs: {document_id}")
        return output
