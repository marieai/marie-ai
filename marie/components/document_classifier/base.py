from abc import abstractmethod
from typing import Optional, List

from docarray import DocumentArray

from marie.base_handler import BaseHandler
from marie.logging.logger import MarieLogger


class BaseDocumentClassifier(BaseHandler):
    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__()
        self.logger = MarieLogger(self.__class__.__name__).logger

    @abstractmethod
    def predict(
        self,
        documents: DocumentArray,
        words: Optional[List[List[str]]] = None,
        boxes: Optional[List[List[List[int]]]] = None,
        batch_size: Optional[int] = None,
    ) -> DocumentArray:
        pass

    def run(
        self,
        documents: DocumentArray,
        words: Optional[List[List[str]]] = None,
        boxes: Optional[List[List[List[int]]]] = None,
        batch_size: Optional[int] = None,
    ) -> DocumentArray:
        """
        Run the document classifier on the given documents.

        :param documents:
        :param words:
        :param boxes:
        :param batch_size:
        :return:
        """
        if documents:
            results = self.predict(
                documents=documents, words=words, boxes=boxes, batch_size=batch_size
            )
        else:
            results = DocumentArray()

        document_id = [document.id for document in documents]

        # output = {"documents": results}
        self.logger.info(f"Classified documents with IDs: {document_id}")
        return results