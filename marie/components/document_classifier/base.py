from abc import abstractmethod
from typing import Optional, List

from docarray import DocList

from marie import DocumentArray
from marie.api.docs import MarieDoc
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
        documents: DocList[MarieDoc],
        words: Optional[List[List[str]]] = None,
        boxes: Optional[List[List[List[int]]]] = None,
        batch_size: Optional[int] = None,
    ) -> DocumentArray:
        pass

    def run(
        self,
        documents: DocList[MarieDoc],
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
            results = DocList[MarieDoc]()

        document_id = [document.id for document in documents]
        self.logger.debug(f"Classified documents with IDs: {document_id}")
        return results
