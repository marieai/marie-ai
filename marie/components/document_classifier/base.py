from abc import abstractmethod
from typing import List, Optional

from docarray import DocList

from marie.api.docs import MarieDoc
from marie.base_handler import BaseHandler
from marie.logging_core.logger import MarieLogger


class BaseDocumentClassifier(BaseHandler):
    """
    Base class for document classifiers.

    This class provides a common interface for document classification models.
    Subclasses should implement the `predict` method to perform the actual classification.

    :param kwargs: Additional keyword arguments.
    """

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
    ) -> DocList[MarieDoc]:
        """
        Predict the class labels for the given documents.

        :param documents: List of documents to classify.
        :param words: List of word tokens for each document.
        :param boxes: List of bounding boxes for each document.
        :param batch_size: Batch size for prediction.
        :return: List of classified documents.
        """
        pass

    def run(
        self,
        documents: DocList[MarieDoc],
        words: Optional[List[List[str]]] = None,
        boxes: Optional[List[List[List[int]]]] = None,
        batch_size: Optional[int] = None,
    ) -> DocList[MarieDoc]:
        """
        Run the document classifier on the given documents.

        :param documents: List of documents to classify.
        :param words: List of word tokens for each document.
        :param boxes: List of bounding boxes for each document.
        :param batch_size: Batch size for prediction.
        :return: List of classified documents.
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
