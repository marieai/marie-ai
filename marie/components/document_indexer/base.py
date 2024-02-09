from abc import abstractmethod
from typing import List, Optional

from docarray import DocList

from marie.base_handler import BaseHandler
from marie.logging.logger import MarieLogger

from ...api.docs import MarieDoc


class BaseDocumentIndexer(BaseHandler):
    """
    Base class for document indexers (Named Entity Recognition, etc.).

    This class provides a common interface for document indexing models.
    Subclasses should implement the `predict` method to perform the actual indexing.

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
        pass

    def run(
        self,
        documents: DocList[MarieDoc],
        words: List[List[str]] = None,
        boxes: List[List[List[int]]] = None,
        batch_size: Optional[int] = None,
    ) -> DocList[MarieDoc]:
        """
        Run the document indexer on the given documents.

        :param documents: List of MarieDoc objects representing the documents to be indexed.
        :param words: List of word tokens for each document.
        :param boxes: List of bounding boxes for each word token in each document.
        :param batch_size: Optional batch size for processing the documents.
        :return: List of MarieDoc objects representing the indexed documents.
        """
        if documents:
            results = self.predict(
                documents=documents, words=words, boxes=boxes, batch_size=batch_size
            )
        else:
            results = DocList[MarieDoc]()

        document_id = [document.id for document in documents]
        self.logger.debug(f"Indexed documents with IDs: {document_id}")
        return results
