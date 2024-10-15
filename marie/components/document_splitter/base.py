from abc import abstractmethod
from typing import List, Optional

from docarray import DocList

from marie.api.docs import MarieDoc
from marie.base_handler import BaseHandler
from marie.logging_core.logger import MarieLogger


class BaseDocumentSplitter(BaseHandler):
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
    ) -> DocList:
        """
        Predict the split documents. This method must be implemented by subclasses.
        :param documents:
        :param words:
        :param boxes:
        :param batch_size:
        """
        pass

    def run(
        self,
        documents: DocList,
        words: Optional[List[List[str]]] = None,
        boxes: Optional[List[List[List[int]]]] = None,
        batch_size: Optional[int] = None,
    ) -> DocList:
        """
        Run the document splitter on the given documents.

        :param documents: the documents to split
        :param words:
        :param boxes:
        :param batch_size:
        :return: the split documents in a DocumentArray
        """
        if documents:
            results = self.predict(
                documents=documents, words=words, boxes=boxes, batch_size=batch_size
            )
        else:
            results = DocList()

        document_id = [document.id for document in documents]

        # output = {"documents": results}
        self.logger.info(f"Split documents with IDs: {document_id}")
        return results
