from abc import abstractmethod
from typing import List, Optional

from docarray import DocList

from marie.api.docs import MarieDoc
from marie.base_handler import BaseHandler
from marie.logging_core.logger import MarieLogger


class BaseDocumentTaxonomy(BaseHandler):
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
        metadata: List[dict],
        taxonomy_key: str = "taxonomy",
        batch_size: Optional[int] = None,
    ) -> DocList:
        """
        Predict document taxonomy. This method must be implemented by subclasses.
        :param documents:
        :param metadata:
        :param taxonomy_key:
        :param batch_size:
        """
        pass

    def run(
        self,
        documents: DocList[MarieDoc],
        metadata: List[dict],
        taxonomy_key: str = "taxonomy",
        batch_size: Optional[int] = None,
    ) -> DocList:
        """
        Run the document taxonomy on the given documents.

        :param documents: the documents to process
        :param metadata: the metadata for the documents
        :param taxonomy_key: the key to use for the taxonomy
        :param batch_size: the batch size to use
        :return: the taxonomy annotated documents in a DocumentArray
        """
        if documents:
            results = self.predict(
                documents=documents,
                metadata=metadata,
                taxonomy_key=taxonomy_key,
                batch_size=batch_size,
            )
        else:
            results = DocList()

        document_id = [document.id for document in documents]

        self.logger.info(f" documents with IDs: {document_id}")
        return results
