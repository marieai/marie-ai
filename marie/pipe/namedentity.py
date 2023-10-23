from abc import ABC, abstractmethod
from pprint import pprint
from typing import Optional, List

from marie import DocumentArray

from marie.logging.logger import MarieLogger
from marie.pipe.base import PipelineComponent, PipelineResult, PipelineContext


class NamedEntityPipelineComponent(PipelineComponent, ABC):
    def __init__(
        self, name: str, document_indexers: dict, logger: MarieLogger = None
    ) -> None:
        """
        :param name: Will be passed to base class
        """
        super().__init__(name, logger=logger)
        self.document_indexers = document_indexers

    def predict(
        self,
        documents: DocumentArray,
        context: Optional[PipelineContext] = None,
        *,  # force users to use keyword arguments
        words: List[List[str]] = None,
        boxes: List[List[List[int]]] = None,
    ) -> PipelineResult:
        print("NamedEntityPipelineComponent.predict()")

        context["metadata"]["page_indexer"] = self.extract_named_entity(
            documents, words, boxes
        )

        return PipelineResult(documents)

    def extract_named_entity(self, documents: DocumentArray, words, boxes):
        """
        Classify document at page level

        :param documents: documents to classify
        :param words: words
        :param boxes: boxes
        :return: classification results
        """

        document_meta = []
        try:
            for key, document_indexer in self.document_indexers.items():
                meta = []

                try:
                    self.logger.info(f"Indexers document : {key}")
                    indexer = document_indexer["indexer"]

                    has_filter = "filter" in document_indexer
                    filtered_documents = []
                    filter_pattern = None
                    filter_type = None

                    if has_filter:
                        indexer_filter = (
                            document_indexer["filter"]
                            if "filter" in document_indexer
                            else {}
                        )
                        filter_type = indexer_filter["type"]
                        filter_pattern = indexer_filter["pattern"]

                    for document in documents:
                        if "classification" not in document.tags:
                            self.logger.warning(
                                f"Document has no classification tag, adding to filtered documents"
                            )
                            filtered_documents.append(document)
                            continue

                        classification = document.tags["classification"]
                        if filter_type == "regex":
                            import re

                            if re.search(filter_pattern, classification):
                                self.logger.info(
                                    f"Document classification matches filter : {classification} : {filter_pattern}"
                                )
                                meta.append(
                                    {
                                        "classification": classification,
                                        "pattern": filter_pattern,
                                    }
                                )
                                filtered_documents.append(document)
                                continue

                    # def extract(
                    #         self, docs: DocumentArray, parameters: Optional[Dict] = None, *args, **kwargs
                    # ):
                    #
                    parameters = {
                        "ref_id": "test",
                        "ref_type": "pid",
                    }

                    indexed_docs = indexer.extract(
                        docs=DocumentArray(filtered_documents), parameters=parameters
                    )

                    pprint(indexed_docs)
                    document_meta.append(
                        {
                            "indexer": key,
                            "details": indexed_docs,
                        }
                    )
                except Exception as e:
                    self.logger.error(f"Error classifying document : {e}")
                    document_meta.append(
                        {
                            "indexer": key,
                            "details": [],
                        }
                    )
        except Exception as e:
            self.logger.error(f"Error indexing document : {e}")

        return document_meta
