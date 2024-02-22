from abc import ABC
from pprint import pprint
from typing import List, Optional

from docarray import DocList

from marie import DocumentArray
from marie.excepts import BadConfigSource
from marie.logging.logger import MarieLogger
from marie.pipe.base import PipelineComponent, PipelineContext, PipelineResult


class NamedEntityPipelineComponent(PipelineComponent, ABC):
    def __init__(
        self,
        name: str,
        document_indexers: dict,
        logger: MarieLogger = None,
        silence_exceptions: bool = False,
    ) -> None:
        """
        Initialize the NamedEntityPipelineComponent.

        :param name: The name of the pipeline component.
        :param document_indexers: A dictionary containing document indexers.
        :param logger: An optional logger for logging messages.
        :param silence_exceptions: Whether to suppress exceptions.
        """
        super().__init__(name, logger=logger)
        self.silence_exceptions = silence_exceptions
        self.document_indexers = document_indexers

    def predict(
        self,
        documents: DocumentArray,
        context: Optional[PipelineContext] = None,
        *,
        words: List[List[str]] = None,
        boxes: List[List[List[int]]] = None,
    ) -> PipelineResult:
        """
        Predict the named entities in the documents.

        :param documents: The input documents.
        :param context: The pipeline context.
        :param words: The list of words.
        :param boxes: The list of boxes.
        :return: The pipeline result.
        """
        context["metadata"]["page_indexer"] = self.extract_named_entity(
            documents, words, boxes
        )

        return PipelineResult(documents)

    def extract_named_entity(
        self, documents: DocumentArray, words, boxes
    ) -> List[dict]:
        """
        Extract named entities from the documents.

        :param documents: The documents to extract named entities from.
        :param words: The list of words.
        :param boxes: The list of boxes.
        :return: The extracted named entities.
        """
        document_meta = []
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

                    classification = str(document.tags["classification"])
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
                    else:
                        raise NotImplementedError("Exact filter not implemented")

                results = indexer.run(
                    documents=DocumentArray(filtered_documents),
                    words=words,
                    boxes=boxes,
                )

                for rdoc, document in zip(results, filtered_documents):
                    indexed_page = rdoc.tags.get("indexer", [])
                    document_meta.append(
                        {
                            "indexer": key,
                            "details": indexed_page,
                        }
                    )

            except Exception as e:
                if not self.silence_exceptions:
                    raise ValueError("Error indexing document") from e

                self.logger.warning("Error indexing document : ", exc_info=1)
                document_meta.append(
                    {
                        "indexer": key,
                        "details": {},
                    }
                )

        return document_meta
