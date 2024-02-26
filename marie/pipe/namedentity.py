import re
from abc import ABC
from typing import List, Optional

from docarray import DocList

from marie.api.docs import DOC_KEY_INDEXER, DOC_KEY_PAGE_NUMBER
from marie.logging.logger import MarieLogger
from marie.pipe.base import PipelineComponent, PipelineContext, PipelineResult
from marie.utils import filter_node


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
        if not isinstance(document_indexers, dict):
            raise ValueError("document_indexers must be a dictionary")
        self.document_indexers = document_indexers

    def predict(
        self,
        documents: DocList,
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
        self,
        documents: DocList,
        words: List[List[str]] = None,
        boxes: List[List[List[int]]] = None,
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
                self.logger.info(f"Indexing document : {key}")
                indexer = document_indexer["indexer"]

                has_filter = "filter" in document_indexer
                filtered_docs = []
                filtered_words = []
                filtered_boxes = []

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

                for i, document in enumerate(documents):
                    assert DOC_KEY_PAGE_NUMBER in document.tags
                    if "classification" not in document.tags:
                        self.logger.warning(
                            f"Document has no classification tag, adding to filtered documents"
                        )
                        filtered_docs.append(document)
                        filtered_words.append(words[i])
                        filtered_boxes.append(boxes[i])
                        continue

                    classification = str(document.tags["classification"])
                    if filter_type == "regex":
                        if re.search(filter_pattern, classification):
                            self.logger.info(
                                f"Document classification matches filter : {classification} : {filter_pattern}"
                            )
                            filtered_docs.append(document)
                            filtered_words.append(words[i])
                            filtered_boxes.append(boxes[i])
                            continue
                    else:
                        raise NotImplementedError("Exact filter not implemented")

                results = indexer.run(
                    documents=DocList(filtered_docs),
                    words=filtered_words,
                    boxes=filtered_boxes,
                )

                for idx, (rdoc, document) in enumerate(zip(results, filtered_docs)):
                    assert DOC_KEY_INDEXER in document.tags
                    assert DOC_KEY_PAGE_NUMBER in document.tags
                    indexed_values = document.tags[DOC_KEY_INDEXER]
                    filter_node(indexed_values, filters=["page"])

                    meta.append(
                        {
                            # Using string to avoid type conversion issues
                            "page": f"{document.tags[DOC_KEY_PAGE_NUMBER]}",
                            "indexing": indexed_values,
                        }
                    )

                document_meta.append(
                    {
                        "indexer": key,
                        "details": meta,
                    }
                )

            except Exception as e:
                if not self.silence_exceptions:
                    raise ValueError("Error indexing document") from e
                self.logger.warning("Error indexing document : ", exc_info=1)
                document_meta.append(
                    {
                        "indexer": key,
                        "details": [],
                    }
                )

        return document_meta
