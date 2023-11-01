from abc import ABC
from typing import Optional, List

from docarray import DocList

from marie.api.docs import MarieDoc
from marie.logging.logger import MarieLogger
from marie.pipe.base import PipelineComponent, PipelineResult, PipelineContext


class ClassifierPipelineComponent(PipelineComponent, ABC):
    def __init__(
        self, name: str, document_classifiers: dict, logger: MarieLogger = None
    ) -> None:
        """
        :param document_classifiers:
        :param name: Will be passed to base class
        """
        super().__init__(name, logger=logger)
        self.document_classifiers = document_classifiers

    def predict(
        self,
        documents: DocList[MarieDoc],
        context: Optional[PipelineContext] = None,
        *,  # force users to use keyword arguments
        words: List[List[str]] = None,
        boxes: List[List[List[int]]] = None,
    ) -> PipelineResult:

        context["metadata"]["page_classifier"] = self.classify(documents, words, boxes)

        return PipelineResult(documents)

    def classify(
        self,
        documents: DocList[MarieDoc],
        words: List[List[str]],
        boxes: List[List[List[int]]],
    ):
        """
        Classify document at page level

        :param documents: documents to classify
        :param words: words
        :param boxes: boxes
        :return: classification results
        """

        document_meta = []
        for key, document_classifier in self.document_classifiers.items():
            meta = []

            try:
                classified_docs = document_classifier.run(
                    documents=documents, words=words, boxes=boxes
                )

                for idx, document in enumerate(classified_docs):
                    assert "classification" in document.tags
                    classification = document.tags["classification"]
                    # document.tags.pop("classification")

                    assert "label" in classification
                    assert "score" in classification
                    meta.append(
                        {
                            "page": f"{idx}",  # Using string to avoid type conversion issues
                            "classification": classification["label"],
                            "score": round(classification["score"], 4),
                        }
                    )

                self.logger.debug(f"Classification : {meta}")
                document_meta.append(
                    {
                        "classifier": key,
                        "details": meta,
                    }
                )
            except Exception as e:
                self.logger.error(f"Error classifying document : {e}")
                document_meta.append(
                    {
                        "classifier": key,
                        "details": [],
                    }
                )

        return document_meta
