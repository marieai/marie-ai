from abc import ABC
from typing import List, Optional

from docarray import DocList

from marie.api.docs import DOC_KEY_CLASSIFICATION, DOC_KEY_PAGE_NUMBER, MarieDoc
from marie.logging.logger import MarieLogger
from marie.pipe.base import PipelineComponent, PipelineContext, PipelineResult


class ClassifierPipelineComponent(PipelineComponent, ABC):
    def __init__(
        self,
        name: str,
        document_classifiers: dict,
        logger: MarieLogger = None,
        silence_exceptions: bool = False,
    ) -> None:
        """
        :param document_classifiers: A dictionary containing document classifiers.
        :param name: Will be passed to base class for logging purposes
        """
        super().__init__(name, logger=logger)
        if not isinstance(document_classifiers, dict):
            raise ValueError("document_classifiers must be a dictionary")
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
        for key, classifier in self.document_classifiers.items():
            if "classifier" not in classifier or "filter" not in classifier:
                raise ValueError(f"Invalid classifier : {classifier}")

            document_classifier = classifier["classifier"]
            filter = classifier["filter"]

            meta = []

            try:
                classified_docs = document_classifier.run(
                    documents=documents, words=words, boxes=boxes
                )

                for idx, document in enumerate(classified_docs):
                    assert DOC_KEY_PAGE_NUMBER in document.tags
                    assert DOC_KEY_CLASSIFICATION in document.tags
                    classification = document.tags[DOC_KEY_CLASSIFICATION]

                    assert "label" in classification
                    assert "score" in classification
                    meta.append(
                        {
                            "page": f"{document.tags[DOC_KEY_PAGE_NUMBER]}",
                            # Using string to avoid type conversion issues
                            "classification": str(classification["label"]),
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
