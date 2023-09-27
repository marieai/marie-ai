from abc import ABC, abstractmethod
from typing import Optional, List

from docarray import DocumentArray

from marie.pipe.base import PipelineComponent


class ClassifierPipelineComponent(PipelineComponent, ABC):
    def __init__(self, name: str, document_classifiers: []) -> None:
        """
        :param document_classifiers:
        :param name: Will be passed to base class
        """
        super().__init__(name)

    def predict(
        self,
        documents: DocumentArray,
        words: Optional[List[List[str]]] = None,
        boxes: Optional[List[List[List[int]]]] = None,
        batch_size: Optional[int] = None,
    ) -> DocumentArray:
        print("ClassifierPipelineComponent.predict()")

        return documents
