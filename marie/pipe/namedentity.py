from abc import ABC, abstractmethod
from typing import Optional, List

from docarray import DocumentArray

from marie.pipe.base import PipelineComponent


class NamedEntityPipelineComponent(PipelineComponent, ABC):
    def __init__(
        self,
        name: str,
    ) -> None:
        """
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
        print("NamedEntityPipelineComponent.predict()")

        return documents
