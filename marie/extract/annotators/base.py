from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from marie.extract.structures.unstructured_document import UnstructuredDocument


class AnnotatorCapabilities(Enum):
    """
    Enum representing the various capabilities an annotator can support.
    """

    EXTRACTOR = "EXTRACTOR"
    SEGMENTER = "SEGMENTER"
    NOOP = "NOOP"  # Does nothing, just validates input or maintains compatibility


class DocumentAnnotator(ABC):
    """
    Abstract base class for all annotators.
    Annotators declare their capabilities and implement annotation logic.
    """

    def __init__(self):
        pass

    @property
    @abstractmethod
    def capabilities(self) -> list[AnnotatorCapabilities]:
        """
        Define a list indicating the annotator's supported capabilities.
        Derived classes must override this to declare their capabilities.
        """
        pass

    def supports_capability(self, capability: AnnotatorCapabilities) -> bool:
        """
        Check if the annotator supports a given capability.
        :param capability: The capability to check, e.g., AnnotatorCapabilities.EXTRACTOR
        :return: True if the annotator supports the capability, False otherwise.
        """
        return capability in self.capabilities

    @abstractmethod
    def annotate(self, document: "UnstructuredDocument", frames: List):
        """
        Perform annotation operations on the provided document.
        Derived classes must override this method.
        """

    async def aannotate(self, document: "UnstructuredDocument", frames: List) -> None:
        """
        Perform annotation operations on the provided document.
        Derived classes must override this method.
        """
        return self.annotate(document, frames)

    @abstractmethod
    def parse_output(self, raw_output: str):
        """
        Parse raw model output into meaningful annotations.
        Derived classes must override this method.
        """
        pass

    def validate_document(self, document: "UnstructuredDocument"):
        """
        Ensure the input document is valid (non-empty content).
        """
        if not document:
            raise ValueError("Document content cannot be empty.")
