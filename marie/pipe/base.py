from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List

from docarray import DocumentArray


class PipelineContext:
    def __init__(self):
        print("PipelineContext.__init__()")


class PipelineResult:
    success: bool
    error: str = None


class PipelineComponent(ABC):
    """
    Base class for pipeline components. Pipeline components are the parts that make up a pipeline.
    """

    def __init__(self, name: str):
        """
        :param name: The name of the pipeline component. The name will be used to identify a pipeline component in a
                     pipeline. Use something that describe the task of the pipeline.
        """
        self.name = name
        self.timer_on = False

    @abstractmethod
    def predict(
        self,
        documents: DocumentArray,
        words: Optional[List[List[str]]] = None,
        boxes: Optional[List[List[List[int]]]] = None,
        batch_size: Optional[int] = None,
    ) -> DocumentArray:
        pass

    def run(
        self,
        context: PipelineContext,
        documents: DocumentArray,
        words: Optional[List[List[str]]] = None,
        boxes: Optional[List[List[List[int]]]] = None,
        batch_size: Optional[int] = None,
    ) -> DocumentArray:
        """
        Run the document classifier on the given documents.

        :param documents:
        :param words:
        :param boxes:
        :param batch_size:
        :return:
        """
        if documents:
            results = self.predict(context=context, documents=documents)
        else:
            results = DocumentArray()

        document_id = [document.id for document in documents]

        # output = {"documents": results}
        self.logger.info(f"Classified documents with IDs: {document_id}")
        return results
