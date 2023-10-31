from abc import ABC, abstractmethod
from collections.abc import MutableMapping
from typing import Optional

from docarray import DocList

from marie.api.docs import MarieDoc
from marie.logging.logger import MarieLogger


class PipelineContext(MutableMapping):
    """
    A :class:`PipelineContext` provides access to info about a pipeline.
    """

    def __init__(self, pipeline_id: str, *args, **kwargs):
        self.pipeline_id = pipeline_id
        self._state = {}
        self._state.update(dict(*args, **kwargs))

    def __getitem__(self, key):
        return self._state[self.keytransform(key)]

    def __setitem__(self, key, value):
        self._state[self.keytransform(key)] = value

    def __delitem__(self, key):
        del self._state[self.keytransform(key)]

    def __iter__(self):
        return iter(self._state)

    def __len__(self):
        return len(self._state)

    @staticmethod
    def keytransform(key: str):
        return key.lower()


class PipelineResult(object):
    """A :class:`PipelineResult` provides access to info about a pipeline."""

    def __init__(self, state):
        self._state = state

    @property
    def state(self):
        """Return the current state of the pipeline execution."""
        return self._state


class PipelineComponent(ABC):
    """
    Base class for pipeline components. Pipeline components are the parts that make up a pipeline.
    """

    def __init__(self, name: str, logger: MarieLogger = None):
        """
        :param name: The name of the pipeline component. The name will be used to identify a pipeline component in a
                     pipeline. Use something that describe the task of the pipeline.
        """
        self.name = name
        self.timer_on = False
        self.logger = logger or MarieLogger(context=self.__class__.__name__)

    @abstractmethod
    def predict(
        self,
        documents: DocList[MarieDoc],
        context: Optional[PipelineContext] = None,
        **kwargs,
    ) -> PipelineResult:
        pass

    def run(
        self,
        documents: DocList[MarieDoc],
        context: Optional[PipelineContext] = None,
        **kwargs,
    ) -> PipelineResult:
        """
        Run the document classifier on the given documents.

        :param context: The context of the pipeline. The context is used to share data between pipeline components.
        :param documents: The documents to process.
        :return:
        """
        if documents is None:
            raise ValueError("Documents cannot be None")
        results = self.predict(documents, context, **kwargs)

        document_id = [document.id for document in documents]
        self.logger.info(f"Processed documents with IDs: {document_id}")
        return results
