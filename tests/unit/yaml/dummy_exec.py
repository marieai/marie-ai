from marie import DocumentArray

from marie import requests
from marie.serve.executors import BaseExecutor


class DummyExternalIndexer(BaseExecutor):
    @requests
    def index(self, docs: DocumentArray, **kwargs):
        for doc in docs:
            doc.text = "indexed"
