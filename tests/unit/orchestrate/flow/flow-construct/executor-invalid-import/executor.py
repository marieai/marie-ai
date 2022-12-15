import some.missing.depdency
from jina import DocumentArray, Executor, requests


class InvalidImportExec(Executor):
    @requests
    def foo(self, docs: DocumentArray, **kwargs):
        for doc in docs:
            doc.text = 'done'
