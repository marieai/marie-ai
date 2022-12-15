from jina import DocumentArray, Executor, requests


class CustomExec(Executor):
    @requests
    def foo(self, docs: DocumentArray, **kwargs):
        for doc in docs:
            doc.text = 'done'
