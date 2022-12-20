from jina import DocumentArray, Executor, requests


class {{exec_name}}(Executor):
    """{{exec_description}}"""
    @requests
    def foo(self, docs: DocumentArray, **kwargs):
        pass