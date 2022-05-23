import pytest

# from docarray import Document, DocumentArray
from marie import Executor, requests
from marie import DocumentArray, Document
from marie.logging.logger import MarieLogger

# from marie.parsers import set_pod_parser
from marie.serve.runtimes.request_handlers.data_request_handler import (
    DataRequestHandler,
)
# from jina.clients.request import request_generator


class NewDocsExecutor(Executor):
    @requests
    def foo(self, docs, **kwargs):
        return DocumentArray([Document(text="new document")])


class AsyncNewDocsExecutor(Executor):
    @requests
    async def foo(self, docs, **kwargs):
        return DocumentArray([Document(text="new document")])


class ChangeDocsExecutor(Executor):
    @requests
    def foo(self, docs, **kwargs):
        for doc in docs:
            doc.text = "changed document"


class MergeChangeDocsExecutor(Executor):
    @requests
    def foo(self, docs, **kwargs):
        for doc in docs:
            doc.text = "changed document"
        return docs


class ClearDocsExecutor(Executor):
    @requests
    def foo(self, docs, **kwargs):
        docs.clear()


@pytest.fixture()
def logger():
    return MarieLogger("data request handler")


@pytest.mark.asyncio
async def test_data_request_handler_new_docs(logger):
    args = set_pod_parser().parse_args(['--uses', 'NewDocsExecutor'])
    handler = DataRequestHandler(args, logger)
    req = list(
        request_generator(
            '/', DocumentArray([Document(text='input document') for _ in range(10)])
        )
    )[0]
    assert len(req.docs) == 10
    response = await handler.handle(requests=[req])

    assert len(response.docs) == 1
    assert response.docs[0].text == 'new document'
