import pytest

from marie import Executor, requests
from docarray import DocumentArray, Document

from marie.clients.request import request_generator
from marie.logging.logger import MarieLogger

from marie.parsers import set_pod_parser
from marie.serve.runtimes.request_handlers.data_request_handler import (
    DataRequestHandler,
)


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
    args = set_pod_parser().parse_args(["--uses", "NewDocsExecutor"])
    handler = DataRequestHandler(args, logger, executor=NewDocsExecutor())
    req = list(
        request_generator(
            "/", DocumentArray([Document(text="input document") for _ in range(10)])
        )
    )[0]

    assert len(req.docs) == 10
    response = await handler.handle(requests=[req])

    assert len(response.docs) == 1
    assert response.docs[0].text == 'new document'


@pytest.mark.asyncio
async def test_aync_data_request_handler_new_docs(logger):
    args = set_pod_parser().parse_args(['--uses', 'AsyncNewDocsExecutor'])
    handler = DataRequestHandler(args, logger, executor=AsyncNewDocsExecutor())
    req = list(
        request_generator(
            '/', DocumentArray([Document(text='input document') for _ in range(10)])
        )
    )[0]
    assert len(req.docs) == 10
    response = await handler.handle(requests=[req])

    assert len(response.docs) == 1
    assert response.docs[0].text == 'new document'


@pytest.mark.asyncio
async def test_data_request_handler_change_docs(logger):
    args = set_pod_parser().parse_args(['--uses', 'ChangeDocsExecutor'])
    handler = DataRequestHandler(args, logger, executor=ChangeDocsExecutor())

    req = list(
        request_generator(
            '/', DocumentArray([Document(text='input document') for _ in range(10)])
        )
    )[0]
    assert len(req.docs) == 10
    response = await handler.handle(requests=[req])

    assert len(response.docs) == 10
    for doc in response.docs:
        assert doc.text == 'changed document'


@pytest.mark.asyncio
async def test_data_request_handler_change_docs_from_partial_requests(logger):
    NUM_PARTIAL_REQUESTS = 5
    args = set_pod_parser().parse_args(['--uses', 'MergeChangeDocsExecutor'])
    handler = DataRequestHandler(args, logger, executor=MergeChangeDocsExecutor())

    partial_reqs = [
        list(
            request_generator(
                '/', DocumentArray([Document(text='input document') for _ in range(10)])
            )
        )[0]
    ] * NUM_PARTIAL_REQUESTS
    assert len(partial_reqs) == 5
    assert len(partial_reqs[0].docs) == 10
    response = await handler.handle(requests=partial_reqs)

    assert len(response.docs) == 10 * NUM_PARTIAL_REQUESTS
    for doc in response.docs:
        assert doc.text == 'changed document'


@pytest.mark.asyncio
async def test_data_request_handler_clear_docs(logger):
    args = set_pod_parser().parse_args(['--uses', 'ClearDocsExecutor'])
    handler = DataRequestHandler(args, logger, executor=ClearDocsExecutor())

    req = list(
        request_generator(
            '/', DocumentArray([Document(text='input document') for _ in range(10)])
        )
    )[0]
    assert len(req.docs) == 10
    response = await handler.handle(requests=[req])

    assert len(response.docs) == 0
