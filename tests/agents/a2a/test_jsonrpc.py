"""Tests for A2A JSON-RPC handling."""

import pytest

from marie.agent.a2a.constants import A2AErrorCode, A2AMethod
from marie.agent.a2a.errors import (
    InvalidParamsError,
    InvalidRequestError,
    MethodNotFoundError,
)
from marie.agent.a2a.jsonrpc import (
    JSONRPCDispatcher,
    create_error_response,
    create_success_response,
)


class TestJSONRPCDispatcher:
    """Tests for JSONRPCDispatcher."""

    @pytest.fixture
    def dispatcher(self):
        """Create a dispatcher with test handlers."""
        d = JSONRPCDispatcher()

        async def echo_handler(params):
            return {"echo": params.get("message", "")}

        async def add_handler(params):
            return {"sum": params.get("a", 0) + params.get("b", 0)}

        d.register("test/echo", echo_handler)
        d.register("test/add", add_handler)

        return d

    @pytest.mark.asyncio
    async def test_dispatch_success(self, dispatcher):
        """Test successful request dispatch."""
        request = {
            "jsonrpc": "2.0",
            "id": "1",
            "method": "test/echo",
            "params": {"message": "hello"},
        }

        response = await dispatcher.dispatch(request)

        assert response.id == "1"
        assert response.result == {"echo": "hello"}

    @pytest.mark.asyncio
    async def test_dispatch_with_json_string(self, dispatcher):
        """Test dispatch with JSON string input."""
        request = '{"jsonrpc": "2.0", "id": "1", "method": "test/echo", "params": {"message": "test"}}'

        response = await dispatcher.dispatch(request)

        assert response.result == {"echo": "test"}

    @pytest.mark.asyncio
    async def test_dispatch_method_not_found(self, dispatcher):
        """Test dispatch with unknown method."""
        request = {
            "jsonrpc": "2.0",
            "id": "1",
            "method": "unknown/method",
            "params": {},
        }

        response = await dispatcher.dispatch(request)

        assert response.error is not None
        assert response.error.code == A2AErrorCode.METHOD_NOT_FOUND

    @pytest.mark.asyncio
    async def test_dispatch_invalid_json(self, dispatcher):
        """Test dispatch with invalid JSON."""
        request = "not valid json"

        response = await dispatcher.dispatch(request)

        assert response.error is not None
        assert response.error.code == A2AErrorCode.INVALID_REQUEST

    @pytest.mark.asyncio
    async def test_dispatch_missing_jsonrpc_version(self, dispatcher):
        """Test dispatch without jsonrpc version."""
        request = {
            "id": "1",
            "method": "test/echo",
        }

        response = await dispatcher.dispatch(request)

        assert response.error is not None
        assert response.error.code == A2AErrorCode.INVALID_REQUEST

    @pytest.mark.asyncio
    async def test_dispatch_missing_method(self, dispatcher):
        """Test dispatch without method."""
        request = {
            "jsonrpc": "2.0",
            "id": "1",
        }

        response = await dispatcher.dispatch(request)

        assert response.error is not None
        assert response.error.code == A2AErrorCode.INVALID_REQUEST

    def test_is_streaming_method(self, dispatcher):
        """Test streaming method detection."""
        async def stream_handler(params):
            yield {"data": "chunk"}

        dispatcher.register("test/stream", stream_handler, streaming=True)

        assert dispatcher.is_streaming_method("test/stream") is True
        assert dispatcher.is_streaming_method("test/echo") is False

    @pytest.mark.asyncio
    async def test_dispatch_streaming(self, dispatcher):
        """Test streaming dispatch."""
        async def stream_handler(params):
            for i in range(3):
                yield {"count": i}

        dispatcher.register("test/stream", stream_handler, streaming=True)

        request = {
            "jsonrpc": "2.0",
            "id": "1",
            "method": "test/stream",
            "params": {},
        }

        events = []
        async for event in dispatcher.dispatch_streaming(request):
            events.append(event)

        assert len(events) == 3


class TestResponseHelpers:
    """Tests for response helper functions."""

    def test_create_success_response(self):
        response = create_success_response("req-1", {"data": "value"})

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == "req-1"
        assert response["result"] == {"data": "value"}

    def test_create_error_response(self):
        response = create_error_response(
            "req-1",
            A2AErrorCode.TASK_NOT_FOUND,
            "Task not found",
            {"task_id": "123"},
        )

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == "req-1"
        assert response["error"]["code"] == -32001
        assert response["error"]["message"] == "Task not found"
        assert response["error"]["data"]["task_id"] == "123"

    def test_create_error_response_without_data(self):
        response = create_error_response(
            None,
            A2AErrorCode.INTERNAL_ERROR,
            "Internal error",
        )

        assert response["id"] is None
        assert "data" not in response["error"]


class TestA2AMethod:
    """Tests for A2AMethod enum."""

    def test_method_values(self):
        assert A2AMethod.SEND_MESSAGE.value == "message/send"
        assert A2AMethod.SEND_MESSAGE_STREAM.value == "message/stream"
        assert A2AMethod.GET_TASK.value == "tasks/get"
        assert A2AMethod.CANCEL_TASK.value == "tasks/cancel"

    def test_method_in_dispatcher(self):
        dispatcher = JSONRPCDispatcher()

        async def handler(params):
            return {}

        dispatcher.register(A2AMethod.SEND_MESSAGE, handler)

        # Should be registered by value
        assert "message/send" in dispatcher._handlers
