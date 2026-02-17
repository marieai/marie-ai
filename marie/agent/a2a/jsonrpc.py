"""JSON-RPC 2.0 handler and dispatcher for A2A protocol.

This module provides the JSON-RPC request routing and response formatting
for A2A protocol endpoints.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Coroutine, Optional, Union

from marie.agent.a2a.constants import JSONRPC_VERSION, A2AErrorCode, A2AMethod
from marie.agent.a2a.errors import (
    A2AError,
    InvalidParamsError,
    InvalidRequestError,
    MethodNotFoundError,
)
from marie.agent.a2a.types import (
    JSONRPCError,
    JSONRPCErrorResponse,
    JSONRPCRequest,
    JSONRPCSuccessResponse,
)

logger = logging.getLogger(__name__)

# Type alias for handler functions
Handler = Callable[[dict[str, Any]], Coroutine[Any, Any, Any]]


class JSONRPCDispatcher:
    """JSON-RPC 2.0 request dispatcher.

    Routes incoming JSON-RPC requests to registered handlers based on
    the method name. Handles error responses and validation.
    """

    def __init__(self) -> None:
        self._handlers: dict[str, Handler] = {}
        self._streaming_handlers: dict[str, Handler] = {}

    def register(
        self,
        method: Union[str, A2AMethod],
        handler: Handler,
        streaming: bool = False,
    ) -> None:
        """Register a handler for a JSON-RPC method.

        Args:
            method: The method name or A2AMethod enum.
            handler: Async function to handle requests for this method.
            streaming: Whether this handler returns a streaming response.
        """
        method_name = method.value if isinstance(method, A2AMethod) else method
        if streaming:
            self._streaming_handlers[method_name] = handler
        else:
            self._handlers[method_name] = handler

    def is_streaming_method(self, method: str) -> bool:
        """Check if a method returns a streaming response."""
        return method in self._streaming_handlers

    async def dispatch(
        self,
        request: Union[str, bytes, dict[str, Any]],
    ) -> Union[JSONRPCSuccessResponse, JSONRPCErrorResponse]:
        """Dispatch a JSON-RPC request.

        Args:
            request: The request as JSON string, bytes, or parsed dict.

        Returns:
            JSON-RPC response (success or error).
        """
        request_id: Optional[Union[str, int]] = None

        try:
            # Parse request
            parsed = self._parse_request(request)
            request_id = parsed.id

            # Validate and get handler
            handler = self._get_handler(parsed.method)

            # Execute handler
            result = await handler(parsed.params or {})

            return JSONRPCSuccessResponse(
                id=request_id,
                result=result,
            )

        except A2AError as e:
            logger.warning(f"A2A error: {e.message}", exc_info=True)
            return self._error_response(request_id, e)
        except Exception as e:
            logger.exception(f"Unexpected error in JSON-RPC dispatch: {e}")
            return self._error_response(
                request_id,
                A2AError(str(e), A2AErrorCode.INTERNAL_ERROR),
            )

    async def dispatch_streaming(
        self,
        request: Union[str, bytes, dict[str, Any]],
    ):
        """Dispatch a streaming JSON-RPC request.

        Args:
            request: The request as JSON string, bytes, or parsed dict.

        Yields:
            Events from the streaming handler.
        """
        request_id: Optional[Union[str, int]] = None

        try:
            parsed = self._parse_request(request)
            request_id = parsed.id

            handler = self._get_handler(parsed.method, streaming=True)

            async for event in handler(parsed.params or {}):
                yield event

        except A2AError as e:
            logger.warning(f"A2A streaming error: {e.message}")
            yield self._error_response(request_id, e)
        except Exception as e:
            logger.exception(f"Unexpected error in streaming dispatch: {e}")
            yield self._error_response(
                request_id,
                A2AError(str(e), A2AErrorCode.INTERNAL_ERROR),
            )

    def _parse_request(
        self,
        request: Union[str, bytes, dict[str, Any]],
    ) -> JSONRPCRequest:
        """Parse and validate a JSON-RPC request."""
        if isinstance(request, (str, bytes)):
            try:
                data = json.loads(request)
            except json.JSONDecodeError as e:
                raise InvalidRequestError(f"Invalid JSON: {e}")
        else:
            data = request

        if not isinstance(data, dict):
            raise InvalidRequestError("Request must be a JSON object")

        # Validate required fields
        if data.get("jsonrpc") != JSONRPC_VERSION:
            raise InvalidRequestError("Invalid or missing jsonrpc version")

        if "method" not in data:
            raise InvalidRequestError("Missing method field")

        try:
            return JSONRPCRequest(**data)
        except Exception as e:
            raise InvalidRequestError(f"Invalid request structure: {e}")

    def _get_handler(self, method: str, streaming: bool = False) -> Handler:
        """Get the handler for a method."""
        handlers = self._streaming_handlers if streaming else self._handlers

        if method not in handlers:
            # Check if it's in the other handler set
            other = self._handlers if streaming else self._streaming_handlers
            if method in other:
                raise InvalidRequestError(
                    f"Method {method} requires {'streaming' if not streaming else 'non-streaming'} endpoint"
                )
            raise MethodNotFoundError(method)

        return handlers[method]

    def _error_response(
        self,
        request_id: Optional[Union[str, int]],
        error: A2AError,
    ) -> JSONRPCErrorResponse:
        """Create an error response."""
        return JSONRPCErrorResponse(
            id=request_id,
            error=JSONRPCError(
                code=int(error.code),
                message=error.message,
                data=error.data,
            ),
        )


def create_success_response(
    request_id: Optional[Union[str, int]],
    result: Any,
) -> dict[str, Any]:
    """Create a JSON-RPC success response dict."""
    return {
        "jsonrpc": JSONRPC_VERSION,
        "id": request_id,
        "result": result,
    }


def create_error_response(
    request_id: Optional[Union[str, int]],
    code: A2AErrorCode,
    message: str,
    data: Optional[Any] = None,
) -> dict[str, Any]:
    """Create a JSON-RPC error response dict."""
    error: dict[str, Any] = {
        "code": int(code),
        "message": message,
    }
    if data is not None:
        error["data"] = data

    return {
        "jsonrpc": JSONRPC_VERSION,
        "id": request_id,
        "error": error,
    }
