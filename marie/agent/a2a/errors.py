"""A2A-specific exception classes.

This module defines exception types for A2A protocol operations,
mapping to JSON-RPC error codes and providing rich error context.
"""

from typing import Any, Optional

from marie.agent.a2a.constants import A2AErrorCode


class A2AError(Exception):
    """Base exception for all A2A errors."""

    def __init__(
        self,
        message: str,
        code: A2AErrorCode = A2AErrorCode.INTERNAL_ERROR,
        data: Optional[Any] = None,
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.data = data

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-RPC error object."""
        error: dict[str, Any] = {
            "code": int(self.code),
            "message": self.message,
        }
        if self.data is not None:
            error["data"] = self.data
        return error


class A2AClientError(A2AError):
    """Error during A2A client operations (calling external agents)."""

    def __init__(
        self,
        message: str,
        code: A2AErrorCode = A2AErrorCode.INTERNAL_ERROR,
        data: Optional[Any] = None,
        agent_url: Optional[str] = None,
    ):
        super().__init__(message, code, data)
        self.agent_url = agent_url


class A2AServerError(A2AError):
    """Error during A2A server operations (handling incoming requests)."""

    pass


class A2AProtocolError(A2AError):
    """Error in A2A protocol handling (invalid requests, malformed data)."""

    pass


class TaskNotFoundError(A2AError):
    """Task with specified ID was not found."""

    def __init__(self, task_id: str, message: Optional[str] = None):
        super().__init__(
            message or f"Task not found: {task_id}",
            A2AErrorCode.TASK_NOT_FOUND,
            {"task_id": task_id},
        )
        self.task_id = task_id


class TaskNotCancelableError(A2AError):
    """Task cannot be canceled in its current state."""

    def __init__(self, task_id: str, state: str, message: Optional[str] = None):
        super().__init__(
            message or f"Task {task_id} cannot be canceled - current state: {state}",
            A2AErrorCode.TASK_NOT_CANCELABLE,
            {"task_id": task_id, "state": state},
        )
        self.task_id = task_id
        self.state = state


class InvalidRequestError(A2AProtocolError):
    """Invalid JSON-RPC request."""

    def __init__(self, message: str = "Invalid request", data: Optional[Any] = None):
        super().__init__(message, A2AErrorCode.INVALID_REQUEST, data)


class InvalidParamsError(A2AProtocolError):
    """Invalid method parameters."""

    def __init__(self, message: str = "Invalid params", data: Optional[Any] = None):
        super().__init__(message, A2AErrorCode.INVALID_PARAMS, data)


class MethodNotFoundError(A2AProtocolError):
    """Unknown JSON-RPC method."""

    def __init__(self, method: str):
        super().__init__(
            f"Method not found: {method}",
            A2AErrorCode.METHOD_NOT_FOUND,
            {"method": method},
        )
        self.method = method


class UnsupportedOperationError(A2AError):
    """Operation not supported by this agent."""

    def __init__(self, operation: str, message: Optional[str] = None):
        super().__init__(
            message or f"Unsupported operation: {operation}",
            A2AErrorCode.UNSUPPORTED_OPERATION,
            {"operation": operation},
        )
        self.operation = operation


class ContentTypeNotSupportedError(A2AError):
    """Content type not supported by the agent."""

    def __init__(
        self,
        content_type: str,
        supported: list[str],
        message: Optional[str] = None,
    ):
        super().__init__(
            message or f"Content type not supported: {content_type}",
            A2AErrorCode.CONTENT_TYPE_NOT_SUPPORTED,
            {"content_type": content_type, "supported": supported},
        )
        self.content_type = content_type
        self.supported = supported


class AgentDiscoveryError(A2AClientError):
    """Error during agent discovery."""

    def __init__(self, agent_url: str, message: Optional[str] = None):
        super().__init__(
            message or f"Failed to discover agent at {agent_url}",
            A2AErrorCode.INTERNAL_ERROR,
            {"agent_url": agent_url},
            agent_url,
        )
