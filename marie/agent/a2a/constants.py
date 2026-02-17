"""A2A protocol constants and enumerations.

This module defines protocol-level constants including JSON-RPC methods,
task states, error codes, and well-known paths.
"""

from enum import Enum, IntEnum
from typing import Final


class A2AMethod(str, Enum):
    """A2A JSON-RPC method names."""

    # Message methods
    SEND_MESSAGE = "message/send"
    SEND_MESSAGE_STREAM = "message/stream"

    # Task methods
    GET_TASK = "tasks/get"
    CANCEL_TASK = "tasks/cancel"
    RESUBSCRIBE = "tasks/resubscribe"

    # Push notification methods
    SET_PUSH_CONFIG = "tasks/pushNotificationConfig/set"
    GET_PUSH_CONFIG = "tasks/pushNotificationConfig/get"
    LIST_PUSH_CONFIG = "tasks/pushNotificationConfig/list"
    DELETE_PUSH_CONFIG = "tasks/pushNotificationConfig/delete"

    # Agent card methods
    GET_EXTENDED_CARD = "agent/getAuthenticatedExtendedCard"


class TaskState(str, Enum):
    """A2A task lifecycle states."""

    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"
    COMPLETED = "completed"
    CANCELED = "canceled"
    FAILED = "failed"
    REJECTED = "rejected"
    AUTH_REQUIRED = "auth-required"
    UNKNOWN = "unknown"


class A2AErrorCode(IntEnum):
    """A2A JSON-RPC error codes.

    Standard JSON-RPC 2.0 errors: -32700 to -32600
    A2A-specific errors: -32001 to -32099
    """

    # Standard JSON-RPC errors
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603

    # A2A-specific errors
    TASK_NOT_FOUND = -32001
    TASK_NOT_CANCELABLE = -32002
    PUSH_NOT_SUPPORTED = -32003
    UNSUPPORTED_OPERATION = -32004
    CONTENT_TYPE_NOT_SUPPORTED = -32005
    INVALID_AGENT_RESPONSE = -32006
    EXTENDED_CARD_NOT_CONFIGURED = -32007


# Well-known paths
AGENT_CARD_PATH: Final[str] = "/.well-known/agent.json"
AGENT_CARD_PATH_DEPRECATED: Final[str] = "/agent.json"
EXTENDED_CARD_PATH: Final[str] = "/agent/authenticatedExtendedCard"
DEFAULT_RPC_PATH: Final[str] = "/"

# JSON-RPC version
JSONRPC_VERSION: Final[str] = "2.0"

# Default timeouts (seconds)
DEFAULT_REQUEST_TIMEOUT: Final[int] = 60
DEFAULT_STREAM_TIMEOUT: Final[int] = 300
DEFAULT_DISCOVERY_CACHE_TTL: Final[int] = 3600  # 1 hour

# Terminal task states (no further transitions possible)
TERMINAL_STATES: frozenset[TaskState] = frozenset(
    {
        TaskState.COMPLETED,
        TaskState.CANCELED,
        TaskState.FAILED,
        TaskState.REJECTED,
    }
)
