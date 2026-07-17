"""Middleware system for agent execution.

Provides middleware protocol and built-in middleware implementations.
"""

from marie.agent.middleware.audit import AuditMiddleware
from marie.agent.middleware.content_filter import (
    ContentFilterError,
    ContentFilterMiddleware,
)
from marie.agent.middleware.debug_capture import DebugCaptureMiddleware
from marie.agent.middleware.protocol import (
    BaseMiddleware,
    MiddlewareList,
    RunMiddlewareProtocol,
)
from marie.agent.middleware.secrets_detection import (
    SecretsDetectionError,
    SecretsDetectionMiddleware,
)
from marie.agent.middleware.trajectory import TrajectoryMiddleware

__all__ = [
    # Protocol
    "BaseMiddleware",
    "MiddlewareList",
    "RunMiddlewareProtocol",
    # Built-in middleware
    "AuditMiddleware",
    "ContentFilterError",
    "ContentFilterMiddleware",
    "DebugCaptureMiddleware",
    "SecretsDetectionError",
    "SecretsDetectionMiddleware",
    "TrajectoryMiddleware",
]
