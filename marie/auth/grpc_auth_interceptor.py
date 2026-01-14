"""
gRPC Authentication Interceptor.

Extracts API key from gRPC metadata and validates via existing APIKeyManager.
Supports both "Bearer <token>" and raw token formats.
"""

import logging
from typing import Callable, Optional, Tuple

import grpc

from marie.auth.api_key_manager import APIKeyManager

logger = logging.getLogger(__name__)


class GrpcAuthInterceptor(grpc.aio.ServerInterceptor):
    """
    gRPC server interceptor for API key authentication.

    Extracts API key from metadata and validates via existing APIKeyManager.
    Supports both "Bearer <token>" and raw token formats for consistency
    with HTTP authentication patterns.
    """

    METADATA_KEY = "authorization"

    def __init__(self, api_key_manager: Optional[type] = None):
        """
        Initialize the interceptor.

        Args:
            api_key_manager: The APIKeyManager class to use for validation.
                           Defaults to the global APIKeyManager.
        """
        self._manager = api_key_manager or APIKeyManager

    async def intercept_service(
        self,
        continuation: Callable,
        handler_call_details: grpc.HandlerCallDetails,
    ):
        """
        Intercept incoming gRPC calls to validate authentication.

        Args:
            continuation: The next interceptor or the actual RPC handler.
            handler_call_details: Details about the incoming call.

        Returns:
            The result of continuation if authenticated, or an error handler.
        """
        # Extract metadata
        metadata = dict(handler_call_details.invocation_metadata)
        auth_value = metadata.get(self.METADATA_KEY, "")

        # Support "Bearer <token>" format (consistent with HTTP)
        token = self._extract_token(auth_value)

        # Validate token
        if not token or not self._manager.is_valid(token):
            logger.warning(
                f"gRPC auth failed: invalid or missing token "
                f"(method={handler_call_details.method})"
            )
            return grpc.unary_unary_rpc_method_handler(self._unauthenticated_handler)

        # Token is valid - continue to actual handler
        return await continuation(handler_call_details)

    def _extract_token(self, auth_value: str) -> str:
        """
        Extract token from authorization header value.

        Supports:
        - "Bearer <token>" format
        - Raw token format

        Args:
            auth_value: The authorization header value.

        Returns:
            The extracted token, or empty string if not found.
        """
        if not auth_value:
            return ""

        if auth_value.startswith("Bearer "):
            return auth_value[7:]

        # Try lowercase "bearer" as well
        if auth_value.lower().startswith("bearer "):
            return auth_value[7:]

        return auth_value

    async def _unauthenticated_handler(
        self,
        request,
        context: grpc.aio.ServicerContext,
    ):
        """Handler that returns UNAUTHENTICATED error."""
        await context.abort(
            grpc.StatusCode.UNAUTHENTICATED,
            "Invalid or missing API key",
        )


def extract_api_key_from_context(
    context: grpc.aio.ServicerContext,
) -> Optional[str]:
    """
    Extract API key from gRPC context metadata.

    Use this in servicer methods to get the authenticated API key
    for authorization or logging purposes.

    Args:
        context: The gRPC servicer context.

    Returns:
        The API key if present, None otherwise.
    """
    metadata = dict(context.invocation_metadata())
    auth_value = metadata.get("authorization", "")

    if not auth_value:
        return None

    if auth_value.startswith("Bearer "):
        return auth_value[7:]

    if auth_value.lower().startswith("bearer "):
        return auth_value[7:]

    return auth_value if auth_value else None


def extract_metadata_value(
    context: grpc.aio.ServicerContext,
    key: str,
) -> Optional[str]:
    """
    Extract a specific metadata value from gRPC context.

    Args:
        context: The gRPC servicer context.
        key: The metadata key to extract.

    Returns:
        The metadata value if present, None otherwise.
    """
    metadata = dict(context.invocation_metadata())
    return metadata.get(key)
