"""
Host Function Implementations for Wasm Components.

These classes implement the WIT interfaces that Wasm modules can import.
Each interface is gated by permissions for security.
"""

import logging
from typing import Any, Callable, Optional, Protocol
from urllib.parse import urlparse

from marie_wasm.types import Permissions

logger = logging.getLogger(__name__)


class HttpClient(Protocol):
    """Protocol for HTTP client implementation."""

    def request(
        self,
        method: str,
        url: str,
        headers: list[tuple[str, str]],
        body: Optional[bytes],
    ) -> tuple[int, list[tuple[str, str]], bytes]:
        """Make HTTP request, return (status, headers, body)."""
        ...


class KeyValueStore(Protocol):
    """Protocol for key-value store implementation."""

    def get(self, key: str) -> Optional[bytes]:
        """Get value by key."""
        ...

    def set(self, key: str, value: bytes, ttl_seconds: Optional[int] = None) -> None:
        """Set value with optional TTL."""
        ...

    def delete(self, key: str) -> None:
        """Delete key."""
        ...


class EventEmitter(Protocol):
    """Protocol for event emission."""

    def emit(self, event_type: str, payload: str) -> None:
        """Emit an event."""
        ...


class HostImplementations:
    """
    WIT import implementations with permission checking.

    These functions are called by Wasm modules and are subject to
    capability-based security. Each operation is validated against
    the permissions granted to the module.
    """

    def __init__(
        self,
        permissions: Permissions,
        credentials: Optional[dict[str, str]] = None,
        http_client: Optional[HttpClient] = None,
        kv_store: Optional[KeyValueStore] = None,
        logger_func: Optional[Callable[[str, str], None]] = None,
        event_emitter: Optional[EventEmitter] = None,
        execution_id: Optional[str] = None,
    ):
        """
        Initialize host implementations.

        Args:
            permissions: Capability permissions for this execution
            credentials: Secret credentials available to the module
            http_client: HTTP client for outbound requests
            kv_store: Key-value store backend
            logger_func: Logging function (level, message) -> None
            event_emitter: Event emission backend
            execution_id: Execution ID for logging/correlation
        """
        self.permissions = permissions
        self.credentials = credentials or {}
        self._http_client = http_client
        self._kv_store = kv_store
        self._logger_func = logger_func or self._default_logger
        self._event_emitter = event_emitter
        self.execution_id = execution_id

    def _default_logger(self, level: str, message: str) -> None:
        """Default logging implementation."""
        log_func = getattr(logger, level.lower(), logger.info)
        prefix = f"[wasm:{self.execution_id}] " if self.execution_id else "[wasm] "
        log_func(f"{prefix}{message}")

    # =========================================
    # http-client interface
    # =========================================

    def http_request(self, request: dict) -> dict:
        """
        Make an HTTP request with host validation.

        WIT signature:
            request: func(req: http-request) -> result<http-response, string>

        Args:
            request: Dict with method, url, headers, body

        Returns:
            Dict with status, headers, body on success, or error string
        """
        if not self.permissions.allow_http:
            return {"error": "HTTP requests not permitted"}

        url = request.get("url", "")
        method = request.get("method", "GET").upper()
        headers = request.get("headers", [])
        body = request.get("body")

        # Validate host against allowed list
        try:
            parsed = urlparse(url)
            host = parsed.netloc
            # Remove port if present
            if ":" in host:
                host = host.split(":")[0]
        except Exception as e:
            return {"error": f"Invalid URL: {e}"}

        if not self.permissions.is_host_allowed(host):
            logger.warning(f"HTTP request blocked: host '{host}' not in allowed list")
            return {"error": f"Host not allowed: {host}"}

        if not self._http_client:
            return {"error": "HTTP client not configured"}

        try:
            self._logger_func("debug", f"HTTP {method} {url}")

            status, resp_headers, resp_body = self._http_client.request(
                method=method,
                url=url,
                headers=headers,
                body=body,
            )

            return {
                "ok": {
                    "status": status,
                    "headers": resp_headers,
                    "body": list(resp_body) if resp_body else [],
                }
            }

        except Exception as e:
            logger.error(f"HTTP request failed: {e}")
            return {"error": f"HTTP request failed: {e}"}

    # =========================================
    # secrets interface
    # =========================================

    def secrets_get(self, name: str) -> Optional[str]:
        """
        Get a secret by name with validation.

        WIT signature:
            get: func(name: string) -> option<string>

        Args:
            name: Secret name

        Returns:
            Secret value or None if not available
        """
        if not self.permissions.allow_secrets:
            logger.debug(f"Secret access denied: secrets not permitted")
            return None

        if not self.permissions.is_secret_allowed(name):
            logger.warning(f"Secret access denied: '{name}' not in allowed list")
            return None

        secret = self.credentials.get(name)
        if secret:
            self._logger_func("debug", f"Secret accessed: {name}")
        return secret

    # =========================================
    # key-value interface
    # =========================================

    def _prefix_key(self, key: str) -> str:
        """Apply KV prefix to isolate node data."""
        return f"{self.permissions.kv_prefix}{key}"

    def kv_get(self, key: str) -> Optional[list[int]]:
        """
        Get a value from the key-value store.

        WIT signature:
            get: func(key: string) -> option<list<u8>>

        Args:
            key: Storage key

        Returns:
            Value bytes as list of ints, or None if not found
        """
        if not self.permissions.allow_kv:
            return None

        if not self._kv_store:
            return None

        try:
            prefixed_key = self._prefix_key(key)
            value = self._kv_store.get(prefixed_key)
            if value is not None:
                return list(value)
            return None
        except Exception as e:
            logger.error(f"KV get failed: {e}")
            return None

    def kv_set(
        self, key: str, value: list[int], ttl_seconds: Optional[int] = None
    ) -> Optional[str]:
        """
        Set a value in the key-value store.

        WIT signature:
            set: func(key: string, value: list<u8>, ttl-seconds: option<u32>)
                 -> result<_, string>

        Args:
            key: Storage key
            value: Value bytes as list of ints
            ttl_seconds: Optional TTL in seconds

        Returns:
            None on success, error string on failure
        """
        if not self.permissions.allow_kv:
            return "KV store not permitted"

        if not self._kv_store:
            return "KV store not configured"

        try:
            prefixed_key = self._prefix_key(key)
            self._kv_store.set(prefixed_key, bytes(value), ttl_seconds)
            self._logger_func("debug", f"KV set: {key}")
            return None  # Success
        except Exception as e:
            logger.error(f"KV set failed: {e}")
            return f"KV set failed: {e}"

    def kv_delete(self, key: str) -> Optional[str]:
        """
        Delete a key from the key-value store.

        WIT signature:
            delete: func(key: string) -> result<_, string>

        Args:
            key: Storage key

        Returns:
            None on success, error string on failure
        """
        if not self.permissions.allow_kv:
            return "KV store not permitted"

        if not self._kv_store:
            return "KV store not configured"

        try:
            prefixed_key = self._prefix_key(key)
            self._kv_store.delete(prefixed_key)
            self._logger_func("debug", f"KV delete: {key}")
            return None  # Success
        except Exception as e:
            logger.error(f"KV delete failed: {e}")
            return f"KV delete failed: {e}"

    # =========================================
    # logging interface
    # =========================================

    def log(self, level: str, message: str) -> None:
        """
        Log a message at the specified level.

        WIT signature:
            log: func(level: log-level, message: string)

        Args:
            level: Log level (debug, info, warn, error)
            message: Log message
        """
        # Map WIT enum to string
        level_str = level.lower() if isinstance(level, str) else str(level)
        self._logger_func(level_str, message)

    # =========================================
    # events interface
    # =========================================

    def emit(self, event_type: str, payload: str) -> None:
        """
        Emit an event to the workflow engine.

        WIT signature:
            emit: func(event-type: string, payload: string)

        Args:
            event_type: Type of event
            payload: JSON payload
        """
        self._logger_func("debug", f"Event emitted: {event_type}")

        if self._event_emitter:
            try:
                self._event_emitter.emit(event_type, payload)
            except Exception as e:
                logger.error(f"Event emission failed: {e}")

    # =========================================
    # Binding helpers for Wasmtime
    # =========================================

    def get_bindings(self) -> dict[str, dict[str, Any]]:
        """
        Get all host function bindings for Wasmtime linker.

        Returns dict mapping interface names to function dicts.
        """
        return {
            "marie:node/http-client": {
                "request": self.http_request,
            },
            "marie:node/secrets": {
                "get": self.secrets_get,
            },
            "marie:node/key-value": {
                "get": self.kv_get,
                "set": self.kv_set,
                "delete": self.kv_delete,
            },
            "marie:node/logging": {
                "log": self.log,
            },
            "marie:node/events": {
                "emit": self.emit,
            },
        }


class DefaultHttpClient:
    """
    Default HTTP client implementation using httpx.

    This is a simple sync implementation for basic use cases.
    Production deployments should use async httpx or aiohttp.
    """

    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout
        self._client: Optional[Any] = None

    def _get_client(self) -> Any:
        """Lazy-initialize httpx client."""
        if self._client is None:
            try:
                import httpx

                self._client = httpx.Client(timeout=self.timeout)
            except ImportError:
                raise RuntimeError("httpx not installed")
        return self._client

    def request(
        self,
        method: str,
        url: str,
        headers: list[tuple[str, str]],
        body: Optional[bytes],
    ) -> tuple[int, list[tuple[str, str]], bytes]:
        """Make HTTP request."""
        client = self._get_client()

        response = client.request(
            method=method,
            url=url,
            headers=dict(headers),
            content=body,
        )

        resp_headers = [(k, v) for k, v in response.headers.items()]
        return response.status_code, resp_headers, response.content

    def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            self._client.close()
            self._client = None


class InMemoryKVStore:
    """
    Simple in-memory key-value store for testing.

    Does not support TTL - all values persist until deleted.
    """

    def __init__(self) -> None:
        self._store: dict[str, bytes] = {}

    def get(self, key: str) -> Optional[bytes]:
        """Get value by key."""
        return self._store.get(key)

    def set(self, key: str, value: bytes, ttl_seconds: Optional[int] = None) -> None:
        """Set value (TTL ignored in this implementation)."""
        self._store[key] = value

    def delete(self, key: str) -> None:
        """Delete key."""
        self._store.pop(key, None)

    def clear(self) -> None:
        """Clear all values."""
        self._store.clear()
