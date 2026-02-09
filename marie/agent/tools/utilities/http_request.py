"""HTTP request tool for agent framework.

Provides a full-featured HTTP client with authentication, retries,
timeout handling, and response format detection.
"""

from __future__ import annotations

import asyncio
import json
import ssl
import urllib.error
import urllib.parse
import urllib.request
from enum import Enum
from http.client import HTTPResponse
from typing import Any, Literal

from pydantic import BaseModel, Field

from marie.agent.tools.base import AgentTool, ToolMetadata, ToolOutput


class HttpMethod(str, Enum):
    """Supported HTTP methods."""

    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"


class AuthType(str, Enum):
    """Supported authentication types."""

    NONE = "none"
    BASIC = "basic"
    BEARER = "bearer"
    HEADER = "header"
    QUERY = "query"


class ContentType(str, Enum):
    """Supported content types for request body."""

    JSON = "json"
    FORM = "form"
    RAW = "raw"


class ResponseFormat(str, Enum):
    """Response format handling."""

    AUTO = "auto"
    JSON = "json"
    TEXT = "text"
    BINARY = "binary"


class HttpRequestInput(BaseModel):
    """Input schema for HttpRequestTool."""

    url: str = Field(
        ..., description="URL to request (must start with http:// or https://)"
    )
    method: str = Field(
        "GET", description="HTTP method: GET, POST, PUT, PATCH, DELETE, HEAD, OPTIONS"
    )
    headers: dict[str, str] | None = Field(
        None, description="Custom headers as key-value pairs"
    )
    query_params: dict[str, str] | None = Field(
        None, description="Query parameters as key-value pairs"
    )
    body: str | dict | None = Field(
        None, description="Request body (JSON object or string)"
    )
    content_type: str = Field("json", description="Content type: json, form, raw")
    auth_type: str = Field(
        "none", description="Auth type: none, basic, bearer, header, query"
    )
    auth_credentials: dict[str, str] | None = Field(
        None,
        description="Auth credentials. For basic: {username, password}. For bearer: {token}. "
        "For header: {name, value}. For query: {name, value}",
    )
    timeout: int = Field(30, description="Request timeout in seconds")
    follow_redirects: bool = Field(True, description="Follow HTTP redirects")
    max_redirects: int = Field(10, description="Maximum number of redirects to follow")
    response_format: str = Field(
        "auto", description="Response format: auto, json, text, binary"
    )
    verify_ssl: bool = Field(True, description="Verify SSL certificates")


# Headers that should be redacted in logs
REDACTED_HEADERS = frozenset(
    {
        "authorization",
        "x-api-key",
        "x-auth-token",
        "cookie",
        "proxy-authorization",
        "x-access-token",
        "api-key",
        "apikey",
    }
)

# Content types that indicate binary data
BINARY_CONTENT_TYPES = (
    "image/",
    "audio/",
    "video/",
    "application/octet-stream",
    "application/zip",
    "application/gzip",
    "application/pdf",
    "application/x-tar",
    "application/x-7z-compressed",
    "application/vnd.rar",
)

REDACTED = "**hidden**"


def _sanitize_for_logging(
    url: str,
    method: str,
    headers: dict[str, str] | None,
    body: Any,
) -> dict[str, Any]:
    """Sanitize request details for safe logging."""
    sanitized = {
        "url": url,
        "method": method,
    }

    if headers:
        sanitized["headers"] = {
            k: REDACTED if k.lower() in REDACTED_HEADERS else v
            for k, v in headers.items()
        }

    if body is not None:
        if isinstance(body, bytes) and len(body) > 1000:
            sanitized["body"] = f"<binary data: {len(body)} bytes>"
        elif isinstance(body, str) and len(body) > 1000:
            sanitized["body"] = body[:1000] + "... (truncated)"
        else:
            sanitized["body"] = body

    return sanitized


def _detect_response_format(content_type: str, body: bytes) -> str:
    """Auto-detect response format from content type and body."""
    content_type_lower = content_type.lower()

    if "application/json" in content_type_lower:
        return "json"

    for binary_type in BINARY_CONTENT_TYPES:
        if binary_type in content_type_lower:
            return "binary"

    # Try to detect JSON even if content-type doesn't say so
    if body:
        try:
            body_str = body.decode("utf-8").strip()
            if body_str.startswith(("{", "[")):
                json.loads(body_str)
                return "json"
        except (UnicodeDecodeError, json.JSONDecodeError):
            pass

    return "text"


class HttpRequestTool(AgentTool):
    """Full-featured HTTP request tool.

    Supports multiple HTTP methods, authentication types, custom headers,
    query parameters, request body formatting, and response handling.

    Features:
    - GET, POST, PUT, PATCH, DELETE, HEAD, OPTIONS methods
    - Basic, Bearer, Header, and Query parameter authentication
    - JSON, form-urlencoded, and raw body content types
    - Automatic response format detection
    - Timeout and redirect handling
    - SSL verification control
    - Request/response logging with credential redaction
    """

    USER_AGENT = "Marie-AI/1.0"
    MAX_RESPONSE_SIZE = 10_000_000  # 10MB

    def __init__(
        self,
        default_timeout: int = 30,
        default_headers: dict[str, str] | None = None,
    ):
        """Initialize HttpRequestTool.

        Args:
            default_timeout: Default request timeout in seconds
            default_headers: Default headers to include in all requests
        """
        self.default_timeout = default_timeout
        self.default_headers = default_headers or {}

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="http_request",
            description=(
                "Make HTTP requests to APIs and web services. "
                "Supports GET, POST, PUT, PATCH, DELETE methods with authentication, "
                "custom headers, query parameters, and various body formats."
            ),
            fn_schema=HttpRequestInput,
        )

    def call(
        self,
        url: str,
        method: str = "GET",
        headers: dict[str, str] | None = None,
        query_params: dict[str, str] | None = None,
        body: str | dict | None = None,
        content_type: str = "json",
        auth_type: str = "none",
        auth_credentials: dict[str, str] | None = None,
        timeout: int | None = None,
        follow_redirects: bool = True,
        max_redirects: int = 10,
        response_format: str = "auto",
        verify_ssl: bool = True,
        **kwargs: Any,
    ) -> ToolOutput:
        """Make an HTTP request.

        Args:
            url: URL to request
            method: HTTP method
            headers: Custom headers
            query_params: Query parameters
            body: Request body
            content_type: Body content type (json, form, raw)
            auth_type: Authentication type
            auth_credentials: Authentication credentials
            timeout: Request timeout in seconds
            follow_redirects: Whether to follow redirects
            max_redirects: Maximum redirects to follow
            response_format: Expected response format
            verify_ssl: Whether to verify SSL certificates

        Returns:
            ToolOutput with response data or error
        """
        timeout = timeout or self.default_timeout
        raw_input = {
            "url": url,
            "method": method,
            "content_type": content_type,
            "auth_type": auth_type,
            "timeout": timeout,
        }

        # Validate URL
        if not url.startswith(("http://", "https://")):
            result = {
                "error": "Invalid URL",
                "message": "URL must start with http:// or https://",
                "url": url,
            }
            return ToolOutput(
                content=json.dumps(result),
                tool_name=self.name,
                raw_input=raw_input,
                raw_output=result,
                is_error=True,
            )

        # Validate method
        method = method.upper()
        valid_methods = {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}
        if method not in valid_methods:
            result = {
                "error": "Invalid method",
                "message": f"Method must be one of: {', '.join(valid_methods)}",
                "method": method,
            }
            return ToolOutput(
                content=json.dumps(result),
                tool_name=self.name,
                raw_input=raw_input,
                raw_output=result,
                is_error=True,
            )

        try:
            # Build request
            request_headers = {**self.default_headers}
            request_headers["User-Agent"] = self.USER_AGENT

            # Add custom headers
            if headers:
                request_headers.update(headers)

            # Apply authentication
            auth_error = self._apply_auth(
                auth_type,
                auth_credentials,
                request_headers,
                query_params if query_params else {},
            )
            if auth_error:
                return ToolOutput(
                    content=json.dumps(auth_error),
                    tool_name=self.name,
                    raw_input=raw_input,
                    raw_output=auth_error,
                    is_error=True,
                )

            # Build URL with query params
            if query_params:
                url_parts = list(urllib.parse.urlparse(url))
                existing_query = urllib.parse.parse_qs(url_parts[4])
                existing_query.update({k: [v] for k, v in query_params.items()})
                url_parts[4] = urllib.parse.urlencode(existing_query, doseq=True)
                url = urllib.parse.urlunparse(url_parts)

            # Prepare body
            request_body = None
            if body is not None and method in {"POST", "PUT", "PATCH"}:
                request_body, body_content_type = self._prepare_body(body, content_type)
                if body_content_type:
                    request_headers["Content-Type"] = body_content_type

            # Create request
            req = urllib.request.Request(
                url,
                data=request_body,
                headers=request_headers,
                method=method,
            )

            # Configure SSL
            ssl_context = None
            if url.startswith("https://"):
                ssl_context = ssl.create_default_context()
                if not verify_ssl:
                    ssl_context.check_hostname = False
                    ssl_context.verify_mode = ssl.CERT_NONE

            # Make request
            response = self._make_request_with_redirects(
                req,
                timeout=timeout,
                ssl_context=ssl_context,
                follow_redirects=follow_redirects,
                max_redirects=max_redirects,
            )

            # Process response
            response_body = response.read(self.MAX_RESPONSE_SIZE)
            status_code = response.status
            response_headers = dict(response.headers)
            content_type_header = response_headers.get("Content-Type", "")

            # Determine response format
            if response_format == "auto":
                detected_format = _detect_response_format(
                    content_type_header, response_body
                )
            else:
                detected_format = response_format

            # Parse response based on format
            if detected_format == "json":
                try:
                    parsed_body = json.loads(response_body.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    parsed_body = response_body.decode("utf-8", errors="replace")
            elif detected_format == "binary":
                # Return base64 for binary data
                import base64

                parsed_body = {
                    "type": "binary",
                    "data": base64.b64encode(response_body).decode("ascii"),
                    "size": len(response_body),
                    "content_type": content_type_header,
                }
            else:
                parsed_body = response_body.decode("utf-8", errors="replace")

            result = {
                "status_code": status_code,
                "headers": response_headers,
                "body": parsed_body,
                "url": response.url,
                "format": detected_format,
            }

            # Log sanitized request for debugging
            result["_request"] = _sanitize_for_logging(
                url, method, request_headers, body
            )

            is_error = status_code >= 400

            return ToolOutput(
                content=json.dumps(result, default=str),
                tool_name=self.name,
                raw_input=raw_input,
                raw_output=result,
                is_error=is_error,
            )

        except urllib.error.HTTPError as e:
            error_body = ""
            try:
                error_body = e.read().decode("utf-8", errors="replace")
            except Exception:
                pass

            result = {
                "error": f"HTTP {e.code}",
                "status_code": e.code,
                "reason": e.reason,
                "body": error_body,
                "url": url,
                "_request": _sanitize_for_logging(url, method, headers, body),
            }
            return ToolOutput(
                content=json.dumps(result),
                tool_name=self.name,
                raw_input=raw_input,
                raw_output=result,
                is_error=True,
            )

        except urllib.error.URLError as e:
            result = {
                "error": "URL Error",
                "message": str(e.reason),
                "url": url,
            }
            return ToolOutput(
                content=json.dumps(result),
                tool_name=self.name,
                raw_input=raw_input,
                raw_output=result,
                is_error=True,
            )

        except TimeoutError:
            result = {
                "error": "Timeout",
                "message": f"Request timed out after {timeout} seconds",
                "url": url,
            }
            return ToolOutput(
                content=json.dumps(result),
                tool_name=self.name,
                raw_input=raw_input,
                raw_output=result,
                is_error=True,
            )

        except Exception as e:
            result = {
                "error": type(e).__name__,
                "message": str(e),
                "url": url,
            }
            return ToolOutput(
                content=json.dumps(result),
                tool_name=self.name,
                raw_input=raw_input,
                raw_output=result,
                is_error=True,
            )

    def _apply_auth(
        self,
        auth_type: str,
        credentials: dict[str, str] | None,
        headers: dict[str, str],
        query_params: dict[str, str],
    ) -> dict | None:
        """Apply authentication to the request.

        Returns error dict if authentication fails, None otherwise.
        """
        if auth_type == "none":
            return None

        if not credentials:
            return {
                "error": "Missing credentials",
                "message": f"auth_credentials required for auth_type={auth_type}",
            }

        if auth_type == "basic":
            username = credentials.get("username", "")
            password = credentials.get("password", "")
            if not username:
                return {"error": "Missing username for basic auth"}
            import base64

            auth_string = base64.b64encode(f"{username}:{password}".encode()).decode()
            headers["Authorization"] = f"Basic {auth_string}"

        elif auth_type == "bearer":
            token = credentials.get("token", "")
            if not token:
                return {"error": "Missing token for bearer auth"}
            headers["Authorization"] = f"Bearer {token}"

        elif auth_type == "header":
            name = credentials.get("name", "")
            value = credentials.get("value", "")
            if not name:
                return {"error": "Missing header name for header auth"}
            headers[name] = value

        elif auth_type == "query":
            name = credentials.get("name", "")
            value = credentials.get("value", "")
            if not name:
                return {"error": "Missing query param name for query auth"}
            query_params[name] = value

        else:
            return {
                "error": "Invalid auth_type",
                "message": f"Supported: none, basic, bearer, header, query",
            }

        return None

    def _prepare_body(
        self,
        body: str | dict | None,
        content_type: str,
    ) -> tuple[bytes | None, str | None]:
        """Prepare request body and determine content type header.

        Returns (body_bytes, content_type_header).
        """
        if body is None:
            return None, None

        if content_type == "json":
            if isinstance(body, dict):
                return json.dumps(body).encode("utf-8"), "application/json"
            elif isinstance(body, str):
                # Validate it's valid JSON
                try:
                    json.loads(body)
                    return body.encode("utf-8"), "application/json"
                except json.JSONDecodeError:
                    # Treat as raw JSON string
                    return json.dumps(body).encode("utf-8"), "application/json"
            else:
                return json.dumps(body).encode("utf-8"), "application/json"

        elif content_type == "form":
            if isinstance(body, dict):
                return (
                    urllib.parse.urlencode(body).encode("utf-8"),
                    "application/x-www-form-urlencoded",
                )
            elif isinstance(body, str):
                return body.encode("utf-8"), "application/x-www-form-urlencoded"

        else:  # raw
            if isinstance(body, str):
                return body.encode("utf-8"), None
            elif isinstance(body, bytes):
                return body, None
            else:
                return str(body).encode("utf-8"), None

        return None, None

    def _make_request_with_redirects(
        self,
        req: urllib.request.Request,
        timeout: int,
        ssl_context: ssl.SSLContext | None,
        follow_redirects: bool,
        max_redirects: int,
    ) -> HTTPResponse:
        """Make request with manual redirect handling."""

        class NoRedirectHandler(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                return None

        if not follow_redirects:
            opener = urllib.request.build_opener(NoRedirectHandler)
            return opener.open(req, timeout=timeout)

        redirects = 0
        current_req = req

        while redirects < max_redirects:
            opener = urllib.request.build_opener(NoRedirectHandler)

            try:
                if ssl_context and current_req.full_url.startswith("https://"):
                    response = urllib.request.urlopen(
                        current_req, timeout=timeout, context=ssl_context
                    )
                else:
                    response = opener.open(current_req, timeout=timeout)

                # Check for redirect status codes
                if response.status in (301, 302, 303, 307, 308):
                    location = response.headers.get("Location")
                    if location:
                        # Handle relative URLs
                        location = urllib.parse.urljoin(current_req.full_url, location)

                        # For 303, always use GET
                        new_method = (
                            "GET" if response.status == 303 else current_req.method
                        )

                        current_req = urllib.request.Request(
                            location,
                            data=None if response.status == 303 else current_req.data,
                            headers=dict(current_req.headers),
                            method=new_method,
                        )
                        redirects += 1
                        continue

                return response

            except urllib.error.HTTPError as e:
                if e.code in (301, 302, 303, 307, 308):
                    location = e.headers.get("Location")
                    if location and follow_redirects:
                        location = urllib.parse.urljoin(current_req.full_url, location)
                        new_method = "GET" if e.code == 303 else current_req.method
                        current_req = urllib.request.Request(
                            location,
                            data=None if e.code == 303 else current_req.data,
                            headers=dict(current_req.headers),
                            method=new_method,
                        )
                        redirects += 1
                        continue
                raise

        raise urllib.error.URLError(f"Too many redirects (max {max_redirects})")

    async def acall(
        self,
        url: str,
        method: str = "GET",
        headers: dict[str, str] | None = None,
        query_params: dict[str, str] | None = None,
        body: str | dict | None = None,
        content_type: str = "json",
        auth_type: str = "none",
        auth_credentials: dict[str, str] | None = None,
        timeout: int | None = None,
        follow_redirects: bool = True,
        max_redirects: int = 10,
        response_format: str = "auto",
        verify_ssl: bool = True,
        **kwargs: Any,
    ) -> ToolOutput:
        """Async version of call - runs sync call in thread pool."""
        return await asyncio.to_thread(
            self.call,
            url=url,
            method=method,
            headers=headers,
            query_params=query_params,
            body=body,
            content_type=content_type,
            auth_type=auth_type,
            auth_credentials=auth_credentials,
            timeout=timeout,
            follow_redirects=follow_redirects,
            max_redirects=max_redirects,
            response_format=response_format,
            verify_ssl=verify_ssl,
            **kwargs,
        )
