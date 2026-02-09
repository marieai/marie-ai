"""Tests for HttpRequestTool."""

from __future__ import annotations

import base64
import json
from http.client import HTTPMessage
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from marie.agent.tools.utilities import HttpRequestTool
from marie.agent.tools.utilities.http_request import (
    REDACTED,
    REDACTED_HEADERS,
    _detect_response_format,
    _sanitize_for_logging,
)


class TestHttpRequestTool:
    """Tests for HttpRequestTool."""

    def test_metadata(self):
        """Test tool metadata is correct."""
        tool = HttpRequestTool()
        assert tool.name == "http_request"
        assert "HTTP" in tool.description
        assert tool.metadata.fn_schema is not None

    def test_invalid_url_no_protocol(self):
        """Test error when URL has no protocol."""
        tool = HttpRequestTool()
        result = tool.call(url="example.com")

        assert result.is_error is True
        data = json.loads(result.content)
        assert "Invalid URL" in data["error"]

    def test_invalid_url_wrong_protocol(self):
        """Test error when URL has wrong protocol."""
        tool = HttpRequestTool()
        result = tool.call(url="ftp://example.com")

        assert result.is_error is True
        data = json.loads(result.content)
        assert "Invalid URL" in data["error"]

    def test_invalid_method(self):
        """Test error when method is invalid."""
        tool = HttpRequestTool()
        result = tool.call(url="https://example.com", method="INVALID")

        assert result.is_error is True
        data = json.loads(result.content)
        assert "Invalid method" in data["error"]

    @patch("urllib.request.urlopen")
    def test_successful_get_json(self, mock_urlopen):
        """Test successful GET request with JSON response."""
        mock_response = MagicMock()
        mock_response.read.return_value = b'{"key": "value"}'
        mock_response.status = 200
        mock_response.url = "https://api.example.com/data"
        mock_response.headers = HTTPMessage()
        mock_response.headers["Content-Type"] = "application/json"
        mock_urlopen.return_value = mock_response

        tool = HttpRequestTool()
        result = tool.call(url="https://api.example.com/data")

        assert result.is_error is False
        data = json.loads(result.content)
        assert data["status_code"] == 200
        assert data["body"] == {"key": "value"}
        assert data["format"] == "json"

    @patch("urllib.request.urlopen")
    def test_successful_post_json(self, mock_urlopen):
        """Test successful POST request with JSON body."""
        mock_response = MagicMock()
        mock_response.read.return_value = b'{"id": 123}'
        mock_response.status = 201
        mock_response.url = "https://api.example.com/create"
        mock_response.headers = HTTPMessage()
        mock_response.headers["Content-Type"] = "application/json"
        mock_urlopen.return_value = mock_response

        tool = HttpRequestTool()
        result = tool.call(
            url="https://api.example.com/create",
            method="POST",
            body={"name": "test"},
        )

        assert result.is_error is False
        data = json.loads(result.content)
        assert data["status_code"] == 201
        assert data["body"]["id"] == 123

    @patch("urllib.request.urlopen")
    def test_text_response(self, mock_urlopen):
        """Test response parsed as text."""
        mock_response = MagicMock()
        mock_response.read.return_value = b"Plain text response"
        mock_response.status = 200
        mock_response.url = "https://example.com/text"
        mock_response.headers = HTTPMessage()
        mock_response.headers["Content-Type"] = "text/plain"
        mock_urlopen.return_value = mock_response

        tool = HttpRequestTool()
        result = tool.call(url="https://example.com/text")

        data = json.loads(result.content)
        assert data["format"] == "text"
        assert data["body"] == "Plain text response"

    @patch("urllib.request.urlopen")
    def test_binary_response(self, mock_urlopen):
        """Test binary response returns base64."""
        binary_data = b"\x89PNG\r\n\x1a\n\x00\x00"
        mock_response = MagicMock()
        mock_response.read.return_value = binary_data
        mock_response.status = 200
        mock_response.url = "https://example.com/image.png"
        mock_response.headers = HTTPMessage()
        mock_response.headers["Content-Type"] = "image/png"
        mock_urlopen.return_value = mock_response

        tool = HttpRequestTool()
        result = tool.call(url="https://example.com/image.png")

        data = json.loads(result.content)
        assert data["format"] == "binary"
        assert data["body"]["type"] == "binary"
        assert data["body"]["data"] == base64.b64encode(binary_data).decode("ascii")
        assert data["body"]["size"] == len(binary_data)

    @patch("urllib.request.urlopen")
    def test_custom_headers(self, mock_urlopen):
        """Test custom headers are sent."""
        mock_response = MagicMock()
        mock_response.read.return_value = b"{}"
        mock_response.status = 200
        mock_response.url = "https://api.example.com"
        mock_response.headers = HTTPMessage()
        mock_response.headers["Content-Type"] = "application/json"
        mock_urlopen.return_value = mock_response

        tool = HttpRequestTool()
        result = tool.call(
            url="https://api.example.com",
            headers={"X-Custom-Header": "custom-value"},
        )

        # Verify the request was made (can check mock call args)
        assert result.is_error is False

    @patch("urllib.request.urlopen")
    def test_bearer_auth(self, mock_urlopen):
        """Test bearer token authentication."""
        mock_response = MagicMock()
        mock_response.read.return_value = b'{"authenticated": true}'
        mock_response.status = 200
        mock_response.url = "https://api.example.com/protected"
        mock_response.headers = HTTPMessage()
        mock_response.headers["Content-Type"] = "application/json"
        mock_urlopen.return_value = mock_response

        tool = HttpRequestTool()
        result = tool.call(
            url="https://api.example.com/protected",
            auth_type="bearer",
            auth_credentials={"token": "secret-token"},
        )

        assert result.is_error is False
        # Check that Authorization header was set (via _request in result)
        data = json.loads(result.content)
        assert data["_request"]["headers"]["Authorization"] == REDACTED

    @patch("urllib.request.urlopen")
    def test_basic_auth(self, mock_urlopen):
        """Test basic authentication."""
        mock_response = MagicMock()
        mock_response.read.return_value = b"{}"
        mock_response.status = 200
        mock_response.url = "https://api.example.com"
        mock_response.headers = HTTPMessage()
        mock_response.headers["Content-Type"] = "application/json"
        mock_urlopen.return_value = mock_response

        tool = HttpRequestTool()
        result = tool.call(
            url="https://api.example.com",
            auth_type="basic",
            auth_credentials={"username": "user", "password": "pass"},
        )

        assert result.is_error is False
        data = json.loads(result.content)
        # Authorization header should be redacted
        assert data["_request"]["headers"]["Authorization"] == REDACTED

    def test_missing_credentials_error(self):
        """Test error when auth requires credentials but none provided."""
        tool = HttpRequestTool()
        result = tool.call(
            url="https://api.example.com",
            auth_type="bearer",
            auth_credentials=None,
        )

        assert result.is_error is True
        data = json.loads(result.content)
        assert "Missing credentials" in data["error"]

    def test_missing_token_error(self):
        """Test error when bearer auth missing token."""
        tool = HttpRequestTool()
        result = tool.call(
            url="https://api.example.com",
            auth_type="bearer",
            auth_credentials={"wrong_key": "value"},
        )

        assert result.is_error is True
        data = json.loads(result.content)
        assert "Missing token" in data["error"]

    @patch("urllib.request.urlopen")
    def test_query_params(self, mock_urlopen):
        """Test query parameters are added to URL."""
        mock_response = MagicMock()
        mock_response.read.return_value = b"{}"
        mock_response.status = 200
        mock_response.url = "https://api.example.com/search?q=test&limit=10"
        mock_response.headers = HTTPMessage()
        mock_response.headers["Content-Type"] = "application/json"
        mock_urlopen.return_value = mock_response

        tool = HttpRequestTool()
        result = tool.call(
            url="https://api.example.com/search",
            query_params={"q": "test", "limit": "10"},
        )

        assert result.is_error is False

    @patch("urllib.request.urlopen")
    def test_form_body(self, mock_urlopen):
        """Test form-urlencoded body."""
        mock_response = MagicMock()
        mock_response.read.return_value = b"{}"
        mock_response.status = 200
        mock_response.url = "https://api.example.com/login"
        mock_response.headers = HTTPMessage()
        mock_response.headers["Content-Type"] = "application/json"
        mock_urlopen.return_value = mock_response

        tool = HttpRequestTool()
        result = tool.call(
            url="https://api.example.com/login",
            method="POST",
            body={"username": "user", "password": "pass"},
            content_type="form",
        )

        assert result.is_error is False

    @patch("urllib.request.urlopen")
    def test_http_error_handling(self, mock_urlopen):
        """Test handling of HTTP errors."""
        import urllib.error

        mock_urlopen.side_effect = urllib.error.HTTPError(
            "https://api.example.com", 404, "Not Found", {}, BytesIO(b"Not found")
        )

        tool = HttpRequestTool()
        result = tool.call(url="https://api.example.com/notfound")

        assert result.is_error is True
        data = json.loads(result.content)
        assert data["status_code"] == 404
        assert "404" in data["error"]

    @patch("urllib.request.urlopen")
    def test_url_error_handling(self, mock_urlopen):
        """Test handling of URL errors (network issues)."""
        import urllib.error

        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

        tool = HttpRequestTool()
        result = tool.call(url="https://api.example.com")

        assert result.is_error is True
        data = json.loads(result.content)
        assert "URL Error" in data["error"]

    @patch("urllib.request.urlopen")
    def test_timeout_error_handling(self, mock_urlopen):
        """Test handling of timeout errors."""
        mock_urlopen.side_effect = TimeoutError("Timed out")

        tool = HttpRequestTool()
        result = tool.call(url="https://api.example.com", timeout=5)

        assert result.is_error is True
        data = json.loads(result.content)
        assert "Timeout" in data["error"]

    def test_default_timeout(self):
        """Test custom default timeout."""
        tool = HttpRequestTool(default_timeout=60)
        assert tool.default_timeout == 60

    def test_default_headers(self):
        """Test custom default headers."""
        tool = HttpRequestTool(default_headers={"X-Api-Version": "v1"})
        assert tool.default_headers["X-Api-Version"] == "v1"

    @patch("urllib.request.urlopen")
    def test_4xx_error_marked_as_error(self, mock_urlopen):
        """Test that 4xx responses are marked as errors."""
        mock_response = MagicMock()
        mock_response.read.return_value = b'{"error": "Bad request"}'
        mock_response.status = 400
        mock_response.url = "https://api.example.com"
        mock_response.headers = HTTPMessage()
        mock_response.headers["Content-Type"] = "application/json"
        mock_urlopen.return_value = mock_response

        tool = HttpRequestTool()
        result = tool.call(url="https://api.example.com")

        assert result.is_error is True
        data = json.loads(result.content)
        assert data["status_code"] == 400

    @patch("urllib.request.urlopen")
    def test_5xx_error_marked_as_error(self, mock_urlopen):
        """Test that 5xx responses are marked as errors."""
        mock_response = MagicMock()
        mock_response.read.return_value = b'{"error": "Internal error"}'
        mock_response.status = 500
        mock_response.url = "https://api.example.com"
        mock_response.headers = HTTPMessage()
        mock_response.headers["Content-Type"] = "application/json"
        mock_urlopen.return_value = mock_response

        tool = HttpRequestTool()
        result = tool.call(url="https://api.example.com")

        assert result.is_error is True
        data = json.loads(result.content)
        assert data["status_code"] == 500


class TestSanitizeForLogging:
    """Tests for _sanitize_for_logging helper."""

    def test_redacts_authorization_header(self):
        """Test that Authorization header is redacted."""
        result = _sanitize_for_logging(
            url="https://api.example.com",
            method="GET",
            headers={"Authorization": "Bearer secret-token"},
            body=None,
        )

        assert result["headers"]["Authorization"] == REDACTED

    def test_redacts_api_key_headers(self):
        """Test that API key headers are redacted."""
        result = _sanitize_for_logging(
            url="https://api.example.com",
            method="GET",
            headers={
                "X-Api-Key": "secret-key",
                "api-key": "another-secret",
                "X-Custom": "not-secret",
            },
            body=None,
        )

        assert result["headers"]["X-Api-Key"] == REDACTED
        assert result["headers"]["api-key"] == REDACTED
        assert result["headers"]["X-Custom"] == "not-secret"

    def test_truncates_large_body(self):
        """Test that large bodies are truncated."""
        large_body = "x" * 2000
        result = _sanitize_for_logging(
            url="https://api.example.com",
            method="POST",
            headers=None,
            body=large_body,
        )

        assert "truncated" in result["body"]
        assert len(result["body"]) < len(large_body)

    def test_marks_binary_body(self):
        """Test that binary bodies are marked."""
        binary_body = b"x" * 2000
        result = _sanitize_for_logging(
            url="https://api.example.com",
            method="POST",
            headers=None,
            body=binary_body,
        )

        assert "binary data" in result["body"]


class TestDetectResponseFormat:
    """Tests for _detect_response_format helper."""

    def test_detects_json_content_type(self):
        """Test JSON detection from content type."""
        assert _detect_response_format("application/json", b"{}") == "json"
        assert _detect_response_format("application/json; charset=utf-8", b"{}") == "json"

    def test_detects_binary_content_types(self):
        """Test binary detection from content type."""
        assert _detect_response_format("image/png", b"\x89PNG") == "binary"
        assert _detect_response_format("application/pdf", b"%PDF") == "binary"
        assert _detect_response_format("application/octet-stream", b"\x00\x01") == "binary"

    def test_detects_json_from_body(self):
        """Test JSON detection from body content."""
        assert _detect_response_format("text/plain", b'{"key": "value"}') == "json"
        assert _detect_response_format("text/plain", b'[1, 2, 3]') == "json"

    def test_defaults_to_text(self):
        """Test default to text for unknown content."""
        assert _detect_response_format("text/html", b"<html></html>") == "text"
        assert _detect_response_format("text/plain", b"plain text") == "text"
