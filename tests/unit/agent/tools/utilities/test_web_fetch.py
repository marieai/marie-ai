"""Tests for WebFetchTool."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from marie.agent.tools.utilities import WebFetchTool


class TestWebFetchTool:
    """Tests for WebFetchTool."""

    def test_metadata(self):
        """Test tool metadata is correct."""
        tool = WebFetchTool()
        assert tool.name == "web_fetch"
        assert "URL" in tool.description or "Fetch" in tool.description
        assert tool.metadata.fn_schema is not None

    def test_empty_url_error(self):
        """Test error when URL is empty."""
        tool = WebFetchTool()
        result = tool.call(url="")

        assert result.is_error is True
        data = json.loads(result.content)
        assert "required" in data["error"].lower()

    def test_custom_timeout_default(self):
        """Test custom timeout in constructor."""
        tool = WebFetchTool(timeout=30, max_size=50000)
        assert tool.default_timeout == 30
        assert tool.default_max_size == 50000

    @patch("urllib.request.urlopen")
    def test_successful_fetch(self, mock_urlopen):
        """Test successful URL fetch."""
        mock_response = MagicMock()
        mock_response.read.return_value = b"<html><body>Hello World</body></html>"
        mock_response.headers = {"Content-Type": "text/html"}
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        tool = WebFetchTool()
        result = tool.call(url="https://example.com")

        assert result.is_error is False
        data = json.loads(result.content)
        assert "content" in data
        assert "Hello World" in data["content"]

    @patch("urllib.request.urlopen")
    def test_html_text_extraction(self, mock_urlopen):
        """Test that HTML tags are stripped and text extracted."""
        html_content = b"""
        <html>
        <head><title>Test</title><script>var x = 1;</script></head>
        <body>
        <h1>Header</h1>
        <p>Paragraph text</p>
        <style>.class { color: red; }</style>
        </body>
        </html>
        """
        mock_response = MagicMock()
        mock_response.read.return_value = html_content
        mock_response.headers = {"Content-Type": "text/html; charset=utf-8"}
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        tool = WebFetchTool()
        result = tool.call(url="https://example.com")

        data = json.loads(result.content)
        # Should contain text content
        assert "Header" in data["content"]
        assert "Paragraph text" in data["content"]
        # Should not contain script/style content
        assert "var x = 1" not in data["content"]
        assert ".class" not in data["content"]

    @patch("urllib.request.urlopen")
    def test_plain_text_passthrough(self, mock_urlopen):
        """Test that plain text content is passed through."""
        mock_response = MagicMock()
        mock_response.read.return_value = b"Plain text content"
        mock_response.headers = {"Content-Type": "text/plain"}
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        tool = WebFetchTool()
        result = tool.call(url="https://example.com/text.txt")

        data = json.loads(result.content)
        assert data["content"] == "Plain text content"

    @patch("urllib.request.urlopen")
    def test_http_error_handling(self, mock_urlopen):
        """Test handling of HTTP errors."""
        import urllib.error

        mock_urlopen.side_effect = urllib.error.HTTPError(
            "https://example.com", 404, "Not Found", {}, None
        )

        tool = WebFetchTool()
        result = tool.call(url="https://example.com/notfound")

        assert result.is_error is True
        data = json.loads(result.content)
        assert "404" in data["error"]

    @patch("urllib.request.urlopen")
    def test_url_error_handling(self, mock_urlopen):
        """Test handling of URL errors (network issues)."""
        import urllib.error

        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

        tool = WebFetchTool()
        result = tool.call(url="https://example.com")

        assert result.is_error is True
        data = json.loads(result.content)
        assert "error" in data

    @patch("urllib.request.urlopen")
    def test_timeout_error_handling(self, mock_urlopen):
        """Test handling of timeout errors."""
        mock_urlopen.side_effect = TimeoutError("Request timed out")

        tool = WebFetchTool()
        result = tool.call(url="https://example.com", timeout=1)

        assert result.is_error is True
        data = json.loads(result.content)
        assert "timed out" in data["error"].lower()

    def test_tool_output_structure(self):
        """Test that output has correct structure."""
        tool = WebFetchTool()
        result = tool.call(url="")  # Will error but structure should be correct

        assert result.tool_name == "web_fetch"
        assert "url" in result.raw_input

    @patch("urllib.request.urlopen")
    def test_content_length_in_result(self, mock_urlopen):
        """Test that content length is included in result."""
        mock_response = MagicMock()
        mock_response.read.return_value = b"Test content"
        mock_response.headers = {"Content-Type": "text/plain"}
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        tool = WebFetchTool()
        result = tool.call(url="https://example.com")

        data = json.loads(result.content)
        assert "length" in data
        assert data["length"] == len("Test content")
