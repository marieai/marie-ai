"""Web fetch tool for agent framework."""

from __future__ import annotations

import json
import urllib.request
from html.parser import HTMLParser
from typing import Any

from pydantic import BaseModel, Field

from marie.agent.tools.base import AgentTool, ToolMetadata, ToolOutput


class WebFetchInput(BaseModel):
    """Input schema for WebFetchTool."""

    url: str = Field(..., description="URL to fetch")
    timeout: int = Field(10, description="Timeout in seconds")
    max_size: int = Field(100_000, description="Max content size in bytes")


class _TextExtractor(HTMLParser):
    """Extract text content from HTML."""

    def __init__(self):
        super().__init__()
        self.text = []
        self.skip_tags = {"script", "style", "head", "meta", "noscript"}
        self.current_tag = None

    def handle_starttag(self, tag, attrs):
        self.current_tag = tag

    def handle_endtag(self, tag):
        if tag == self.current_tag:
            self.current_tag = None

    def handle_data(self, data):
        if self.current_tag not in self.skip_tags:
            text = data.strip()
            if text:
                self.text.append(text)

    def get_text(self):
        return " ".join(self.text)


class WebFetchTool(AgentTool):
    """Fetch URL content and extract text.

    Supports HTML text extraction with configurable timeout and size limits.
    """

    USER_AGENT = "Marie-AI/1.0"
    MAX_TEXT_LENGTH = 10_000

    def __init__(self, timeout: int = 10, max_size: int = 100_000):
        """Initialize WebFetchTool.

        Args:
            timeout: Default request timeout in seconds
            max_size: Default maximum content size in bytes
        """
        self.default_timeout = timeout
        self.default_max_size = max_size

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="web_fetch",
            description="Fetch content from a URL and extract text. Returns text content of web pages.",
            fn_schema=WebFetchInput,
        )

    def call(
        self,
        url: str,
        timeout: int | None = None,
        max_size: int | None = None,
        **kwargs: Any,
    ) -> ToolOutput:
        """Fetch content from a URL.

        Args:
            url: URL to fetch
            timeout: Request timeout in seconds
            max_size: Maximum content size in bytes

        Returns:
            ToolOutput with page content or error
        """
        timeout = timeout or self.default_timeout
        max_size = max_size or self.default_max_size
        raw_input = {"url": url, "timeout": timeout, "max_size": max_size}

        if not url:
            result = {"error": "URL is required", "url": url}
            return ToolOutput(
                content=json.dumps(result),
                tool_name=self.name,
                raw_input=raw_input,
                raw_output=result,
                is_error=True,
            )

        try:
            req = urllib.request.Request(url, headers={"User-Agent": self.USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as response:
                content = response.read(max_size).decode("utf-8", errors="ignore")
                content_type = response.headers.get("Content-Type", "")

            # Extract text from HTML
            if "html" in content_type.lower():
                parser = _TextExtractor()
                parser.feed(content)
                text = parser.get_text()[: self.MAX_TEXT_LENGTH]
            else:
                text = content[: self.MAX_TEXT_LENGTH]

            result = {
                "url": url,
                "content": text,
                "content_type": content_type,
                "length": len(text),
            }
            return ToolOutput(
                content=json.dumps(result),
                tool_name=self.name,
                raw_input=raw_input,
                raw_output=result,
                is_error=False,
            )

        except urllib.error.HTTPError as e:
            result = {
                "error": f"HTTP error: {e.code}",
                "url": url,
                "reason": str(e.reason),
            }
            return ToolOutput(
                content=json.dumps(result),
                tool_name=self.name,
                raw_input=raw_input,
                raw_output=result,
                is_error=True,
            )
        except urllib.error.URLError as e:
            result = {"error": f"URL error: {e.reason}", "url": url}
            return ToolOutput(
                content=json.dumps(result),
                tool_name=self.name,
                raw_input=raw_input,
                raw_output=result,
                is_error=True,
            )
        except TimeoutError:
            result = {"error": "Request timed out", "url": url, "timeout": timeout}
            return ToolOutput(
                content=json.dumps(result),
                tool_name=self.name,
                raw_input=raw_input,
                raw_output=result,
                is_error=True,
            )
        except Exception as e:
            result = {"error": str(e), "url": url}
            return ToolOutput(
                content=json.dumps(result),
                tool_name=self.name,
                raw_input=raw_input,
                raw_output=result,
                is_error=True,
            )
