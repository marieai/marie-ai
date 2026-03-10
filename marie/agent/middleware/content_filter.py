"""Content filter middleware for blocking banned words/patterns.

Provides content moderation by inspecting event data for forbidden
content and optionally blocking the operation.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set

from marie.agent.middleware.protocol import BaseMiddleware
from marie.logging_core.logger import MarieLogger

if TYPE_CHECKING:
    from marie.agent.emitter import Emitter

logger = MarieLogger("marie.agent.middleware.content_filter")


class ContentFilterError(Exception):
    """Raised when content is blocked by the filter."""

    def __init__(self, message: str, matched: str):
        self.matched = matched
        super().__init__(message)


class ContentFilterMiddleware(BaseMiddleware):
    """Middleware that filters content for banned words/patterns.

    Inspects event data for forbidden content and can either:
    - Log a warning (default)
    - Raise an exception to block the operation (strict mode)
    """

    def __init__(
        self,
        banned_words: Optional[List[str]] = None,
        banned_patterns: Optional[List[str]] = None,
        strict: bool = False,
        on_match: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        """Initialize content filter middleware.

        Args:
            banned_words: List of banned words (case-insensitive)
            banned_patterns: List of regex patterns to block
            strict: If True, raise exception on match; otherwise log warning
            on_match: Optional callback when match found (content, matched)
        """
        super().__init__(name="ContentFilterMiddleware", priority=100)
        self._banned_words: Set[str] = {w.lower() for w in (banned_words or [])}
        self._banned_patterns: List[re.Pattern] = [
            re.compile(p, re.IGNORECASE) for p in (banned_patterns or [])
        ]
        self._strict = strict
        self._on_match = on_match

    def bind(self, emitter: "Emitter") -> None:
        """Bind content filtering to emitter events."""
        # Check agent messages
        self._listener_ids.append(
            emitter.on(
                "agent.start",
                self._check_event,
                priority=self.priority,
                is_blocking=self._strict,
            )
        )

        # Check tool inputs
        self._listener_ids.append(
            emitter.on(
                "tool.start",
                self._check_event,
                priority=self.priority,
                is_blocking=self._strict,
            )
        )

        # Check LLM responses via new_token
        self._listener_ids.append(
            emitter.on(
                "llm.new_token",
                self._check_token,
                priority=self.priority,
                is_blocking=self._strict,
            )
        )

    def _check_content(self, content: str) -> Optional[str]:
        """Check content for banned words/patterns.

        Returns:
            The matched word/pattern if found, None otherwise
        """
        content_lower = content.lower()

        # Check banned words
        for word in self._banned_words:
            if word in content_lower:
                return word

        # Check banned patterns
        for pattern in self._banned_patterns:
            match = pattern.search(content)
            if match:
                return match.group()

        return None

    def _handle_match(self, content: str, matched: str) -> None:
        """Handle a content filter match."""
        if self._on_match:
            self._on_match(content, matched)

        if self._strict:
            raise ContentFilterError(
                f"Content blocked: matched forbidden pattern '{matched}'",
                matched=matched,
            )
        else:
            logger.warning(f"Content filter match: '{matched}' in content")

    def _check_event(self, data: Dict[str, Any]) -> None:
        """Check event data for banned content."""
        # Check string values in the event data
        for key, value in data.items():
            if isinstance(value, str):
                matched = self._check_content(value)
                if matched:
                    self._handle_match(value, matched)
            elif isinstance(value, dict):
                # Check nested dict values
                for nested_key, nested_value in value.items():
                    if isinstance(nested_value, str):
                        matched = self._check_content(nested_value)
                        if matched:
                            self._handle_match(nested_value, matched)

    def _check_token(self, data: Dict[str, Any]) -> None:
        """Check token content for banned content."""
        token = data.get("token", "")
        if token:
            matched = self._check_content(token)
            if matched:
                self._handle_match(token, matched)
