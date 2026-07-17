"""Content filter guardrail.

Filters content based on banned words, phrases, and regex patterns.
"""

from __future__ import annotations

import re
from typing import Any, List, Set

from pydantic import Field

from marie.agent.guardrails.base import Guardrail, GuardrailConfig
from marie.agent.guardrails.registry import register_guardrail
from marie.agent.guardrails.result import GuardrailAction, GuardrailResult


class ContentFilterConfig(GuardrailConfig):
    """Configuration for content filtering.

    Attributes:
        banned_words: List of banned words (case-insensitive)
        banned_patterns: List of regex patterns to block
        block_message: Message to return when content is blocked
        case_sensitive: Whether word matching is case-sensitive
    """

    banned_words: List[str] = Field(
        default_factory=list,
        description="Words to block (case-insensitive by default)",
    )
    banned_patterns: List[str] = Field(
        default_factory=list,
        description="Regex patterns to block",
    )
    block_message: str = Field(
        default="Content contains prohibited material",
        description="Message returned when content is blocked",
    )
    case_sensitive: bool = Field(
        default=False,
        description="Whether word matching is case-sensitive",
    )


@register_guardrail("content_filter", "before")
class ContentFilterGuardrail(Guardrail):
    """Filter content based on banned words and patterns.

    Scans input for prohibited words and regex patterns. Use this
    for basic content moderation of user inputs.

    Example:
        ```yaml
        guardrails:
          before:
            - type: content_filter
              config:
                banned_words:
                  - prohibited_word
                  - another_bad_word
                banned_patterns:
                  - "(?i)spam.*link"
                block_message: "Your message contains prohibited content"
        ```
    """

    name = "content_filter"
    phase = "before"
    config_class = ContentFilterConfig

    def __init__(self, config: ContentFilterConfig = None):
        super().__init__(config or ContentFilterConfig())
        self._compiled_patterns: List[re.Pattern] = []
        self._banned_words_set: Set[str] = set()
        self._initialize_filters()

    def _initialize_filters(self) -> None:
        """Initialize compiled patterns and word sets."""
        config = self.config
        if not isinstance(config, ContentFilterConfig):
            return

        # Compile regex patterns
        for pattern in config.banned_patterns:
            try:
                self._compiled_patterns.append(re.compile(pattern))
            except re.error:
                # Skip invalid patterns
                pass

        # Build word set
        if config.case_sensitive:
            self._banned_words_set = set(config.banned_words)
        else:
            self._banned_words_set = {w.lower() for w in config.banned_words}

    def _check_words(self, content: str) -> List[str]:
        """Check content for banned words.

        Returns:
            List of matched word types (not the actual words for privacy)
        """
        if not self._banned_words_set:
            return []

        config = self.config
        if not isinstance(config, ContentFilterConfig):
            return []

        matches = []
        # Tokenize content into words
        words = re.findall(r'\b\w+\b', content)

        for word in words:
            check_word = word if config.case_sensitive else word.lower()
            if check_word in self._banned_words_set:
                matches.append("banned_word")

        return matches

    def _check_patterns(self, content: str) -> List[str]:
        """Check content against regex patterns.

        Returns:
            List of matched pattern indices (not the matched text)
        """
        matches = []
        for i, pattern in enumerate(self._compiled_patterns):
            if pattern.search(content):
                matches.append(f"pattern_{i}")
        return matches

    async def evaluate(self, content: Any, context: dict) -> GuardrailResult:
        """Evaluate content against filters.

        Args:
            content: Input text to check
            context: Execution context

        Returns:
            GuardrailResult with BLOCK if content matches filters
        """
        if not isinstance(content, str):
            return GuardrailResult(action=GuardrailAction.ALLOW)

        config = self.config
        if not isinstance(config, ContentFilterConfig):
            config = ContentFilterConfig()

        # Check banned words
        word_matches = self._check_words(content)

        # Check regex patterns
        pattern_matches = self._check_patterns(content)

        # Combine all matches
        all_matches = word_matches + pattern_matches

        if not all_matches:
            return GuardrailResult(
                action=GuardrailAction.ALLOW,
                score=0.0,
            )

        return GuardrailResult(
            action=GuardrailAction.BLOCK,
            score=1.0,
            message=config.block_message,
            metadata={
                "match_count": len(all_matches),
                "word_matches": len(word_matches),
                "pattern_matches": len(pattern_matches),
            },
        )
