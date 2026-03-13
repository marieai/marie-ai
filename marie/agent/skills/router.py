"""Skill router for request-to-skill matching.

This module provides routing logic to match user messages to
appropriate skills, supporting both explicit invocation (/skill-name)
and automatic matching.

Enhanced with BM25 search for faster and more accurate skill matching.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from marie.agent.skills.models import Skill, SkillContext
from marie.agent.skills.registry import (
    SKILL_REGISTRY,
    SkillNotFoundError,
    SkillRegistry,
)
from marie.agent.skills.search import SkillSearchIndex
from marie.logging_core.logger import MarieLogger

logger = MarieLogger("marie.agent.skills.router")

# Pattern for explicit skill invocation: /skill-name [args]
SLASH_COMMAND_PATTERN = re.compile(
    r"^/([a-z0-9][a-z0-9-]*[a-z0-9]|[a-z0-9])(?:\s+(.*))?$"
)


class SkillRouter:
    """Routes requests to appropriate skills.

    The router supports two modes:
    1. Explicit invocation via /skill-name
    2. Automatic matching based on message content

    Example:
        ```python
        router = SkillRouter(registry=SKILL_REGISTRY)

        # Explicit invocation
        context = await router.route("/document-extraction invoice.pdf")
        assert context.skill.name == "document-extraction"

        # Automatic matching
        context = await router.route("Extract text from this PDF")
        if context.skill:
            print(f"Matched: {context.skill.name}")
        ```
    """

    def __init__(
        self,
        registry: Optional[SkillRegistry] = None,
        auto_match_threshold: float = 0.3,
        use_bm25: bool = True,
    ):
        """Initialize router.

        Args:
            registry: Skill registry to use (defaults to global)
            auto_match_threshold: Minimum score for auto-matching
            use_bm25: Use BM25 search for skill matching (faster, more accurate)
        """
        self.registry = registry or SKILL_REGISTRY
        self.auto_match_threshold = auto_match_threshold
        self._use_bm25 = use_bm25
        self._search_index = SkillSearchIndex()
        self._index_built = False

    def _ensure_index(self) -> None:
        """Ensure BM25 index is built (lazy initialization)."""
        if self._use_bm25 and not self._index_built:
            skills = self.registry.list_skills()
            self._search_index.build_index(skills)
            self._index_built = True

    def rebuild_index(self) -> None:
        """Rebuild the BM25 index (call after skills change)."""
        if self._use_bm25:
            skills = self.registry.list_skills()
            self._search_index.build_index(skills)
            self._index_built = True
            logger.debug(f"Rebuilt BM25 index with {len(skills)} skills")

    def parse_slash_command(self, message: str) -> tuple[Optional[str], str]:
        """Parse /skill-name from message.

        Args:
            message: User message

        Returns:
            Tuple of (skill_name, remaining_message) or (None, original_message)
        """
        message = message.strip()
        match = SLASH_COMMAND_PATTERN.match(message)

        if match:
            skill_name = match.group(1)
            remaining = match.group(2) or ""
            return skill_name, remaining.strip()

        return None, message

    def invoke_by_name(self, skill_name: str) -> Skill:
        """Get skill by explicit name.

        Args:
            skill_name: Skill name

        Returns:
            Skill instance

        Raises:
            SkillNotFoundError: If skill not found or not user-invokable
        """
        skill = self.registry.get(skill_name)

        if not skill.metadata.user_invokable:
            raise SkillNotFoundError(
                f"Skill '{skill_name}' exists but is not user-invokable"
            )

        return skill

    def match_skill(
        self,
        message: str,
        provider: Optional[str] = None,
    ) -> Optional[Tuple[Skill, float]]:
        """Auto-match skill based on message content.

        Uses BM25 search for ranking when available, falling back to
        keyword matching. Filters by provider compatibility and
        disable_model_invocation DURING iteration to ensure compatible
        lower-scoring skills are still considered.

        CRITICAL: Evaluates ALL skills, not just top-k, to avoid missing
        compatible skills ranked beyond an arbitrary cutoff.

        Args:
            message: User message to match
            provider: Optional provider to filter compatible skills

        Returns:
            Tuple of (matched_skill, score) or None
        """
        if self._use_bm25:
            return self._match_skill_bm25(message, provider)
        else:
            return self._match_skill_linear(message, provider)

    def _match_skill_bm25(
        self,
        message: str,
        provider: Optional[str] = None,
    ) -> Optional[Tuple[Skill, float]]:
        """Match skill using BM25 search.

        Args:
            message: User message to match
            provider: Optional provider to filter compatible skills

        Returns:
            Tuple of (matched_skill, score) or None
        """
        self._ensure_index()

        # Get ALL candidates - do NOT truncate before compatibility filtering
        # This matches the original behavior where all skills are evaluated
        num_skills = self._search_index.num_skills
        if num_skills == 0:
            return None

        candidates = self._search_index.search(
            query=message,
            top_k=num_skills,  # Get ALL skills
            threshold=0.0,  # Filter by auto_match_threshold after compatibility
        )

        # Filter during iteration (matches original router behavior)
        for skill, score in candidates:
            # Skip if model invocation is disabled
            if skill.metadata.disable_model_invocation:
                continue

            # Skip if provider not compatible
            if provider and provider not in skill.metadata.providers:
                continue

            # Check threshold after compatibility filtering
            if score >= self.auto_match_threshold:
                return (skill, score)

        return None

    def _match_skill_linear(
        self,
        message: str,
        provider: Optional[str] = None,
    ) -> Optional[Tuple[Skill, float]]:
        """Match skill using linear keyword matching (original behavior).

        Args:
            message: User message to match
            provider: Optional provider to filter compatible skills

        Returns:
            Tuple of (matched_skill, score) or None
        """
        best_skill: Optional[Skill] = None
        best_score = 0.0

        for skill in self.registry.list_skills(user_invokable_only=False):
            # Skip if model invocation is disabled
            if skill.metadata.disable_model_invocation:
                continue

            # Skip if provider not compatible
            if provider and provider not in skill.metadata.providers:
                continue

            score = skill.metadata.matches_query(message)

            if score > best_score:
                best_score = score
                best_skill = skill

        if best_skill and best_score >= self.auto_match_threshold:
            return best_skill, best_score

        return None

    def search_skills(
        self,
        query: str,
        top_k: int = 5,
        tags: Optional[List[str]] = None,
        provider: Optional[str] = None,
    ) -> List[Tuple[Skill, float]]:
        """Search skills with optional filtering.

        CRITICAL: Searches ALL skills before filtering to avoid missing
        valid results ranked beyond an arbitrary cutoff.

        Args:
            query: Search query string
            top_k: Maximum number of results to return
            tags: Filter by tags (any match)
            provider: Filter by provider compatibility

        Returns:
            List of (skill, score) tuples
        """
        self._ensure_index()

        # Get ALL candidates - filter during iteration
        num_skills = self._search_index.num_skills
        if num_skills == 0:
            return []

        candidates = self._search_index.search(query, top_k=num_skills)

        results = []
        for skill, score in candidates:
            # Apply provider filter
            if provider and provider not in skill.metadata.providers:
                continue

            # Apply tag filter
            if tags and not any(tag in skill.metadata.tags for tag in tags):
                continue

            results.append((skill, score))

            if len(results) >= top_k:
                break

        return results

    async def route(
        self,
        message: str,
        explicit_skill: Optional[str] = None,
        auto_match: bool = True,
        provider: Optional[str] = None,
    ) -> SkillContext:
        """Route message to appropriate skill.

        Args:
            message: User message
            explicit_skill: Explicitly requested skill name
            auto_match: Whether to auto-match if no explicit skill
            provider: Provider to filter compatible skills

        Returns:
            SkillContext with matched skill (or None)
        """
        # Check for explicit skill first
        if explicit_skill:
            try:
                skill = self.invoke_by_name(explicit_skill)
                logger.debug(f"Explicit skill invocation: {skill.name}")
                return SkillContext(
                    skill=skill,
                    message=message,
                    explicit_invocation=True,
                    matched_score=1.0,
                )
            except SkillNotFoundError:
                logger.warning(f"Explicit skill not found: {explicit_skill}")
                return SkillContext(skill=None, message=message)

        # Check for /skill-name in message
        parsed_skill, remaining_message = self.parse_slash_command(message)
        if parsed_skill:
            try:
                skill = self.invoke_by_name(parsed_skill)
                logger.debug(f"Slash command skill: {skill.name}")
                return SkillContext(
                    skill=skill,
                    message=remaining_message or message,
                    explicit_invocation=True,
                    matched_score=1.0,
                )
            except SkillNotFoundError:
                logger.warning(f"Slash command skill not found: {parsed_skill}")
                # Continue with auto-match if enabled

        # Auto-match
        if auto_match:
            result = self.match_skill(message, provider=provider)
            if result:
                skill, score = result
                logger.debug(f"Auto-matched skill: {skill.name} (score: {score:.2f})")
                return SkillContext(
                    skill=skill,
                    message=message,
                    explicit_invocation=False,
                    matched_score=score,
                )

        # No skill matched
        return SkillContext(skill=None, message=message)

    def list_available_commands(self) -> List[str]:
        """List available slash commands.

        Returns:
            List of skill names available as slash commands
        """
        return [
            skill.name for skill in self.registry.list_skills(user_invokable_only=True)
        ]

    def get_command_help(self, skill_name: str) -> Optional[str]:
        """Get help text for a skill command.

        Args:
            skill_name: Skill name

        Returns:
            Help text or None if skill not found
        """
        try:
            skill = self.registry.get(skill_name)
            parts = [f"/{skill.name}"]

            if skill.metadata.argument_hint:
                parts.append(skill.metadata.argument_hint)

            parts.append(f"- {skill.metadata.description}")

            return " ".join(parts)
        except SkillNotFoundError:
            return None
