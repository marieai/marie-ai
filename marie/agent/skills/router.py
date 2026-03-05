"""Skill router for request-to-skill matching.

This module provides routing logic to match user messages to
appropriate skills, supporting both explicit invocation (/skill-name)
and automatic matching.
"""

from __future__ import annotations

import re
from typing import List, Optional

from marie.agent.skills.models import Skill, SkillContext
from marie.agent.skills.registry import (
    SKILL_REGISTRY,
    SkillNotFoundError,
    SkillRegistry,
)
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
    ):
        """Initialize router.

        Args:
            registry: Skill registry to use (defaults to global)
            auto_match_threshold: Minimum score for auto-matching
        """
        self.registry = registry or SKILL_REGISTRY
        self.auto_match_threshold = auto_match_threshold

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
    ) -> Optional[tuple[Skill, float]]:
        """Auto-match skill based on message content.

        Uses keyword matching against skill descriptions and tags.
        For more sophisticated matching, an LLM can be used externally.

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
