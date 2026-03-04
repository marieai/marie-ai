"""Unit tests for skill router."""

from __future__ import annotations

import pytest

from marie.agent.skills.models import Skill, SkillMetadata
from marie.agent.skills.registry import SkillNotFoundError, SkillRegistry
from marie.agent.skills.router import SLASH_COMMAND_PATTERN, SkillRouter


class TestSlashCommandPattern:
    """Tests for slash command regex pattern."""

    def test_simple_command(self):
        """Test simple slash command."""
        match = SLASH_COMMAND_PATTERN.match("/skill")
        assert match is not None
        assert match.group(1) == "skill"
        assert match.group(2) is None

    def test_command_with_args(self):
        """Test command with arguments."""
        match = SLASH_COMMAND_PATTERN.match("/skill arg1 arg2")
        assert match is not None
        assert match.group(1) == "skill"
        assert match.group(2) == "arg1 arg2"

    def test_hyphenated_command(self):
        """Test hyphenated command name."""
        match = SLASH_COMMAND_PATTERN.match("/document-extraction file.pdf")
        assert match is not None
        assert match.group(1) == "document-extraction"
        assert match.group(2) == "file.pdf"

    def test_invalid_command_underscore(self):
        """Test that underscores are not matched."""
        match = SLASH_COMMAND_PATTERN.match("/invalid_command")
        assert match is None

    def test_invalid_no_slash(self):
        """Test that non-slash messages don't match."""
        match = SLASH_COMMAND_PATTERN.match("skill arg")
        assert match is None

    def test_single_char_command(self):
        """Test single character command."""
        match = SLASH_COMMAND_PATTERN.match("/x")
        assert match is not None
        assert match.group(1) == "x"


class TestSkillRouterParsing:
    """Tests for SkillRouter parsing."""

    def test_parse_slash_command(self, skill_router):
        """Test parsing slash command from message."""
        skill_name, remaining = skill_router.parse_slash_command("/doc file.pdf")

        assert skill_name == "doc"
        assert remaining == "file.pdf"

    def test_parse_no_command(self, skill_router):
        """Test parsing message without slash command."""
        skill_name, remaining = skill_router.parse_slash_command("regular message")

        assert skill_name is None
        assert remaining == "regular message"

    def test_parse_preserves_whitespace(self, skill_router):
        """Test that parsing preserves argument whitespace."""
        skill_name, remaining = skill_router.parse_slash_command(
            "/skill   arg with spaces  "
        )

        assert skill_name == "skill"
        assert remaining == "arg with spaces"


class TestSkillRouterInvocation:
    """Tests for explicit skill invocation."""

    def test_invoke_by_name_success(self, skill_router):
        """Test successful invocation by name."""
        skill = skill_router.invoke_by_name("document-extraction")

        assert skill.name == "document-extraction"

    def test_invoke_by_name_not_found(self, skill_router):
        """Test invocation with non-existent skill."""
        with pytest.raises(SkillNotFoundError):
            skill_router.invoke_by_name("nonexistent")

    def test_invoke_non_user_invokable(self, skill_router):
        """Test invocation of non-user-invokable skill raises error."""
        with pytest.raises(SkillNotFoundError, match="not user-invokable"):
            skill_router.invoke_by_name("internal-helper")


class TestSkillRouterMatching:
    """Tests for automatic skill matching."""

    def test_match_skill_by_name(self, skill_router):
        """Test matching skill by name in query."""
        result = skill_router.match_skill("document extraction from PDFs")

        assert result is not None
        skill, score = result
        assert skill.name == "document-extraction"
        assert score > 0

    def test_match_skill_by_description(self, skill_router):
        """Test matching skill by description content."""
        result = skill_router.match_skill("review my code for bugs")

        assert result is not None
        skill, score = result
        assert skill.name == "code-review"

    def test_match_skill_no_match(self, skill_router):
        """Test no match when query doesn't match any skill."""
        result = skill_router.match_skill("play music")

        assert result is None

    def test_match_respects_threshold(self, populated_registry):
        """Test that matching respects auto_match_threshold."""
        router = SkillRouter(
            registry=populated_registry,
            auto_match_threshold=0.9,  # Very high threshold
        )

        # Partial match should be below threshold
        result = router.match_skill("doc")
        # May or may not match depending on score
        if result:
            _, score = result
            assert score >= 0.9

    def test_match_filters_disabled_model_invocation(self, skill_router):
        """Test that skills with disable_model_invocation are skipped."""
        result = skill_router.match_skill("internal helper utility")

        # internal-helper has disable_model_invocation=True
        if result:
            skill, _ = result
            assert skill.name != "internal-helper"

    def test_match_filters_by_provider(self, skill_router):
        """Test filtering by provider."""
        result = skill_router.match_skill(
            "search documentation",
            provider="vllm",
        )

        # search-docs only supports openai and claude
        if result:
            skill, _ = result
            assert skill.name != "search-docs"


class TestSkillRouterRoute:
    """Tests for the main route method."""

    @pytest.mark.asyncio
    async def test_route_explicit_skill(self, skill_router):
        """Test routing with explicit skill name."""
        context = await skill_router.route(
            message="extract text",
            explicit_skill="document-extraction",
        )

        assert context.has_skill is True
        assert context.skill.name == "document-extraction"
        assert context.explicit_invocation is True
        assert context.matched_score == 1.0

    @pytest.mark.asyncio
    async def test_route_explicit_not_found(self, skill_router):
        """Test routing with non-existent explicit skill."""
        context = await skill_router.route(
            message="test",
            explicit_skill="nonexistent",
        )

        assert context.has_skill is False

    @pytest.mark.asyncio
    async def test_route_slash_command(self, skill_router):
        """Test routing via slash command in message."""
        context = await skill_router.route(
            message="/document-extraction invoice.pdf",
        )

        assert context.has_skill is True
        assert context.skill.name == "document-extraction"
        assert context.explicit_invocation is True
        assert context.message == "invoice.pdf"

    @pytest.mark.asyncio
    async def test_route_slash_command_not_found(self, skill_router):
        """Test routing with non-existent slash command."""
        context = await skill_router.route(
            message="/nonexistent arg",
        )

        # Should fall through to auto-match
        assert context.explicit_invocation is False

    @pytest.mark.asyncio
    async def test_route_auto_match(self, skill_router):
        """Test automatic routing via content matching."""
        context = await skill_router.route(
            message="Can you review this code for bugs?",
        )

        assert context.has_skill is True
        assert context.skill.name == "code-review"
        assert context.explicit_invocation is False
        assert context.matched_score > 0

    @pytest.mark.asyncio
    async def test_route_auto_match_disabled(self, skill_router):
        """Test routing with auto_match disabled."""
        context = await skill_router.route(
            message="extract documents",
            auto_match=False,
        )

        # Without explicit skill and auto_match disabled, no skill
        assert context.has_skill is False

    @pytest.mark.asyncio
    async def test_route_no_match(self, skill_router):
        """Test routing when nothing matches."""
        context = await skill_router.route(
            message="play my favorite song",
        )

        assert context.has_skill is False
        assert context.message == "play my favorite song"


class TestSkillRouterCommands:
    """Tests for command listing and help."""

    def test_list_available_commands(self, skill_router):
        """Test listing available slash commands."""
        commands = skill_router.list_available_commands()

        assert "document-extraction" in commands
        assert "code-review" in commands
        assert "internal-helper" not in commands  # Not user-invokable

    def test_get_command_help(self, populated_registry):
        """Test getting help for a command."""
        # Add a skill with argument hint
        skill = Skill(
            metadata=SkillMetadata(
                name="test-cmd",
                description="Test command skill",
                argument_hint="<file>",
                user_invokable=True,
            )
        )
        populated_registry.register_skill(skill)
        router = SkillRouter(registry=populated_registry)

        help_text = router.get_command_help("test-cmd")

        assert help_text is not None
        assert "/test-cmd" in help_text
        assert "<file>" in help_text
        assert "Test command" in help_text

    def test_get_command_help_not_found(self, skill_router):
        """Test getting help for non-existent command."""
        help_text = skill_router.get_command_help("nonexistent")
        assert help_text is None


class TestSkillRouterEdgeCases:
    """Tests for edge cases and special scenarios."""

    @pytest.mark.asyncio
    async def test_route_empty_message(self, skill_router):
        """Test routing with empty message."""
        context = await skill_router.route(message="")

        assert context.has_skill is False

    @pytest.mark.asyncio
    async def test_route_whitespace_message(self, skill_router):
        """Test routing with whitespace-only message."""
        context = await skill_router.route(message="   ")

        assert context.has_skill is False

    @pytest.mark.asyncio
    async def test_route_preserves_message(self, skill_router):
        """Test that original message is preserved in context."""
        original = "Can you help me extract text from documents?"
        context = await skill_router.route(message=original)

        assert context.message == original

    def test_router_with_custom_threshold(self, populated_registry):
        """Test router with custom auto_match_threshold."""
        router = SkillRouter(
            registry=populated_registry,
            auto_match_threshold=0.1,  # Very low threshold
        )

        # Should match even weak queries
        result = router.match_skill("doc")
        assert result is not None

    def test_router_with_empty_registry(self):
        """Test router with empty registry."""
        empty_registry = SkillRegistry()
        router = SkillRouter(registry=empty_registry)

        result = router.match_skill("anything")
        assert result is None

        commands = router.list_available_commands()
        assert commands == []
