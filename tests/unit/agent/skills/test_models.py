"""Unit tests for skill models."""

from __future__ import annotations

import pytest

from marie.agent.skills.models import (
    Skill,
    SkillContext,
    SkillExample,
    SkillInstructions,
    SkillMetadata,
    SkillResources,
    SkillSource,
)


class TestSkillMetadata:
    """Tests for SkillMetadata class."""

    def test_create_basic_metadata(self):
        """Test creating basic metadata."""
        metadata = SkillMetadata(
            name="test-skill",
            description="A test skill",
        )
        assert metadata.name == "test-skill"
        assert metadata.description == "A test skill"
        assert metadata.version == "1.0.0"
        assert metadata.user_invokable is True

    def test_metadata_defaults(self):
        """Test metadata default values."""
        metadata = SkillMetadata(name="defaults", description="Test")

        assert metadata.disable_model_invocation is False
        assert metadata.argument_hint is None
        assert metadata.allowed_tools == []
        assert "openai" in metadata.providers
        assert "claude" in metadata.providers
        assert metadata.tags == []
        assert metadata.source == SkillSource.BUILTIN

    def test_matches_query_name_exact(self):
        """Test query matching with exact name match."""
        metadata = SkillMetadata(name="document-extraction", description="Extract docs")

        score = metadata.matches_query("document-extraction")
        assert score >= 0.5  # Name match should score high

    def test_matches_query_name_partial(self):
        """Test query matching with partial name match."""
        metadata = SkillMetadata(name="document-extraction", description="Extract docs")

        score = metadata.matches_query("document")
        assert score > 0  # Should match

    def test_matches_query_description(self):
        """Test query matching with description match."""
        metadata = SkillMetadata(name="extractor", description="Extract text from PDFs")

        score = metadata.matches_query("PDF")
        assert score > 0  # Should match description

    def test_matches_query_tags(self):
        """Test query matching with tag match."""
        metadata = SkillMetadata(
            name="code-tool",
            description="Tool for code",
            tags=["python", "javascript", "typescript"],
        )

        score = metadata.matches_query("python")
        assert score > 0  # Should match tag

    def test_matches_query_no_match(self):
        """Test query matching with no match."""
        metadata = SkillMetadata(
            name="document-tool",
            description="Process documents",
            tags=["ocr", "pdf"],
        )

        score = metadata.matches_query("music")
        assert score == 0


class TestSkillInstructions:
    """Tests for SkillInstructions class."""

    def test_from_markdown_basic(self):
        """Test parsing basic markdown instructions."""
        content = """
## When to Use

Use this when you need to process documents.

## Instructions

1. Read the document
2. Extract text
3. Return results
"""
        instructions = SkillInstructions.from_markdown(content)

        assert "process documents" in instructions.when_to_use
        assert "Read the document" in instructions.instructions

    def test_from_markdown_no_sections(self):
        """Test parsing markdown without standard sections."""
        content = """
This is just plain content without sections.
"""
        instructions = SkillInstructions.from_markdown(content)

        assert instructions.when_to_use == ""
        assert instructions.instructions == ""
        assert instructions.full_content == content

    def test_from_markdown_preserves_full_content(self):
        """Test that full content is preserved."""
        content = """
## When to Use
Use when testing.

## Instructions
Test instructions.

## Custom Section
Custom content.
"""
        instructions = SkillInstructions.from_markdown(content)

        assert instructions.full_content == content


class TestSkillExample:
    """Tests for SkillExample class."""

    def test_create_example(self):
        """Test creating skill example."""
        example = SkillExample(
            user_input="Extract text from invoice.pdf",
            expected_action="Run OCR on invoice.pdf",
            description="PDF extraction example",
        )

        assert example.user_input == "Extract text from invoice.pdf"
        assert example.expected_action == "Run OCR on invoice.pdf"
        assert example.description == "PDF extraction example"

    def test_example_optional_description(self):
        """Test example with optional description."""
        example = SkillExample(
            user_input="test",
            expected_action="action",
        )
        assert example.description is None


class TestSkillResources:
    """Tests for SkillResources class."""

    def test_create_empty_resources(self):
        """Test creating empty resources."""
        resources = SkillResources()

        assert resources.scripts == {}
        assert resources.templates == {}
        assert resources.references == []

    def test_create_with_resources(self):
        """Test creating resources with content."""
        resources = SkillResources(
            scripts={"main.py": "print('hello')"},
            templates={"template.md": "# Template"},
            references=["Reference 1", "Reference 2"],
        )

        assert "main.py" in resources.scripts
        assert "template.md" in resources.templates
        assert len(resources.references) == 2


class TestSkill:
    """Tests for Skill class."""

    def test_skill_name_property(self, sample_skill):
        """Test skill name property."""
        assert sample_skill.name == "test-skill"

    def test_skill_description_property(self, sample_skill):
        """Test skill description property."""
        assert "test skill" in sample_skill.description.lower()

    def test_instructions_loaded_property(self, sample_skill):
        """Test instructions_loaded property."""
        assert sample_skill.instructions_loaded is True

    def test_instructions_not_loaded(self, sample_metadata):
        """Test skill without loaded instructions."""
        skill = Skill(metadata=sample_metadata)
        assert skill.instructions_loaded is False

    def test_resources_loaded_property(self, sample_skill):
        """Test resources_loaded property."""
        assert sample_skill.resources_loaded is False

    def test_get_instructions(self, sample_skill):
        """Test getting instructions."""
        instructions = sample_skill.get_instructions()

        assert "testing" in instructions.when_to_use.lower()
        assert "testing instructions" in instructions.instructions.lower()

    def test_get_instructions_not_loaded_raises(self, sample_metadata):
        """Test getting instructions when not loaded raises error."""
        skill = Skill(metadata=sample_metadata)

        with pytest.raises(RuntimeError, match="Failed to load instructions"):
            skill.get_instructions()

    def test_to_system_prompt_injection(self, sample_skill):
        """Test system prompt generation."""
        prompt = sample_skill.to_system_prompt_injection()

        assert "test-skill" in prompt
        assert "When to Use" in prompt
        assert "Instructions" in prompt
        assert "read, write" in prompt  # Allowed tools


class TestSkillContext:
    """Tests for SkillContext class."""

    def test_context_with_skill(self, sample_skill):
        """Test context with matched skill."""
        context = SkillContext(
            skill=sample_skill,
            message="test message",
            explicit_invocation=True,
            matched_score=1.0,
        )

        assert context.has_skill is True
        assert context.skill is sample_skill
        assert context.explicit_invocation is True
        assert context.matched_score == 1.0

    def test_context_without_skill(self):
        """Test context with no matched skill."""
        context = SkillContext(
            skill=None,
            message="test message",
        )

        assert context.has_skill is False
        assert context.matched_score == 0.0


class TestSkillSource:
    """Tests for SkillSource enum."""

    def test_source_values(self):
        """Test skill source enum values."""
        assert SkillSource.BUILTIN == "builtin"
        assert SkillSource.WORKSPACE == "workspace"
        assert SkillSource.USER == "user"
