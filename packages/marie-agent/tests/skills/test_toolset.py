"""Tests for agent-facing skill tools."""

from __future__ import annotations

import json

from marie.agent.skills.registry import SKILL_REGISTRY
from marie.agent.skills.toolset import (
    SkillToolset,
    discover_skills,
    load_skill,
    load_skill_resource,
)


class TestDiscoverSkills:
    """Tests for discover_skills tool."""

    def test_returns_json_array(self, populated_registry):
        """Should return valid JSON array."""
        # Temporarily use populated registry
        original = dict(SKILL_REGISTRY._skills)
        SKILL_REGISTRY._skills = dict(populated_registry._skills)

        try:
            result = discover_skills()
            data = json.loads(result)
            assert isinstance(data, list)
        finally:
            SKILL_REGISTRY._skills = original

    def test_includes_required_fields(self, populated_registry):
        """Each skill should have name, description, tags."""
        original = dict(SKILL_REGISTRY._skills)
        SKILL_REGISTRY._skills = dict(populated_registry._skills)

        try:
            result = discover_skills()
            data = json.loads(result)
            for skill in data:
                assert "name" in skill
                assert "description" in skill
                assert "tags" in skill
                assert "user_invokable" in skill
                assert "allowed_tools" in skill
        finally:
            SKILL_REGISTRY._skills = original

    def test_filter_by_tags(self, populated_registry):
        """Should filter by tags."""
        original = dict(SKILL_REGISTRY._skills)
        SKILL_REGISTRY._skills = dict(populated_registry._skills)

        try:
            result = discover_skills(tags=["document"])
            data = json.loads(result)
            assert len(data) > 0
            assert all("document" in skill["tags"] for skill in data)
        finally:
            SKILL_REGISTRY._skills = original

    def test_search_query(self, populated_registry):
        """Should search by query using BM25."""
        original = dict(SKILL_REGISTRY._skills)
        SKILL_REGISTRY._skills = dict(populated_registry._skills)

        try:
            result = discover_skills(query="document")
            data = json.loads(result)
            assert len(data) > 0
            # First result should be most relevant
            assert data[0]["name"] == "document-extraction"
        finally:
            SKILL_REGISTRY._skills = original

    def test_respects_limit(self, populated_registry):
        """Should respect limit parameter."""
        original = dict(SKILL_REGISTRY._skills)
        SKILL_REGISTRY._skills = dict(populated_registry._skills)

        try:
            result = discover_skills(limit=2)
            data = json.loads(result)
            assert len(data) <= 2
        finally:
            SKILL_REGISTRY._skills = original

    def test_empty_registry(self, clean_registry):
        """Should return empty array for empty registry."""
        original = dict(SKILL_REGISTRY._skills)
        SKILL_REGISTRY._skills = dict(clean_registry._skills)

        try:
            result = discover_skills()
            data = json.loads(result)
            assert data == []
        finally:
            SKILL_REGISTRY._skills = original


class TestLoadSkill:
    """Tests for load_skill tool."""

    def test_returns_structured_json(self, populated_registry):
        """Should return structured JSON with instructions."""
        original = dict(SKILL_REGISTRY._skills)
        SKILL_REGISTRY._skills = dict(populated_registry._skills)

        try:
            result = load_skill(name="document-extraction")
            data = json.loads(result)
            assert "name" in data
            assert "description" in data
            assert "when_to_use" in data
            assert "instructions" in data
            assert "examples" in data
            assert "allowed_tools" in data
        finally:
            SKILL_REGISTRY._skills = original

    def test_skill_not_found(self, populated_registry):
        """Should return error for unknown skill."""
        original = dict(SKILL_REGISTRY._skills)
        SKILL_REGISTRY._skills = dict(populated_registry._skills)

        try:
            result = load_skill(name="nonexistent-skill")
            data = json.loads(result)
            assert "error" in data
            assert "available_skills" in data
            assert "hint" in data
        finally:
            SKILL_REGISTRY._skills = original

    def test_include_full_content(self, skill_with_instructions):
        """Should include full_content when requested."""
        original = dict(SKILL_REGISTRY._skills)
        SKILL_REGISTRY._skills = {
            "test-skill-with-instructions": skill_with_instructions
        }

        try:
            result = load_skill(
                name="test-skill-with-instructions", include_full_content=True
            )
            data = json.loads(result)
            assert "full_content" in data
        finally:
            SKILL_REGISTRY._skills = original

    def test_examples_format(self, skill_with_instructions):
        """Examples should be properly formatted."""
        original = dict(SKILL_REGISTRY._skills)
        SKILL_REGISTRY._skills = {
            "test-skill-with-instructions": skill_with_instructions
        }

        try:
            result = load_skill(name="test-skill-with-instructions")
            data = json.loads(result)
            for example in data["examples"]:
                assert "input" in example
                assert "action" in example
        finally:
            SKILL_REGISTRY._skills = original


class TestLoadSkillResource:
    """Tests for load_skill_resource tool."""

    def test_loads_reference(self, skill_with_resources):
        """Should load reference file content."""
        original = dict(SKILL_REGISTRY._skills)
        SKILL_REGISTRY._skills = {"test-skill-with-resources": skill_with_resources}

        try:
            result = load_skill_resource(
                skill_name="test-skill-with-resources",
                resource_path="guide.md",
                resource_type="references",
            )
            # Should return the content, not JSON error
            assert "error" not in result.lower() or "guide content" in result.lower()
        finally:
            SKILL_REGISTRY._skills = original

    def test_loads_script(self, skill_with_resources):
        """Should load script content."""
        original = dict(SKILL_REGISTRY._skills)
        SKILL_REGISTRY._skills = {"test-skill-with-resources": skill_with_resources}

        try:
            result = load_skill_resource(
                skill_name="test-skill-with-resources",
                resource_path="helper.py",
                resource_type="scripts",
            )
            assert "print" in result or "error" in result.lower()
        finally:
            SKILL_REGISTRY._skills = original

    def test_loads_template(self, skill_with_resources):
        """Should load template content."""
        original = dict(SKILL_REGISTRY._skills)
        SKILL_REGISTRY._skills = {"test-skill-with-resources": skill_with_resources}

        try:
            result = load_skill_resource(
                skill_name="test-skill-with-resources",
                resource_path="output.template",
                resource_type="templates",
            )
            assert "Template" in result or "error" in result.lower()
        finally:
            SKILL_REGISTRY._skills = original

    def test_skill_not_found(self, clean_registry):
        """Should return error for unknown skill."""
        original = dict(SKILL_REGISTRY._skills)
        SKILL_REGISTRY._skills = dict(clean_registry._skills)

        try:
            result = load_skill_resource(
                skill_name="nonexistent",
                resource_path="guide.md",
                resource_type="references",
            )
            data = json.loads(result)
            assert "error" in data
            assert "hint" in data
        finally:
            SKILL_REGISTRY._skills = original

    def test_resource_not_found(self, skill_with_resources):
        """Should return error for unknown resource."""
        original = dict(SKILL_REGISTRY._skills)
        SKILL_REGISTRY._skills = {"test-skill-with-resources": skill_with_resources}

        try:
            result = load_skill_resource(
                skill_name="test-skill-with-resources",
                resource_path="nonexistent.md",
                resource_type="references",
            )
            data = json.loads(result)
            assert "error" in data
            assert "available_resources" in data
        finally:
            SKILL_REGISTRY._skills = original

    def test_invalid_resource_type(self, skill_with_resources):
        """Should return error for invalid resource type."""
        original = dict(SKILL_REGISTRY._skills)
        SKILL_REGISTRY._skills = {"test-skill-with-resources": skill_with_resources}

        try:
            result = load_skill_resource(
                skill_name="test-skill-with-resources",
                resource_path="guide.md",
                resource_type="invalid",
            )
            data = json.loads(result)
            assert "error" in data
            assert "valid_types" in data
            assert "references" in data["valid_types"]
            assert "scripts" in data["valid_types"]
            assert "templates" in data["valid_types"]
        finally:
            SKILL_REGISTRY._skills = original


class TestSkillToolset:
    """Tests for SkillToolset class."""

    def test_get_tool_names(self):
        """Should return all tool names."""
        names = SkillToolset.get_tool_names()
        assert "discover_skills" in names
        assert "load_skill" in names
        assert "load_skill_resource" in names
        assert len(names) == 3

    def test_get_tool_names_returns_copy(self):
        """Should return a copy, not the original list."""
        names1 = SkillToolset.get_tool_names()
        names2 = SkillToolset.get_tool_names()
        assert names1 is not names2
        assert names1 == names2

    def test_is_registered(self):
        """Should detect registration status."""
        # Tools are auto-registered on import
        from marie.agent.skills import toolset  # noqa: F401

        assert SkillToolset.is_registered()
