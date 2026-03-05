"""Unit tests for skill registry."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from marie.agent.skills.models import Skill, SkillMetadata, SkillSource
from marie.agent.skills.registry import (
    SKILL_REGISTRY,
    SkillNotFoundError,
    SkillRegistry,
    get_skill,
    list_skills,
    register_skill,
)


class TestSkillRegistry:
    """Tests for SkillRegistry class."""

    def test_register_skill(self, clean_registry, sample_skill):
        """Test registering a skill."""
        clean_registry.register_skill(sample_skill)

        assert clean_registry.has("test-skill")
        assert len(clean_registry) == 1

    def test_register_duplicate_overwrites(self, clean_registry, sample_skill):
        """Test registering duplicate skill warns and overwrites."""
        clean_registry.register_skill(sample_skill)


        new_skill = Skill(
            metadata=SkillMetadata(
                name="test-skill",
                description="Updated description",
            )
        )
        clean_registry.register_skill(new_skill)

        skill = clean_registry.get("test-skill")
        assert skill.description == "Updated description"

    def test_get_skill(self, populated_registry):
        """Test getting a skill by name."""
        skill = populated_registry.get("document-extraction")

        assert skill.name == "document-extraction"
        assert "document" in skill.description.lower()

    def test_get_skill_not_found(self, clean_registry):
        """Test getting non-existent skill raises error."""
        with pytest.raises(SkillNotFoundError) as exc_info:
            clean_registry.get("nonexistent")

        assert "nonexistent" in str(exc_info.value)

    def test_get_or_none(self, populated_registry):
        """Test get_or_none method."""
        skill = populated_registry.get_or_none("document-extraction")
        assert skill is not None

        none_skill = populated_registry.get_or_none("nonexistent")
        assert none_skill is None

    def test_has_skill(self, populated_registry):
        """Test has method."""
        assert populated_registry.has("document-extraction") is True
        assert populated_registry.has("nonexistent") is False

    def test_unregister_skill(self, populated_registry):
        """Test unregistering a skill."""
        assert populated_registry.has("code-review")

        result = populated_registry.unregister_skill("code-review")

        assert result is True
        assert populated_registry.has("code-review") is False

    def test_unregister_nonexistent(self, clean_registry):
        """Test unregistering non-existent skill returns False."""
        result = clean_registry.unregister_skill("nonexistent")
        assert result is False

    def test_list_skills_all(self, populated_registry):
        """Test listing all skills."""
        skills = populated_registry.list_skills()
        assert len(skills) == 4

    def test_list_skills_by_source(self, clean_registry):
        """Test filtering skills by source."""
        builtin = Skill(
            metadata=SkillMetadata(
                name="builtin-skill",
                description="Builtin",
                source=SkillSource.BUILTIN,
            )
        )
        user = Skill(
            metadata=SkillMetadata(
                name="user-skill",
                description="User",
                source=SkillSource.USER,
            )
        )
        clean_registry.register_skill(builtin)
        clean_registry.register_skill(user)

        builtin_skills = clean_registry.list_skills(source=SkillSource.BUILTIN)
        assert len(builtin_skills) == 1
        assert builtin_skills[0].name == "builtin-skill"

    def test_list_skills_by_tags(self, populated_registry):
        """Test filtering skills by tags."""
        skills = populated_registry.list_skills(tags=["document"])
        assert len(skills) == 1
        assert skills[0].name == "document-extraction"

    def test_list_skills_user_invokable_only(self, populated_registry):
        """Test filtering to user-invokable only."""
        skills = populated_registry.list_skills(user_invokable_only=True)

        names = [s.name for s in skills]
        assert "internal-helper" not in names
        assert "document-extraction" in names

    def test_list_metadata(self, populated_registry):
        """Test listing metadata for all skills."""
        metadata_list = populated_registry.list_metadata()

        assert len(metadata_list) == 4
        assert all(isinstance(m, SkillMetadata) for m in metadata_list)

    def test_list_user_invokable(self, populated_registry):
        """Test listing user-invokable skill metadata."""
        metadata_list = populated_registry.list_user_invokable()

        names = [m.name for m in metadata_list]
        assert "internal-helper" not in names
        assert len(metadata_list) == 3

    def test_search_skills(self, populated_registry):
        """Test searching skills by query."""
        results = populated_registry.search_skills("document")

        assert len(results) > 0
        assert results[0].name == "document-extraction"

    def test_search_skills_with_limit(self, populated_registry):
        """Test search with limit."""
        results = populated_registry.search_skills("", limit=2)
        assert len(results) <= 2

    def test_clear_registry(self, populated_registry):
        """Test clearing the registry."""
        assert len(populated_registry) > 0

        populated_registry.clear()

        assert len(populated_registry) == 0

    def test_len(self, populated_registry):
        """Test __len__ method."""
        assert len(populated_registry) == 4

    def test_contains(self, populated_registry):
        """Test __contains__ method."""
        assert "document-extraction" in populated_registry
        assert "nonexistent" not in populated_registry

    def test_iter(self, populated_registry):
        """Test iterating over registry."""
        skills = list(populated_registry)
        assert len(skills) == 4

    def test_thread_safety(self, clean_registry):
        """Test registry operations are thread-safe."""
        errors = []

        def register_skills(start_idx):
            try:
                for i in range(10):
                    skill = Skill(
                        metadata=SkillMetadata(
                            name=f"skill-{start_idx}-{i}",
                            description=f"Thread test {start_idx}-{i}",
                        )
                    )
                    clean_registry.register_skill(skill)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=register_skills, args=(i,))
            for i in range(5)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(clean_registry) == 50


class TestSkillDiscovery:
    """Tests for skill discovery from filesystem."""

    def test_discover_skills(self, clean_registry, skills_directory):
        """Test discovering skills from directory."""
        count = clean_registry.discover_skills([skills_directory])

        assert count == 2  # skill-one and skill-two
        assert clean_registry.has("skill-one")
        assert clean_registry.has("skill-two")

    def test_discover_invalid_path(self, clean_registry, tmp_path):
        """Test discovering from non-existent path."""
        nonexistent = tmp_path / "nonexistent"
        count = clean_registry.discover_skills([nonexistent])

        assert count == 0

    def test_discover_clear_existing(self, populated_registry, skills_directory):
        """Test clear_existing parameter."""
        initial_count = len(populated_registry)
        assert initial_count > 0

        count = populated_registry.discover_skills(
            [skills_directory],
            clear_existing=True,
        )

        assert len(populated_registry) == count
        assert not populated_registry.has("document-extraction")

    def test_discover_sets_source(self, clean_registry, skills_directory):
        """Test that discovered skills have correct source."""
        clean_registry.discover_skills(
            [skills_directory],
            source=SkillSource.WORKSPACE,
        )

        skill = clean_registry.get("skill-one")
        assert skill.metadata.source == SkillSource.WORKSPACE


class TestRegisterSkillDecorator:
    """Tests for @register_skill decorator."""

    def test_decorator_without_args(self):
        """Test decorator without arguments."""
        registry = SkillRegistry()

        # Clear global registry state for test isolation
        original_skills = dict(SKILL_REGISTRY._skills)

        try:
            @register_skill
            class MyTestSkill:
                """A test skill via decorator."""
                pass

            assert SKILL_REGISTRY.has("mytestskill")
        finally:
            SKILL_REGISTRY._skills = original_skills

    def test_decorator_with_name(self):
        """Test decorator with custom name."""
        original_skills = dict(SKILL_REGISTRY._skills)

        try:
            @register_skill(name="custom-name")
            class AnotherSkill:
                """Another test skill."""
                pass

            assert SKILL_REGISTRY.has("custom-name")
        finally:
            SKILL_REGISTRY._skills = original_skills

    def test_decorator_with_metadata(self):
        """Test decorator with metadata kwargs."""
        original_skills = dict(SKILL_REGISTRY._skills)

        try:
            @register_skill(
                name="meta-skill",
                description="Custom description",
                tags=["test", "meta"],
            )
            class MetaSkill:
                pass

            skill = SKILL_REGISTRY.get("meta-skill")
            assert skill.metadata.description == "Custom description"
            assert "test" in skill.metadata.tags
        finally:
            SKILL_REGISTRY._skills = original_skills


class TestGlobalFunctions:
    """Tests for global convenience functions."""

    def test_get_skill_function(self, populated_registry):
        """Test get_skill global function."""
        # Temporarily replace global registry
        original = dict(SKILL_REGISTRY._skills)
        SKILL_REGISTRY._skills = dict(populated_registry._skills)

        try:
            skill = get_skill("document-extraction")
            assert skill.name == "document-extraction"
        finally:
            SKILL_REGISTRY._skills = original

    def test_list_skills_function(self, populated_registry):
        """Test list_skills global function."""
        original = dict(SKILL_REGISTRY._skills)
        SKILL_REGISTRY._skills = dict(populated_registry._skills)

        try:
            skills = list_skills(user_invokable_only=True)
            assert len(skills) == 3
        finally:
            SKILL_REGISTRY._skills = original
