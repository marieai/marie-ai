"""Tests for skill refresh/reindex functionality."""

from __future__ import annotations

from pathlib import Path

import pytest

from marie.agent.skills.loader import SkillLoader
from marie.agent.skills.models import Skill, SkillMetadata, SkillSource
from marie.agent.skills.registry import SkillRegistry


class TestClearSource:
    """Tests for SkillRegistry.clear_source method."""

    def test_clear_workspace_skills_only(self, clean_registry):
        """Should remove only workspace skills, not built-in."""
        # Add mixed skills
        builtin_skill = Skill(
            metadata=SkillMetadata(
                name="builtin-skill",
                description="A built-in skill",
                source=SkillSource.BUILTIN,
            )
        )
        workspace_skill = Skill(
            metadata=SkillMetadata(
                name="workspace-skill",
                description="A workspace skill",
                source=SkillSource.WORKSPACE,
            )
        )
        user_skill = Skill(
            metadata=SkillMetadata(
                name="user-skill",
                description="A user skill",
                source=SkillSource.USER,
            )
        )

        clean_registry.register_skill(builtin_skill)
        clean_registry.register_skill(workspace_skill)
        clean_registry.register_skill(user_skill)

        assert len(clean_registry) == 3

        # Clear only workspace skills
        removed = clean_registry.clear_source(SkillSource.WORKSPACE)

        assert removed == 1
        assert len(clean_registry) == 2
        assert clean_registry.has("builtin-skill")
        assert not clean_registry.has("workspace-skill")
        assert clean_registry.has("user-skill")

    def test_clear_builtin_skills(self, clean_registry):
        """Should be able to clear built-in skills."""
        skill1 = Skill(
            metadata=SkillMetadata(
                name="builtin-1",
                description="First builtin",
                source=SkillSource.BUILTIN,
            )
        )
        skill2 = Skill(
            metadata=SkillMetadata(
                name="builtin-2",
                description="Second builtin",
                source=SkillSource.BUILTIN,
            )
        )

        clean_registry.register_skill(skill1)
        clean_registry.register_skill(skill2)

        removed = clean_registry.clear_source(SkillSource.BUILTIN)

        assert removed == 2
        assert len(clean_registry) == 0

    def test_clear_empty_source(self, clean_registry):
        """Should return 0 when no skills of that source exist."""
        skill = Skill(
            metadata=SkillMetadata(
                name="builtin-skill",
                description="A built-in skill",
                source=SkillSource.BUILTIN,
            )
        )
        clean_registry.register_skill(skill)

        removed = clean_registry.clear_source(SkillSource.WORKSPACE)

        assert removed == 0
        assert len(clean_registry) == 1


class TestRefreshWorkspaceSkills:
    """Tests for SkillLoader.refresh_workspace_skills method."""

    def test_refresh_reloads_from_disk(self, tmp_path):
        """Should rediscover skills from workspace."""
        registry = SkillRegistry()
        loader = SkillLoader(registry=registry)

        # Create workspace skill directory
        workspace_path = tmp_path / "workspace"
        skills_path = workspace_path / ".marie" / "skills"
        skills_path.mkdir(parents=True)

        # Add a skill
        skill_dir = skills_path / "test-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("""---
name: test-skill
description: A test workspace skill
version: 1.0.0
user_invokable: true
---

## Instructions
Do the test thing.
""")

        # Initial discovery
        count = loader.discover_workspace_skills(str(workspace_path))
        assert count == 1
        assert registry.has("test-skill")

        # Add another skill
        skill_dir2 = skills_path / "new-skill"
        skill_dir2.mkdir()
        (skill_dir2 / "SKILL.md").write_text("""---
name: new-skill
description: A new workspace skill
version: 1.0.0
user_invokable: true
---

## Instructions
Do the new thing.
""")

        # Refresh should find the new skill
        count = loader.refresh_workspace_skills(str(workspace_path))

        assert count == 2
        assert registry.has("test-skill")
        assert registry.has("new-skill")

    def test_refresh_preserves_builtin_skills(self, tmp_path):
        """Should preserve built-in skills during refresh."""
        registry = SkillRegistry()
        loader = SkillLoader(registry=registry)

        # Add a built-in skill manually
        builtin_skill = Skill(
            metadata=SkillMetadata(
                name="builtin-skill",
                description="A built-in skill",
                source=SkillSource.BUILTIN,
            )
        )
        registry.register_skill(builtin_skill)

        # Create workspace skill directory
        workspace_path = tmp_path / "workspace"
        skills_path = workspace_path / ".marie" / "skills"
        skills_path.mkdir(parents=True)

        skill_dir = skills_path / "workspace-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("""---
name: workspace-skill
description: A workspace skill
version: 1.0.0
---

## Instructions
Do the workspace thing.
""")

        # Refresh workspace skills
        count = loader.refresh_workspace_skills(str(workspace_path))

        assert count == 1
        assert registry.has("builtin-skill")  # Preserved
        assert registry.has("workspace-skill")  # Added

    def test_refresh_removes_deleted_skills(self, tmp_path):
        """Should remove skills that no longer exist on disk."""
        registry = SkillRegistry()
        loader = SkillLoader(registry=registry)

        # Create workspace with two skills
        workspace_path = tmp_path / "workspace"
        skills_path = workspace_path / ".marie" / "skills"
        skills_path.mkdir(parents=True)

        skill1_dir = skills_path / "skill-one"
        skill1_dir.mkdir()
        (skill1_dir / "SKILL.md").write_text("""---
name: skill-one
description: First skill
---

## Instructions
First.
""")

        skill2_dir = skills_path / "skill-two"
        skill2_dir.mkdir()
        (skill2_dir / "SKILL.md").write_text("""---
name: skill-two
description: Second skill
---

## Instructions
Second.
""")

        # Initial discovery
        loader.discover_workspace_skills(str(workspace_path))
        assert registry.has("skill-one")
        assert registry.has("skill-two")

        # Delete skill-two from disk
        import shutil

        shutil.rmtree(skill2_dir)

        # Refresh should remove skill-two
        count = loader.refresh_workspace_skills(str(workspace_path))

        assert count == 1
        assert registry.has("skill-one")
        assert not registry.has("skill-two")

    def test_refresh_nonexistent_workspace(self, tmp_path):
        """Should return 0 for nonexistent workspace."""
        registry = SkillRegistry()
        loader = SkillLoader(registry=registry)

        nonexistent = tmp_path / "nonexistent"
        count = loader.refresh_workspace_skills(str(nonexistent))

        assert count == 0

    def test_refresh_empty_workspace(self, tmp_path):
        """Should return 0 for empty workspace skills directory."""
        registry = SkillRegistry()
        loader = SkillLoader(registry=registry)

        workspace_path = tmp_path / "workspace"
        skills_path = workspace_path / ".marie" / "skills"
        skills_path.mkdir(parents=True)

        count = loader.refresh_workspace_skills(str(workspace_path))

        assert count == 0
