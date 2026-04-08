"""Integration tests for skill filesystem discovery.

Tests the end-to-end flow: workspace skills written to the filesystem
are discoverable by the skill registry and accessible via agent tools.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from marie.agent.skills.loader import SkillLoader
from marie.agent.skills.models import Skill, SkillSource
from marie.agent.skills.registry import SkillRegistry


class TestFilesystemDiscovery:
    """Test that skills written to .marie/skills/ are discoverable."""

    def _make_loader(self) -> SkillLoader:
        """Create a SkillLoader with a fresh registry to isolate tests."""
        return SkillLoader(registry=SkillRegistry())

    def test_discover_workspace_skill(self, tmp_path: Path):
        """Skills created in .marie/skills/ should be found by discovery."""
        workspace = tmp_path / "workspace"
        skills_dir = workspace / ".marie" / "skills" / "test-skill"
        skills_dir.mkdir(parents=True)

        (skills_dir / "SKILL.md").write_text(
            """---
name: test-skill
description: A workspace test skill
version: 1.0.0
user_invokable: true
tags:
  - test
  - workspace
---

## When to Use

Use this when running integration tests.

## Instructions

1. Read input
2. Process data
3. Return results
"""
        )

        loader = self._make_loader()
        count = loader.discover_workspace_skills(str(workspace))

        assert count >= 1

        skill = loader.registry.get_or_none("test-skill")
        assert skill is not None
        assert skill.name == "test-skill"
        assert skill.description == "A workspace test skill"
        assert skill.metadata.source == SkillSource.WORKSPACE

    def test_workspace_skill_with_resources(self, tmp_path: Path):
        """Skills with resources/ dirs should load references as Dict[str, str]."""
        workspace = tmp_path / "workspace"
        skill_dir = workspace / ".marie" / "skills" / "rich-skill"
        skill_dir.mkdir(parents=True)

        (skill_dir / "SKILL.md").write_text(
            """---
name: rich-skill
description: Skill with resources
version: 1.0.0
user_invokable: true
---

## Instructions

Use the style guide from references.
"""
        )

        # Create resources
        refs_dir = skill_dir / "references"
        refs_dir.mkdir()
        (refs_dir / "style-guide.md").write_text("# Style Guide\nUse consistent naming.")
        (refs_dir / "api-reference.md").write_text("# API Reference\nEndpoints here.")

        scripts_dir = skill_dir / "scripts"
        scripts_dir.mkdir()
        (scripts_dir / "helper.py").write_text("#!/usr/bin/env python\nprint('hello')")

        loader = self._make_loader()
        loader.discover_workspace_skills(str(workspace))

        skill = loader.registry.get_or_none("rich-skill")
        assert skill is not None

        # Load resources
        loader.load_skill_resources(skill)
        resources = skill.get_resources()

        # References should be Dict[str, str] (not List[str])
        assert isinstance(resources.references, dict)
        assert "style-guide.md" in resources.references
        assert "api-reference.md" in resources.references
        assert "Style Guide" in resources.references["style-guide.md"]

        # Scripts
        assert isinstance(resources.scripts, dict)
        assert "helper.py" in resources.scripts

    def test_refresh_discovers_new_skills(self, tmp_path: Path):
        """refresh_workspace_skills() should find skills added after init."""
        workspace = tmp_path / "workspace"
        skills_dir = workspace / ".marie" / "skills"
        skills_dir.mkdir(parents=True)

        # Initial discovery — no skills
        loader = self._make_loader()
        loader.discover_workspace_skills(str(workspace))
        initial_count = len(loader.registry.list_skills())

        # Create a new skill after initial discovery
        new_skill_dir = skills_dir / "late-skill"
        new_skill_dir.mkdir()
        (new_skill_dir / "SKILL.md").write_text(
            """---
name: late-skill
description: Added after initial discovery
version: 1.0.0
user_invokable: true
---

## Instructions

Do things late.
"""
        )

        # Refresh should find the new skill
        refreshed = loader.refresh_workspace_skills(str(workspace))
        assert refreshed >= 1

        skill = loader.registry.get_or_none("late-skill")
        assert skill is not None
        assert skill.description == "Added after initial discovery"

    def test_workspace_overrides_builtin(self, tmp_path: Path):
        """Workspace skills should override builtin skills with same name."""
        builtin_dir = tmp_path / "builtin"
        workspace_dir = tmp_path / "workspace" / ".marie" / "skills"

        # Create builtin skill
        (builtin_dir / "shared-skill").mkdir(parents=True)
        (builtin_dir / "shared-skill" / "SKILL.md").write_text(
            """---
name: shared-skill
description: Builtin version
version: 1.0.0
user_invokable: true
---

## Instructions

Builtin instructions.
"""
        )

        # Create workspace skill with same name
        (workspace_dir / "shared-skill").mkdir(parents=True)
        (workspace_dir / "shared-skill" / "SKILL.md").write_text(
            """---
name: shared-skill
description: Workspace version (override)
version: 2.0.0
user_invokable: true
---

## Instructions

Workspace instructions override.
"""
        )

        loader = self._make_loader()
        # Discover builtin first, then workspace (workspace takes precedence)
        loader.registry.discover_skills(
            paths=[builtin_dir], source=SkillSource.BUILTIN
        )
        loader.registry.discover_skills(
            paths=[workspace_dir], source=SkillSource.WORKSPACE
        )

        skill = loader.registry.get_or_none("shared-skill")
        assert skill is not None
        # Workspace should take precedence
        assert skill.metadata.source == SkillSource.WORKSPACE
        assert skill.description == "Workspace version (override)"

    def test_evals_json_discovery(self, tmp_path: Path):
        """Skills with evals/evals.json should have discoverable eval data."""
        workspace = tmp_path / "workspace"
        skill_dir = workspace / ".marie" / "skills" / "eval-skill"
        evals_dir = skill_dir / "evals"
        evals_dir.mkdir(parents=True)

        (skill_dir / "SKILL.md").write_text(
            """---
name: eval-skill
description: Skill with evaluation test cases
version: 1.0.0
user_invokable: true
---

## Instructions

Do evaluated things.
"""
        )

        evals_data = {
            "skill_name": "eval-skill",
            "evals": [
                {
                    "id": 1,
                    "prompt": "Test prompt",
                    "expected_output": "Expected result",
                    "assertions": [
                        "Output contains a summary",
                        "No errors in output",
                    ],
                }
            ],
        }
        (evals_dir / "evals.json").write_text(json.dumps(evals_data, indent=2))

        loader = self._make_loader()
        loader.discover_workspace_skills(str(workspace))

        skill = loader.registry.get_or_none("eval-skill")
        assert skill is not None

        # Verify evals.json is readable from the skill path
        evals_path = skill.metadata.skill_path / "evals" / "evals.json"
        assert evals_path.exists()

        loaded_evals = json.loads(evals_path.read_text())
        assert loaded_evals["skill_name"] == "eval-skill"
        assert len(loaded_evals["evals"]) == 1
        assert len(loaded_evals["evals"][0]["assertions"]) == 2

    def test_invalid_skill_ignored(self, tmp_path: Path):
        """Directories without valid SKILL.md should be silently skipped."""
        workspace = tmp_path / "workspace"
        skills_dir = workspace / ".marie" / "skills"

        # Valid skill
        valid = skills_dir / "valid-skill"
        valid.mkdir(parents=True)
        (valid / "SKILL.md").write_text(
            """---
name: valid-skill
description: This one is valid
version: 1.0.0
user_invokable: true
---

## Instructions

Valid.
"""
        )

        # Invalid: no SKILL.md
        (skills_dir / "no-skillmd").mkdir()
        (skills_dir / "no-skillmd" / "README.md").write_text("Not a skill")

        # Invalid: bad YAML
        bad_yaml = skills_dir / "bad-yaml"
        bad_yaml.mkdir()
        (bad_yaml / "SKILL.md").write_text("not valid yaml frontmatter")

        # Invalid: just a file, not a directory
        (skills_dir / "just-a-file.txt").write_text("not a directory")

        loader = self._make_loader()
        count = loader.discover_workspace_skills(str(workspace))

        # Only the valid skill should be discovered
        assert count == 1
        assert loader.registry.get_or_none("valid-skill") is not None
