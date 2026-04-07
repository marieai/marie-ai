"""Fixtures for skills tests."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List

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
from marie.agent.skills.registry import SKILL_REGISTRY, SkillRegistry
from marie.agent.skills.router import SkillRouter


@pytest.fixture
def clean_registry():
    """Provide a clean skill registry for each test."""
    registry = SkillRegistry()
    yield registry
    registry.clear()


@pytest.fixture
def populated_registry(clean_registry):
    """Registry with some test skills."""
    skills = [
        Skill(
            metadata=SkillMetadata(
                name="document-extraction",
                description="Extract text and data from documents",
                tags=["document", "extraction", "ocr"],
                user_invokable=True,
            ),
        ),
        Skill(
            metadata=SkillMetadata(
                name="code-review",
                description="Review code for best practices and bugs",
                tags=["code", "review", "quality"],
                user_invokable=True,
            ),
        ),
        Skill(
            metadata=SkillMetadata(
                name="internal-helper",
                description="Internal helper skill",
                user_invokable=False,
                disable_model_invocation=True,
            ),
        ),
        Skill(
            metadata=SkillMetadata(
                name="search-docs",
                description="Search documentation for answers",
                tags=["search", "documentation"],
                user_invokable=True,
                providers=["openai", "claude"],
            ),
        ),
    ]

    for skill in skills:
        clean_registry.register_skill(skill)

    return clean_registry


@pytest.fixture
def sample_skill():
    """Single test skill."""
    return Skill(
        metadata=SkillMetadata(
            name="test-skill",
            description="A test skill for unit testing",
            version="1.0.0",
            tags=["test", "sample"],
            user_invokable=True,
            allowed_tools=["read", "write"],
            providers=["openai", "claude", "vllm"],
        ),
        _instructions=SkillInstructions(
            when_to_use="Use this skill when testing",
            instructions="Follow these testing instructions",
            examples=[
                SkillExample(
                    user_input="Run test",
                    expected_action="Execute test suite",
                )
            ],
        ),
    )


@pytest.fixture
def sample_metadata():
    """Test skill metadata."""
    return SkillMetadata(
        name="sample-skill",
        description="Sample skill for testing metadata",
        version="2.0.0",
        tags=["sample", "metadata"],
        user_invokable=True,
        argument_hint="<file>",
        allowed_tools=["bash", "read"],
        category="testing",
    )


@pytest.fixture
def skill_router(populated_registry):
    """Router with populated registry."""
    return SkillRouter(registry=populated_registry)


@pytest.fixture
def skill_with_instructions():
    """Skill with loaded instructions."""
    return Skill(
        metadata=SkillMetadata(
            name="test-skill-with-instructions",
            description="A skill with full instructions",
            version="1.0.0",
            tags=["test"],
            user_invokable=True,
            allowed_tools=["read", "write"],
        ),
        _instructions=SkillInstructions(
            when_to_use="Use this skill when testing instructions loading",
            instructions="Follow these detailed testing instructions:\n1. Do this\n2. Then that",
            examples=[
                SkillExample(
                    user_input="Test the feature",
                    expected_action="Run the test suite",
                ),
                SkillExample(
                    user_input="Validate output",
                    expected_action="Check the results",
                ),
            ],
            full_content="# Full Content\nThis is the full markdown content.",
        ),
    )


@pytest.fixture
def skill_with_resources():
    """Skill with loaded resources (Dict-based references)."""
    return Skill(
        metadata=SkillMetadata(
            name="test-skill-with-resources",
            description="A skill with resources",
            version="1.0.0",
            tags=["test", "resources"],
            user_invokable=True,
        ),
        _resources=SkillResources(
            scripts={
                "helper.py": "#!/usr/bin/env python\nprint('Hello from helper')",
                "process.py": "#!/usr/bin/env python\nprint('Processing...')",
            },
            templates={
                "output.template": "Template content: {{ value }}",
                "report.template": "Report: {{ data }}",
            },
            references={
                "guide.md": "# Guide Content\n\nThis is the reference guide.",
                "api.md": "# API Reference\n\nAPI documentation here.",
            },
        ),
    )


@pytest.fixture
def skill_dir(tmp_path):
    """Temporary directory structure for skill discovery."""
    skill_path = tmp_path / "test-skill"
    skill_path.mkdir()

    # Create SKILL.md
    skill_md = skill_path / "SKILL.md"
    skill_md.write_text("""---
name: test-skill
description: A test skill
version: 1.0.0
tags:
  - test
  - fixture
user_invokable: true
---

## When to Use

Use this skill when running tests.

## Instructions

1. Read the input
2. Process the data
3. Return results

## Examples

- Input: "test this"
- Action: Run tests
""")

    return skill_path


@pytest.fixture
def skills_directory(tmp_path):
    """Directory with multiple skills."""
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()

    # Skill 1
    skill1 = skills_dir / "skill-one"
    skill1.mkdir()
    (skill1 / "SKILL.md").write_text("""---
name: skill-one
description: First test skill
version: 1.0.0
user_invokable: true
---

## Instructions
Do the first thing.
""")

    # Skill 2
    skill2 = skills_dir / "skill-two"
    skill2.mkdir()
    (skill2 / "SKILL.md").write_text("""---
name: skill-two
description: Second test skill
version: 1.0.0
user_invokable: false
---

## Instructions
Do the second thing.
""")

    # Invalid skill (no SKILL.md)
    invalid = skills_dir / "invalid-skill"
    invalid.mkdir()
    (invalid / "README.md").write_text("Not a skill")

    return skills_dir
