"""SKILL.md parser following agentskills.io specification.

This module handles parsing of SKILL.md files with YAML frontmatter
and markdown body content.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

from marie.agent.skills.models import (
    Skill,
    SkillInstructions,
    SkillMetadata,
    SkillSource,
)

logger = logging.getLogger("marie.agent.skills.parser")

# YAML frontmatter pattern: starts with ---, content, ends with ---
FRONTMATTER_PATTERN = re.compile(
    r"^---\s*\n(.*?)\n---\s*\n",
    re.DOTALL,
)


def parse_frontmatter(content: str) -> Tuple[Dict[str, Any], str]:
    """Parse YAML frontmatter from SKILL.md content.

    Args:
        content: Full file content

    Returns:
        Tuple of (frontmatter_dict, body_content)
    """
    match = FRONTMATTER_PATTERN.match(content)

    if not match:
        # No frontmatter found, treat entire content as body
        return {}, content

    frontmatter_yaml = match.group(1)
    body = content[match.end() :]

    try:
        frontmatter = yaml.safe_load(frontmatter_yaml) or {}
    except yaml.YAMLError as e:
        logger.warning(f"Failed to parse YAML frontmatter: {e}")
        frontmatter = {}

    return frontmatter, body


def parse_skill_file(path: Path) -> Tuple[Dict[str, Any], str]:
    """Parse a SKILL.md file.

    Args:
        path: Path to SKILL.md file

    Returns:
        Tuple of (frontmatter_dict, body_content)

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    if not path.exists():
        raise FileNotFoundError(f"Skill file not found: {path}")

    content = path.read_text(encoding="utf-8")
    return parse_frontmatter(content)


def frontmatter_to_metadata(
    frontmatter: Dict[str, Any],
    skill_path: Optional[Path] = None,
    source: SkillSource = SkillSource.BUILTIN,
) -> SkillMetadata:
    """Convert frontmatter dict to SkillMetadata.

    Args:
        frontmatter: Parsed YAML frontmatter
        skill_path: Path to skill directory
        source: Skill source type

    Returns:
        SkillMetadata instance

    Raises:
        ValueError: If required fields are missing
    """
    # Required fields per agentskills.io spec
    name = frontmatter.get("name")
    description = frontmatter.get("description")

    if not name:
        raise ValueError("Skill must have a 'name' field in frontmatter")
    if not description:
        raise ValueError("Skill must have a 'description' field in frontmatter")

    # Parse allowed-tools (space-separated string or list)
    allowed_tools_raw = frontmatter.get("allowed-tools", frontmatter.get("tools", []))
    if isinstance(allowed_tools_raw, str):
        allowed_tools = allowed_tools_raw.split()
    elif isinstance(allowed_tools_raw, list):
        allowed_tools = allowed_tools_raw
    else:
        allowed_tools = []

    # Parse providers
    providers_raw = frontmatter.get("providers", ["openai", "claude", "vllm"])
    if isinstance(providers_raw, str):
        providers = providers_raw.split()
    elif isinstance(providers_raw, list):
        providers = providers_raw
    else:
        providers = ["openai", "claude", "vllm"]

    # Parse tags
    tags_raw = frontmatter.get("tags", [])
    if isinstance(tags_raw, str):
        tags = tags_raw.split()
    elif isinstance(tags_raw, list):
        tags = tags_raw
    else:
        tags = []

    # Extract metadata sub-object if present
    metadata_obj = frontmatter.get("metadata", {})

    return SkillMetadata(
        name=name,
        description=description,
        version=frontmatter.get("version", metadata_obj.get("version", "1.0.0")),
        user_invokable=frontmatter.get("user-invokable", True),
        disable_model_invocation=frontmatter.get("disable-model-invocation", False),
        argument_hint=frontmatter.get("argument-hint"),
        allowed_tools=allowed_tools,
        providers=providers,
        compatibility=frontmatter.get("compatibility"),
        tags=tags,
        category=metadata_obj.get("category", frontmatter.get("category")),
        license=frontmatter.get("license"),
        author=metadata_obj.get("author", frontmatter.get("author")),
        source=source,
        skill_path=skill_path,
    )


def parse_skill(
    skill_path: Path,
    source: SkillSource = SkillSource.BUILTIN,
    load_instructions: bool = False,
) -> Skill:
    """Parse a complete skill from a directory.

    Args:
        skill_path: Path to skill directory containing SKILL.md
        source: Skill source type
        load_instructions: Whether to eagerly load instructions (level 2)

    Returns:
        Parsed Skill instance

    Raises:
        FileNotFoundError: If SKILL.md doesn't exist
        ValueError: If required fields are missing
    """
    skill_file = skill_path / "SKILL.md"

    if not skill_file.exists():
        raise FileNotFoundError(f"SKILL.md not found in: {skill_path}")

    frontmatter, body = parse_skill_file(skill_file)
    metadata = frontmatter_to_metadata(frontmatter, skill_path, source)

    # Create skill with lazy loading by default
    skill = Skill(
        metadata=metadata,
        _skill_path=skill_path,
    )

    # Optionally load instructions immediately
    if load_instructions:
        skill._instructions = SkillInstructions.from_markdown(body)

    return skill


def validate_skill_structure(skill_path: Path) -> bool:
    """Validate that a directory contains a valid skill.

    Args:
        skill_path: Path to potential skill directory

    Returns:
        True if valid skill structure
    """
    if not skill_path.is_dir():
        return False

    skill_file = skill_path / "SKILL.md"
    if not skill_file.exists():
        return False

    try:
        frontmatter, _ = parse_skill_file(skill_file)
        # Check required fields
        if not frontmatter.get("name") or not frontmatter.get("description"):
            return False
        return True
    except Exception:
        return False
