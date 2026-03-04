"""Skill validation per agentskills.io specification.

This module provides validation for SKILL.md files and skill metadata
to ensure compliance with the agentskills.io standard.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from marie.agent.skills.models import Skill, SkillMetadata
from marie.logging_core.logger import MarieLogger

logger = MarieLogger("marie.agent.skills.validator")


@dataclass
class ValidationError:
    """A single validation error."""

    field: str
    message: str
    severity: str = "error"  # error, warning, info


@dataclass
class ValidationResult:
    """Result of skill validation."""

    valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)

    def add_error(self, field: str, message: str) -> None:
        """Add a validation error."""
        self.errors.append(
            ValidationError(field=field, message=message, severity="error")
        )
        self.valid = False

    def add_warning(self, field: str, message: str) -> None:
        """Add a validation warning."""
        self.warnings.append(
            ValidationError(field=field, message=message, severity="warning")
        )

    def __bool__(self) -> bool:
        """Return True if validation passed."""
        return self.valid


# agentskills.io required fields
REQUIRED_FIELDS = ["name", "description"]

# agentskills.io recommended fields
RECOMMENDED_FIELDS = ["version", "license", "compatibility"]

# Valid field names per spec
VALID_FIELDS = {
    "name",
    "description",
    "version",
    "license",
    "compatibility",
    "allowed-tools",
    "tools",  # alias for allowed-tools
    "user-invokable",
    "disable-model-invocation",
    "argument-hint",
    "providers",
    "tags",
    "category",
    "author",
    "metadata",
}

# Name validation
MAX_NAME_LENGTH = 64
NAME_PATTERN_DESCRIPTION = "lowercase letters, numbers, and hyphens"


def validate_name(name: str) -> List[ValidationError]:
    """Validate skill name per agentskills.io spec.

    Args:
        name: Skill name to validate

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    if not name:
        errors.append(ValidationError("name", "Name is required"))
        return errors

    if len(name) > MAX_NAME_LENGTH:
        errors.append(
            ValidationError(
                "name",
                f"Name must be {MAX_NAME_LENGTH} characters or less (got {len(name)})",
            )
        )

    # Check for valid characters (lowercase, numbers, hyphens)
    import re

    if not re.match(r"^[a-z0-9][a-z0-9-]*[a-z0-9]$|^[a-z0-9]$", name):
        errors.append(
            ValidationError(
                "name",
                f"Name must contain only {NAME_PATTERN_DESCRIPTION}, "
                "start and end with a letter or number",
            )
        )

    return errors


def validate_description(description: str) -> List[ValidationError]:
    """Validate skill description.

    Args:
        description: Skill description to validate

    Returns:
        List of validation errors
    """
    errors = []

    if not description:
        errors.append(ValidationError("description", "Description is required"))
        return errors

    if len(description) < 10:
        errors.append(
            ValidationError(
                "description",
                "Description should be at least 10 characters",
                severity="warning",
            )
        )

    return errors


def validate_frontmatter(frontmatter: Dict[str, Any]) -> ValidationResult:
    """Validate skill frontmatter per agentskills.io spec.

    Args:
        frontmatter: Parsed YAML frontmatter dict

    Returns:
        ValidationResult with errors and warnings
    """
    result = ValidationResult(valid=True)

    # Check required fields
    for field_name in REQUIRED_FIELDS:
        if field_name not in frontmatter or not frontmatter[field_name]:
            result.add_error(field_name, f"Required field '{field_name}' is missing")

    # Validate name
    if "name" in frontmatter:
        for error in validate_name(frontmatter["name"]):
            if error.severity == "error":
                result.add_error(error.field, error.message)
            else:
                result.add_warning(error.field, error.message)

    # Validate description
    if "description" in frontmatter:
        for error in validate_description(frontmatter["description"]):
            if error.severity == "error":
                result.add_error(error.field, error.message)
            else:
                result.add_warning(error.field, error.message)

    # Check for recommended fields
    for field_name in RECOMMENDED_FIELDS:
        if field_name not in frontmatter:
            result.add_warning(
                field_name,
                f"Recommended field '{field_name}' is missing",
            )

    # Warn about unknown fields
    for field_name in frontmatter:
        if field_name not in VALID_FIELDS:
            result.add_warning(
                field_name,
                f"Unknown field '{field_name}' (may be ignored)",
            )

    return result


def validate_skill_directory(skill_path: Path) -> ValidationResult:
    """Validate a skill directory structure.

    Args:
        skill_path: Path to skill directory

    Returns:
        ValidationResult with errors and warnings
    """
    result = ValidationResult(valid=True)

    # Check directory exists
    if not skill_path.exists():
        result.add_error("path", f"Directory does not exist: {skill_path}")
        return result

    if not skill_path.is_dir():
        result.add_error("path", f"Path is not a directory: {skill_path}")
        return result

    # Check SKILL.md exists
    skill_file = skill_path / "SKILL.md"
    if not skill_file.exists():
        result.add_error("SKILL.md", "SKILL.md file is required")
        return result

    # Parse and validate frontmatter
    try:
        from marie.agent.skills.parser import parse_skill_file

        frontmatter, body = parse_skill_file(skill_file)
        fm_result = validate_frontmatter(frontmatter)

        # Merge results
        result.errors.extend(fm_result.errors)
        result.warnings.extend(fm_result.warnings)
        result.valid = result.valid and fm_result.valid

    except Exception as e:
        result.add_error("SKILL.md", f"Failed to parse SKILL.md: {e}")

    # Check for empty body
    if result.valid and not body.strip():
        result.add_warning("content", "SKILL.md has no content after frontmatter")

    return result


def validate_skill(skill: Skill) -> ValidationResult:
    """Validate a parsed Skill object.

    Args:
        skill: Skill to validate

    Returns:
        ValidationResult with errors and warnings
    """
    result = ValidationResult(valid=True)

    # Validate metadata
    metadata = skill.metadata

    for error in validate_name(metadata.name):
        if error.severity == "error":
            result.add_error(error.field, error.message)
        else:
            result.add_warning(error.field, error.message)

    for error in validate_description(metadata.description):
        if error.severity == "error":
            result.add_error(error.field, error.message)
        else:
            result.add_warning(error.field, error.message)

    # Validate tool references
    if metadata.allowed_tools:
        # Just check format, actual tool existence checked at runtime
        for tool_name in metadata.allowed_tools:
            if not tool_name or not isinstance(tool_name, str):
                result.add_error(
                    "allowed_tools",
                    f"Invalid tool name: {tool_name}",
                )

    return result
