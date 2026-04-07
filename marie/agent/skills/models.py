"""Skill data models per agentskills.io specification.

This module defines the core data structures for skills following the
agentskills.io open standard for AI agent skills.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class SkillSource(str, Enum):
    """Where the skill originates from."""

    BUILTIN = "builtin"
    WORKSPACE = "workspace"
    USER = "user"


@dataclass
class SkillExample:
    """Example usage of a skill."""

    user_input: str
    expected_action: str
    description: Optional[str] = None


@dataclass
class SkillMetadata:
    """Level 1: Lightweight metadata (~100 tokens).

    This is always loaded for all discovered skills and used for
    skill routing/matching without loading full instructions.

    Follows agentskills.io frontmatter specification.
    """

    name: str
    description: str
    version: str = "1.0.0"

    # Invocation control
    user_invokable: bool = True
    disable_model_invocation: bool = False
    argument_hint: Optional[str] = None

    # Tool integration
    allowed_tools: List[str] = field(default_factory=list)

    # Compatibility
    providers: List[str] = field(default_factory=lambda: ["openai", "claude", "vllm"])
    compatibility: Optional[str] = None

    # Organization
    tags: List[str] = field(default_factory=list)
    category: Optional[str] = None

    # Licensing
    license: Optional[str] = None

    # Extended metadata
    author: Optional[str] = None

    # Source tracking
    source: SkillSource = SkillSource.BUILTIN
    skill_path: Optional[Path] = None

    def matches_query(self, query: str) -> float:
        """Calculate match score against a search query.

        Args:
            query: Search query string

        Returns:
            Match score between 0.0 and 1.0
        """
        query_lower = query.lower().strip()
        if not query_lower:
            return 0.0

        score = 0.0
        name_lower = self.name.lower()
        desc_lower = self.description.lower()

        # Tokenize query into words
        query_words = query_lower.split()

        # Check for exact/partial name match with full query
        if query_lower in name_lower:
            score += 0.5
        if name_lower.startswith(query_lower):
            score += 0.3

        # Check individual words against name
        name_parts = name_lower.replace("-", " ").replace("_", " ").split()
        for word in query_words:
            if len(word) >= 3:  # Skip short words
                if word in name_parts or any(word in part for part in name_parts):
                    score += 0.3
                    break

        # Check individual words against description
        for word in query_words:
            if len(word) >= 3 and word in desc_lower:
                score += 0.2
                break

        # Tag match
        for tag in self.tags:
            tag_lower = tag.lower()
            for word in query_words:
                if len(word) >= 3 and word in tag_lower:
                    score += 0.1
                    break
            if score > 0:
                break

        return min(score, 1.0)


@dataclass
class SkillInstructions:
    """Level 2: Full instructions (<5k tokens).

    Loaded on demand when a skill is selected for use.
    Contains the detailed instructions for the model.
    """

    when_to_use: str
    instructions: str
    examples: List[SkillExample] = field(default_factory=list)
    full_content: str = ""

    @classmethod
    def from_markdown(cls, content: str) -> "SkillInstructions":
        """Parse instructions from SKILL.md body content.

        Args:
            content: Markdown content after frontmatter

        Returns:
            Parsed SkillInstructions
        """
        when_to_use = ""
        instructions = ""
        examples: List[SkillExample] = []

        # Simple section parsing
        current_section = None
        current_content: List[str] = []

        for line in content.split("\n"):
            line_lower = line.lower().strip()

            if line_lower.startswith("## when to use"):
                if current_section and current_content:
                    if current_section == "when_to_use":
                        when_to_use = "\n".join(current_content).strip()
                    elif current_section == "instructions":
                        instructions = "\n".join(current_content).strip()
                current_section = "when_to_use"
                current_content = []
            elif line_lower.startswith("## instructions"):
                if current_section and current_content:
                    if current_section == "when_to_use":
                        when_to_use = "\n".join(current_content).strip()
                current_section = "instructions"
                current_content = []
            elif line_lower.startswith("## example"):
                if current_section and current_content:
                    if current_section == "when_to_use":
                        when_to_use = "\n".join(current_content).strip()
                    elif current_section == "instructions":
                        instructions = "\n".join(current_content).strip()
                current_section = "examples"
                current_content = []
            else:
                current_content.append(line)

        # Handle last section
        if current_section and current_content:
            if current_section == "when_to_use":
                when_to_use = "\n".join(current_content).strip()
            elif current_section == "instructions":
                instructions = "\n".join(current_content).strip()

        return cls(
            when_to_use=when_to_use,
            instructions=instructions,
            examples=examples,
            full_content=content,
        )


@dataclass
class SkillResources:
    """Level 3: On-demand resources (unlimited).

    Loaded only when explicitly requested. Contains additional
    files like scripts, templates, and reference documentation.

    All resource types are keyed by filename for path-addressable access.
    """

    scripts: Dict[str, str] = field(default_factory=dict)
    templates: Dict[str, str] = field(default_factory=dict)
    references: Dict[str, str] = field(default_factory=dict)


@dataclass
class Skill:
    """Complete skill with progressive loading.

    Follows 3-tier loading pattern:
    - Level 1 (metadata): Always loaded, ~100 tokens
    - Level 2 (instructions): Loaded on selection, <5k tokens
    - Level 3 (resources): Loaded on demand, unlimited
    """

    metadata: SkillMetadata
    _instructions: Optional[SkillInstructions] = field(default=None, repr=False)
    _resources: Optional[SkillResources] = field(default=None, repr=False)
    _skill_path: Optional[Path] = field(default=None, repr=False)

    @property
    def name(self) -> str:
        """Skill name for quick access."""
        return self.metadata.name

    @property
    def description(self) -> str:
        """Skill description for quick access."""
        return self.metadata.description

    @property
    def instructions_loaded(self) -> bool:
        """Check if instructions have been loaded."""
        return self._instructions is not None

    @property
    def resources_loaded(self) -> bool:
        """Check if resources have been loaded."""
        return self._resources is not None

    def get_instructions(self) -> SkillInstructions:
        """Get instructions, loading if necessary.

        Returns:
            Skill instructions

        Raises:
            RuntimeError: If instructions cannot be loaded
        """
        if self._instructions is None:
            self._load_instructions()
        if self._instructions is None:
            raise RuntimeError(f"Failed to load instructions for skill: {self.name}")
        return self._instructions

    def get_resources(self) -> SkillResources:
        """Get resources, loading if necessary.

        Returns:
            Skill resources

        Raises:
            RuntimeError: If resources cannot be loaded
        """
        if self._resources is None:
            self._load_resources()
        if self._resources is None:
            raise RuntimeError(f"Failed to load resources for skill: {self.name}")
        return self._resources

    def _load_instructions(self) -> None:
        """Load level 2 instructions from disk."""
        if self._skill_path is None:
            return

        skill_file = self._skill_path / "SKILL.md"
        if not skill_file.exists():
            return

        from marie.agent.skills.parser import parse_skill_file

        _, content = parse_skill_file(skill_file)
        self._instructions = SkillInstructions.from_markdown(content)

    def _load_resources(self) -> None:
        """Load level 3 resources from disk."""
        if self._skill_path is None:
            self._resources = SkillResources()
            return

        scripts: Dict[str, str] = {}
        templates: Dict[str, str] = {}
        references: Dict[str, str] = {}

        # Load scripts
        scripts_dir = self._skill_path / "scripts"
        if scripts_dir.exists():
            for script_file in scripts_dir.iterdir():
                if script_file.is_file():
                    scripts[script_file.name] = script_file.read_text()

        # Load templates/assets
        assets_dir = self._skill_path / "assets"
        if assets_dir.exists():
            for asset_file in assets_dir.iterdir():
                if asset_file.is_file():
                    templates[asset_file.name] = asset_file.read_text()

        # Load references (keyed by filename for path-addressable access)
        refs_dir = self._skill_path / "references"
        if refs_dir.exists():
            for ref_file in refs_dir.iterdir():
                if ref_file.is_file() and ref_file.suffix == ".md":
                    references[ref_file.name] = ref_file.read_text()

        self._resources = SkillResources(
            scripts=scripts,
            templates=templates,
            references=references,
        )

    def to_system_prompt_injection(self) -> str:
        """Generate system prompt content for this skill.

        Returns:
            Formatted string to inject into system prompt
        """
        instructions = self.get_instructions()

        parts = [
            f"## Active Skill: {self.metadata.name}",
            "",
            self.metadata.description,
            "",
        ]

        if instructions.when_to_use:
            parts.extend(["### When to Use", instructions.when_to_use, ""])

        if instructions.instructions:
            parts.extend(["### Instructions", instructions.instructions, ""])

        if self.metadata.allowed_tools:
            tools_str = ", ".join(self.metadata.allowed_tools)
            parts.extend(["### Available Tools", f"You can use: {tools_str}", ""])

        return "\n".join(parts)


@dataclass
class SkillContext:
    """Context for skill execution within an agent."""

    skill: Optional[Skill]
    message: str
    explicit_invocation: bool = False
    matched_score: float = 0.0

    @property
    def has_skill(self) -> bool:
        """Check if a skill was matched."""
        return self.skill is not None
