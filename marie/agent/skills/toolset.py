"""Agent-facing skill management tools.

Implements ADK progressive disclosure pattern:
- discover_skills: L1 metadata (~100 tokens each)
- load_skill: L2 instructions (<5k tokens)
- load_skill_resource: L3 resources (on demand)

Usage:
    # Register tools globally (on import)
    from marie.agent.skills import toolset

    # Or include in agent function_list
    agent = BaseAgent(function_list=["discover_skills", "load_skill", "load_skill_resource"])
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from marie.agent.skills.loader import get_skill_loader
from marie.agent.skills.registry import SKILL_REGISTRY, SkillNotFoundError
from marie.agent.tools.registry import register_tool
from marie.logging_core.logger import MarieLogger

logger = MarieLogger("marie.agent.skills.toolset")


# ============================================================================
# Input Schemas
# ============================================================================


class DiscoverSkillsInput(BaseModel):
    """Input schema for discover_skills tool."""

    tags: Optional[List[str]] = Field(
        default=None,
        description="Filter skills by tags (any match)",
    )
    query: Optional[str] = Field(
        default=None,
        description="Search query to match against skill names and descriptions using BM25",
    )
    limit: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum number of skills to return",
    )


class LoadSkillInput(BaseModel):
    """Input schema for load_skill tool."""

    name: str = Field(
        ...,
        description="Exact skill name to load",
    )
    include_full_content: bool = Field(
        default=False,
        description="Include raw markdown content in addition to parsed sections",
    )


class LoadSkillResourceInput(BaseModel):
    """Input schema for load_skill_resource tool."""

    skill_name: str = Field(
        ...,
        description="Name of the skill containing the resource",
    )
    resource_path: str = Field(
        ...,
        description="Path to the resource file (e.g., 'style-guide.md', 'template.py')",
    )
    resource_type: str = Field(
        default="references",
        description="Type of resource: 'references', 'scripts', or 'templates'",
    )


# ============================================================================
# Tool Implementations
# ============================================================================


@register_tool(
    "discover_skills",
    description=(
        "Discover skills relevant to the current task using BM25 search. "
        "Use this to identify which specialized knowledge and capabilities are "
        "available and relevant. Returns skill names, descriptions, and tags "
        "for routing decisions. Call load_skill() to get full instructions "
        "for a specific skill."
    ),
    fn_schema=DiscoverSkillsInput,
)
def discover_skills(
    tags: Optional[List[str]] = None,
    query: Optional[str] = None,
    limit: int = 20,
) -> str:
    """L1: Return lightweight metadata for relevant skills.

    Args:
        tags: Filter skills by tags (any match)
        query: Search query to match against skill names and descriptions
        limit: Maximum number of skills to return

    Returns:
        JSON array of skill metadata objects
    """
    if query:
        skills = SKILL_REGISTRY.search_skills(query, tags=tags, limit=limit)
    else:
        skills = SKILL_REGISTRY.list_skills(tags=tags)[:limit]

    result = [
        {
            "name": s.name,
            "description": s.description,
            "tags": s.metadata.tags,
            "user_invokable": s.metadata.user_invokable,
            "allowed_tools": s.metadata.allowed_tools,
        }
        for s in skills
    ]

    return json.dumps(result, indent=2)


@register_tool(
    "load_skill",
    description=(
        "Load full instructions for a skill. Call this when you need detailed "
        "guidance for a specific task. Returns structured instructions including "
        "when to use the skill, step-by-step instructions, and examples. "
        "Use discover_skills() first to discover available skills."
    ),
    fn_schema=LoadSkillInput,
)
def load_skill(
    name: str,
    include_full_content: bool = False,
) -> str:
    """L2: Load complete skill instructions.

    Args:
        name: Exact skill name to load
        include_full_content: Include raw markdown content

    Returns:
        JSON object with parsed skill instructions

    Raises:
        SkillNotFoundError: If skill does not exist
    """
    try:
        skill = SKILL_REGISTRY.get(name)
    except SkillNotFoundError:
        available = [s.name for s in SKILL_REGISTRY.list_skills()[:10]]
        return json.dumps(
            {
                "error": f"Skill '{name}' not found",
                "available_skills": available,
                "hint": "Use discover_skills() to see all available skills",
            }
        )

    # Ensure instructions are loaded
    loader = get_skill_loader()
    loader.load_skill_instructions(skill)

    instructions = skill.get_instructions()

    result: Dict[str, Any] = {
        "name": skill.name,
        "description": skill.description,
        "when_to_use": instructions.when_to_use,
        "instructions": instructions.instructions,
        "examples": [
            {"input": e.user_input, "action": e.expected_action}
            for e in instructions.examples
        ],
        "allowed_tools": skill.metadata.allowed_tools,
    }

    if include_full_content:
        result["full_content"] = instructions.full_content

    return json.dumps(result, indent=2)


@register_tool(
    "load_skill_resource",
    description=(
        "Load a specific resource file from a skill. Use this to access "
        "reference documentation, templates, or scripts that support skill "
        "execution. Resources are organized by type: 'references' (documentation), "
        "'scripts' (executable code), 'templates' (reusable templates)."
    ),
    fn_schema=LoadSkillResourceInput,
)
def load_skill_resource(
    skill_name: str,
    resource_path: str,
    resource_type: str = "references",
) -> str:
    """L3: Load specific resource file from a skill.

    Args:
        skill_name: Name of the skill containing the resource
        resource_path: Filename of the resource (e.g., 'style-guide.md')
        resource_type: Type of resource: 'references', 'scripts', or 'templates'

    Returns:
        Resource content as string, or error JSON if not found
    """
    try:
        skill = SKILL_REGISTRY.get(skill_name)
    except SkillNotFoundError:
        return json.dumps(
            {
                "error": f"Skill '{skill_name}' not found",
                "hint": "Use discover_skills() to see all available skills",
            }
        )

    # Ensure resources are loaded
    loader = get_skill_loader()
    loader.load_skill_resources(skill)

    resources = skill.get_resources()

    # Get the appropriate resource dict
    resource_dict: Dict[str, str]
    if resource_type == "references":
        resource_dict = resources.references
    elif resource_type == "scripts":
        resource_dict = resources.scripts
    elif resource_type == "templates":
        resource_dict = resources.templates
    else:
        return json.dumps(
            {
                "error": f"Invalid resource_type: {resource_type}",
                "valid_types": ["references", "scripts", "templates"],
            }
        )

    if resource_path not in resource_dict:
        available = list(resource_dict.keys())
        return json.dumps(
            {
                "error": f"Resource '{resource_path}' not found in {resource_type}",
                "available_resources": available,
                "hint": (
                    f"Available {resource_type}: {available}"
                    if available
                    else f"No {resource_type} in this skill"
                ),
            }
        )

    return resource_dict[resource_path]


# ============================================================================
# Toolset Class (Alternative API)
# ============================================================================


class SkillToolset:
    """Container for skill management tools.

    Provides an alternative API for agents that want explicit control
    over which skill tools are available.

    Example:
        toolset = SkillToolset()
        agent = BaseAgent(function_list=toolset.get_tool_names())
    """

    TOOL_NAMES = ["discover_skills", "load_skill", "load_skill_resource"]

    @classmethod
    def get_tool_names(cls) -> List[str]:
        """Return list of skill tool names for function_list."""
        return cls.TOOL_NAMES.copy()

    @classmethod
    def is_registered(cls) -> bool:
        """Check if skill tools are registered in the global registry."""
        from marie.agent.tools.registry import TOOL_REGISTRY

        return all(TOOL_REGISTRY.has(name) for name in cls.TOOL_NAMES)
