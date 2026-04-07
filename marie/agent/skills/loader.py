"""Progressive skill loader with 3-tier architecture.

This module implements the progressive loading pattern:
- Level 1: Metadata (~100 tokens) - Always loaded
- Level 2: Instructions (<5k tokens) - Loaded on selection
- Level 3: Resources (unlimited) - Loaded on demand
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from marie.agent.skills.models import Skill, SkillSource
from marie.agent.skills.parser import parse_skill
from marie.agent.skills.registry import SKILL_REGISTRY, SkillRegistry
from marie.logging_core.logger import MarieLogger

logger = MarieLogger("marie.agent.skills.loader")

# Default skill paths
DEFAULT_BUILTIN_PATH = "config/skills"
WORKSPACE_SKILL_PATH = ".marie/skills"


class SkillLoader:
    """Progressive skill loader.

    Handles skill discovery and loading with lazy evaluation
    of instructions and resources.

    Example:
        ```python
        loader = SkillLoader()

        # Discover built-in skills
        loader.discover_builtin_skills()

        # Discover workspace skills
        loader.discover_workspace_skills("/path/to/workspace")

        # Get a skill (metadata only loaded)
        skill = loader.get_skill("document-extraction")

        # Load instructions when needed
        instructions = skill.get_instructions()

        # Load resources when needed
        resources = skill.get_resources()
        ```
    """

    def __init__(
        self,
        registry: Optional[SkillRegistry] = None,
        builtin_path: Optional[str] = None,
    ):
        """Initialize loader.

        Args:
            registry: Skill registry to populate
            builtin_path: Path to built-in skills directory
        """
        self.registry = registry or SKILL_REGISTRY
        self.builtin_path = Path(builtin_path or DEFAULT_BUILTIN_PATH)
        self._initialized = False

    def discover_builtin_skills(self) -> int:
        """Discover and register built-in skills.

        Returns:
            Number of skills discovered
        """
        paths = self._resolve_builtin_paths()
        count = self.registry.discover_skills(paths, source=SkillSource.BUILTIN)
        logger.info(f"Loaded {count} built-in skills")
        return count

    def discover_workspace_skills(self, workspace_path: str) -> int:
        """Discover workspace-specific skills.

        Args:
            workspace_path: Path to workspace root

        Returns:
            Number of skills discovered
        """
        workspace_skills_path = Path(workspace_path) / WORKSPACE_SKILL_PATH

        if not workspace_skills_path.exists():
            logger.debug(f"No workspace skills at: {workspace_skills_path}")
            return 0

        count = self.registry.discover_skills(
            [workspace_skills_path],
            source=SkillSource.WORKSPACE,
        )
        logger.info(f"Loaded {count} workspace skills from {workspace_path}")
        return count

    def discover_all(self, workspace_path: Optional[str] = None) -> int:
        """Discover all skills from built-in and workspace paths.

        Args:
            workspace_path: Optional workspace path

        Returns:
            Total number of skills discovered
        """
        count = self.discover_builtin_skills()

        if workspace_path:
            count += self.discover_workspace_skills(workspace_path)

        self._initialized = True
        return count

    def _resolve_builtin_paths(self) -> List[Path]:
        """Resolve built-in skill paths.

        Checks multiple possible locations for built-in skills.

        Returns:
            List of existing paths
        """
        candidates = [
            self.builtin_path,
            Path(__file__).parent.parent.parent.parent / "config" / "skills",
            Path(os.getcwd()) / "config" / "skills",
        ]

        # Also check MARIE_SKILLS_PATH environment variable
        env_path = os.getenv("MARIE_SKILLS_PATH")
        if env_path:
            candidates.insert(0, Path(env_path))

        return [p for p in candidates if p.exists()]

    def get_skill(self, name: str) -> Skill:
        """Get skill from registry.

        Args:
            name: Skill name

        Returns:
            Skill instance
        """
        return self.registry.get(name)

    def load_skill_instructions(self, skill: Skill) -> None:
        """Eagerly load skill instructions (Level 2).

        Args:
            skill: Skill to load instructions for
        """
        if not skill.instructions_loaded:
            _ = skill.get_instructions()

    def load_skill_resources(self, skill: Skill) -> None:
        """Eagerly load skill resources (Level 3).

        Args:
            skill: Skill to load resources for
        """
        if not skill.resources_loaded:
            _ = skill.get_resources()

    def preload_skill(self, name: str, level: int = 2) -> Skill:
        """Preload a skill to specified level.

        Args:
            name: Skill name
            level: Loading level (1=metadata, 2=instructions, 3=resources)

        Returns:
            Loaded skill
        """
        skill = self.get_skill(name)

        if level >= 2:
            self.load_skill_instructions(skill)

        if level >= 3:
            self.load_skill_resources(skill)

        return skill

    def is_initialized(self) -> bool:
        """Check if loader has been initialized."""
        return self._initialized

    def refresh_workspace_skills(self, workspace_path: str) -> int:
        """Rediscover workspace skills without clearing built-in skills.

        This method clears only workspace skills from the registry and
        rediscovers them from disk, allowing runtime updates to workspace
        skills without restarting the agent.

        Args:
            workspace_path: Path to workspace root

        Returns:
            Number of workspace skills discovered after refresh
        """
        # Clear only workspace skills from registry
        removed = self.registry.clear_source(SkillSource.WORKSPACE)
        logger.info(f"Cleared {removed} workspace skills for refresh")

        # Rediscover from workspace path
        workspace_skills_path = Path(workspace_path) / WORKSPACE_SKILL_PATH
        if workspace_skills_path.exists():
            count = self.registry.discover_skills(
                [workspace_skills_path],
                source=SkillSource.WORKSPACE,
            )
            logger.info(
                f"Discovered {count} workspace skills from {workspace_skills_path}"
            )
            return count

        logger.warning(f"Workspace skills path does not exist: {workspace_skills_path}")
        return 0


# Global loader instance
_default_loader: Optional[SkillLoader] = None


def get_skill_loader() -> SkillLoader:
    """Get or create the default skill loader.

    Returns:
        SkillLoader instance
    """
    global _default_loader
    if _default_loader is None:
        _default_loader = SkillLoader()
    return _default_loader


def initialize_skills(workspace_path: Optional[str] = None) -> int:
    """Initialize the skill system.

    Call this during application startup to discover and register skills.

    Args:
        workspace_path: Optional workspace path for workspace-specific skills

    Returns:
        Number of skills loaded
    """
    loader = get_skill_loader()

    if loader.is_initialized():
        logger.debug("Skills already initialized")
        return len(SKILL_REGISTRY)

    return loader.discover_all(workspace_path)
