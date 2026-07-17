"""Skill registry for discovery and management.

This module provides thread-safe skill discovery, registration,
and lookup functionality.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Callable, Dict, List, Optional, TypeVar

from marie.agent.skills.models import Skill, SkillMetadata, SkillSource
from marie.agent.skills.parser import parse_skill, validate_skill_structure

logger = logging.getLogger("marie.agent.skills.registry")

T = TypeVar("T", bound=Skill)


class SkillNotFoundError(Exception):
    """Raised when a skill is not found in the registry."""

    def __init__(self, skill_name: str):
        self.skill_name = skill_name
        super().__init__(f"Skill not found: {skill_name}")


class SkillRegistry:
    """Thread-safe skill discovery and registration.

    The registry manages skills from multiple sources:
    - Built-in skills from config/skills/
    - Workspace skills from <workspace>/.marie/skills/
    - Programmatically registered skills

    Example:
        ```python
        registry = SkillRegistry()

        # Discover skills
        registry.discover_skills([Path("config/skills")])

        # Get a skill
        skill = registry.get("document-extraction")

        # List all skills
        for skill in registry.list_skills():
            print(f"{skill.name}: {skill.description}")
        ```
    """

    def __init__(self):
        """Initialize empty registry."""
        self._skills: Dict[str, Skill] = {}
        self._lock = threading.RLock()
        self._discovery_paths: List[Path] = []

    def discover_skills(
        self,
        paths: List[Path],
        source: SkillSource = SkillSource.BUILTIN,
        clear_existing: bool = False,
    ) -> int:
        """Scan directories for SKILL.md files and register them.

        Args:
            paths: List of directories to scan
            source: Source type for discovered skills
            clear_existing: If True, clear registry before discovering

        Returns:
            Number of skills discovered
        """
        with self._lock:
            if clear_existing:
                self._skills.clear()

            count = 0
            for base_path in paths:
                base_path = Path(base_path)
                if not base_path.exists():
                    logger.debug(f"Skill path does not exist: {base_path}")
                    continue

                self._discovery_paths.append(base_path)

                # Scan for skill directories
                for item in base_path.iterdir():
                    if not item.is_dir():
                        continue

                    if validate_skill_structure(item):
                        try:
                            skill = parse_skill(item, source=source)
                            self._register_internal(skill)
                            count += 1
                            logger.debug(f"Discovered skill: {skill.name} from {item}")
                        except Exception as e:
                            logger.warning(f"Failed to load skill from {item}: {e}")

            logger.info(f"Discovered {count} skills from {len(paths)} paths")
            return count

    def register_skill(self, skill: Skill) -> None:
        """Register a skill by name.

        Args:
            skill: Skill to register

        Raises:
            ValueError: If skill with same name already exists
        """
        with self._lock:
            if skill.name in self._skills:
                logger.warning(f"Overwriting existing skill: {skill.name}")
            self._register_internal(skill)

    def _register_internal(self, skill: Skill) -> None:
        """Internal registration without locking."""
        self._skills[skill.name] = skill

    def unregister_skill(self, name: str) -> bool:
        """Remove a skill from the registry.

        Args:
            name: Skill name to remove

        Returns:
            True if skill was removed, False if not found
        """
        with self._lock:
            if name in self._skills:
                del self._skills[name]
                return True
            return False

    def get(self, name: str) -> Skill:
        """Get skill by exact name.

        Args:
            name: Skill name

        Returns:
            Skill instance

        Raises:
            SkillNotFoundError: If skill not found
        """
        with self._lock:
            skill = self._skills.get(name)
            if skill is None:
                raise SkillNotFoundError(name)
            return skill

    def get_or_none(self, name: str) -> Optional[Skill]:
        """Get skill by name, returning None if not found.

        Args:
            name: Skill name

        Returns:
            Skill instance or None
        """
        with self._lock:
            return self._skills.get(name)

    def has(self, name: str) -> bool:
        """Check if skill exists in registry.

        Args:
            name: Skill name

        Returns:
            True if skill exists
        """
        with self._lock:
            return name in self._skills

    def list_skills(
        self,
        source: Optional[SkillSource] = None,
        tags: Optional[List[str]] = None,
        user_invokable_only: bool = False,
    ) -> List[Skill]:
        """List all registered skills.

        Args:
            source: Filter by source type
            tags: Filter by tags (any match)
            user_invokable_only: Only return user-invokable skills

        Returns:
            List of matching skills
        """
        with self._lock:
            skills = list(self._skills.values())

            if source is not None:
                skills = [s for s in skills if s.metadata.source == source]

            if tags:
                tag_set = set(tags)
                skills = [s for s in skills if tag_set & set(s.metadata.tags)]

            if user_invokable_only:
                skills = [s for s in skills if s.metadata.user_invokable]

            return skills

    def list_metadata(self) -> List[SkillMetadata]:
        """List metadata for all skills.

        Returns:
            List of SkillMetadata (lightweight, no instructions loaded)
        """
        with self._lock:
            return [skill.metadata for skill in self._skills.values()]

    def list_user_invokable(self) -> List[SkillMetadata]:
        """List skills available as slash commands.

        Returns:
            List of user-invokable SkillMetadata
        """
        with self._lock:
            return [
                skill.metadata
                for skill in self._skills.values()
                if skill.metadata.user_invokable
            ]

    def search_skills(
        self,
        query: str,
        tags: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[Skill]:
        """Search skills by description/tags.

        Args:
            query: Search query
            tags: Optional tag filter
            limit: Maximum results to return

        Returns:
            List of matching skills, sorted by relevance
        """
        with self._lock:
            candidates = self.list_skills(tags=tags)

            # Score and sort by relevance
            scored = [
                (skill, skill.metadata.matches_query(query)) for skill in candidates
            ]
            scored = [(s, score) for s, score in scored if score > 0]
            scored.sort(key=lambda x: x[1], reverse=True)

            return [s for s, _ in scored[:limit]]

    def clear(self) -> None:
        """Clear all registered skills."""
        with self._lock:
            self._skills.clear()
            self._discovery_paths.clear()

    def clear_source(self, source: SkillSource) -> int:
        """Remove all skills from a specific source.

        Useful for refreshing workspace skills without clearing built-in skills.

        Args:
            source: Source type to clear (e.g., SkillSource.WORKSPACE)

        Returns:
            Number of skills removed
        """
        with self._lock:
            to_remove = [
                name
                for name, skill in self._skills.items()
                if skill.metadata.source == source
            ]
            for name in to_remove:
                del self._skills[name]
            logger.debug(f"Cleared {len(to_remove)} skills from source: {source.value}")
            return len(to_remove)

    def __len__(self) -> int:
        """Return number of registered skills."""
        with self._lock:
            return len(self._skills)

    def __contains__(self, name: str) -> bool:
        """Check if skill exists."""
        return self.has(name)

    def __iter__(self):
        """Iterate over skills."""
        with self._lock:
            return iter(list(self._skills.values()))


# Global skill registry instance
SKILL_REGISTRY = SkillRegistry()


def register_skill(
    name: Optional[str] = None,
    description: Optional[str] = None,
    **metadata_kwargs,
) -> Callable[[type], type]:
    """Decorator for programmatic skill registration.

    Can be used with or without arguments.

    Example:
        ```python
        @register_skill
        class MySkill(Skill): ...


        @register_skill(name="custom-name", tags=["utility"])
        class AnotherSkill(Skill): ...
        ```

    Args:
        name: Optional skill name (uses class name if not provided)
        description: Optional description
        **metadata_kwargs: Additional metadata fields

    Returns:
        Decorator function
    """

    def decorator(cls: type) -> type:
        skill_name = name or cls.__name__.lower().replace("_", "-")
        skill_desc = description or cls.__doc__ or f"Skill: {skill_name}"

        metadata = SkillMetadata(
            name=skill_name,
            description=skill_desc,
            source=SkillSource.USER,
            **metadata_kwargs,
        )

        skill = Skill(metadata=metadata)

        SKILL_REGISTRY.register_skill(skill)
        return cls

    # Handle both @register_skill and @register_skill()
    if callable(name):
        # Called without parentheses: @register_skill
        cls = name
        name = None
        return decorator(cls)

    return decorator


def get_skill(name: str) -> Skill:
    """Get skill from global registry.

    Args:
        name: Skill name

    Returns:
        Skill instance

    Raises:
        SkillNotFoundError: If skill not found
    """
    return SKILL_REGISTRY.get(name)


def list_skills(**kwargs) -> List[Skill]:
    """List skills from global registry.

    Args:
        **kwargs: Passed to SkillRegistry.list_skills()

    Returns:
        List of skills
    """
    return SKILL_REGISTRY.list_skills(**kwargs)
