"""Configuration models for agent coordination.

The main CoordinationConfig is defined in marie.agent.config and re-exported
here for convenience. This module adds coordination-specific enums.
"""

from __future__ import annotations

from enum import Enum

from marie.agent.config import CoordinationConfig


class Topology(str, Enum):
    """Coordination topology for multi-agent execution."""

    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"


class MergeStrategy(str, Enum):
    """Strategy for merging results from multiple agents."""

    AGGREGATE = "aggregate"
    VOTE = "vote"
    FIRST_WINS = "first_wins"
    BEST_SCORE = "best_score"


def get_topology_enum(config: CoordinationConfig) -> Topology:
    """Get topology as enum value from config."""
    return Topology(config.topology)


def get_merge_strategy_enum(config: CoordinationConfig) -> MergeStrategy:
    """Get merge strategy as enum value from config."""
    return MergeStrategy(config.merge_strategy)


__all__ = [
    "CoordinationConfig",
    "Topology",
    "MergeStrategy",
    "get_topology_enum",
    "get_merge_strategy_enum",
]
