"""Agent coordination module for multi-agent orchestration.

This module provides coordination patterns for running multiple agents
together, including parallel (fan-out) and sequential (chain) topologies.
"""

from marie.agent.coordination.chain import ChainCoordinator
from marie.agent.coordination.config import CoordinationConfig, MergeStrategy, Topology
from marie.agent.coordination.fan_out import FanOutCoordinator
from marie.agent.coordination.topology import AgentResult as CoordinationResult
from marie.agent.coordination.topology import (
    BaseCoordinator,
    CoordinatorFactory,
)

__all__ = [
    "CoordinationConfig",
    "Topology",
    "MergeStrategy",
    "BaseCoordinator",
    "FanOutCoordinator",
    "ChainCoordinator",
    "CoordinatorFactory",
    "CoordinationResult",
]
