"""Base coordinator and factory for multi-agent orchestration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type

from marie.logging_core.logger import MarieLogger

if TYPE_CHECKING:
    from marie.agent.base import BaseAgent
    from marie.agent.coordination.config import CoordinationConfig
    from marie.agent.coordination.state import AgentWorkflowState
    from marie.agent.message import Message

logger = MarieLogger("marie.agent.coordination")


@dataclass
class AgentResult:
    """Result from a single agent execution within coordination."""

    agent_name: str
    output: Any
    messages: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "completed"
    error: Optional[str] = None
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        return self.status == "completed" and self.error is None


@dataclass
class CoordinationResult:
    """Result from coordinated multi-agent execution."""

    results: List[AgentResult]
    merged_output: Any
    topology: str
    merge_strategy: str
    total_duration_ms: float
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    workflow_state: Optional["AgentWorkflowState"] = None

    @property
    def success_count(self) -> int:
        return sum(1 for r in self.results if r.is_success)

    @property
    def failure_count(self) -> int:
        return len(self.results) - self.success_count

    @property
    def all_succeeded(self) -> bool:
        return self.failure_count == 0


class BaseCoordinator(ABC):
    """Base class for multi-agent coordinators.

    Coordinators manage the execution of multiple agents according to
    a specific topology (parallel, sequential, etc.) and merge their
    results using a configured strategy.
    """

    def __init__(self, config: "CoordinationConfig"):
        self.config = config
        self._agents: List["BaseAgent"] = []
        self._group_memory: Optional[Any] = None

    def add_agent(self, agent: "BaseAgent") -> None:
        """Add an agent to the coordination group."""
        self._agents.append(agent)

    def add_agents(self, agents: List["BaseAgent"]) -> None:
        """Add multiple agents to the coordination group."""
        self._agents.extend(agents)

    def clear_agents(self) -> None:
        """Remove all agents from the coordination group."""
        self._agents.clear()

    @property
    def agents(self) -> List["BaseAgent"]:
        return self._agents

    @abstractmethod
    async def run(
        self,
        messages: List["Message"],
        **kwargs: Any,
    ) -> CoordinationResult:
        """Execute all agents according to the coordination topology.

        Args:
            messages: Input messages to process
            **kwargs: Additional arguments passed to agents

        Returns:
            CoordinationResult with merged outputs
        """
        pass

    def _merge_results(self, results: List[AgentResult]) -> Any:
        """Merge results according to the configured strategy."""
        strategy = self.config.merge_strategy
        successful = [r for r in results if r.is_success]

        if not successful:
            return None

        if strategy == "aggregate":
            return self._merge_aggregate(successful)
        elif strategy == "vote":
            return self._merge_vote(successful)
        elif strategy == "first_wins":
            return self._merge_first_wins(successful)
        elif strategy == "best_score":
            return self._merge_best_score(successful)
        else:
            return self._merge_aggregate(successful)

    def _merge_aggregate(self, results: List[AgentResult]) -> Dict[str, Any]:
        """Aggregate all outputs into a combined result."""
        return {
            "outputs": [r.output for r in results],
            "agents": [r.agent_name for r in results],
        }

    def _merge_vote(self, results: List[AgentResult]) -> Any:
        """Use majority voting to select the result."""
        from collections import Counter

        outputs = [str(r.output) for r in results]
        most_common = Counter(outputs).most_common(1)
        if most_common:
            winning_output = most_common[0][0]
            for r in results:
                if str(r.output) == winning_output:
                    return r.output
        return results[0].output if results else None

    def _merge_first_wins(self, results: List[AgentResult]) -> Any:
        """Return the first successful result."""
        return results[0].output if results else None

    def _merge_best_score(self, results: List[AgentResult]) -> Any:
        """Return the result with the highest confidence score."""
        best = max(
            results,
            key=lambda r: r.metadata.get("confidence", 0.0),
            default=None,
        )
        return best.output if best else None


class CoordinatorFactory:
    """Factory for creating coordinators based on configuration."""

    _registry: Dict[str, Type[BaseCoordinator]] = {}

    @classmethod
    def register(cls, topology: str, coordinator_cls: Type[BaseCoordinator]) -> None:
        """Register a coordinator class for a topology."""
        cls._registry[topology] = coordinator_cls

    @classmethod
    def create(cls, config: "CoordinationConfig") -> BaseCoordinator:
        """Create a coordinator based on configuration.

        Args:
            config: Coordination configuration

        Returns:
            Appropriate coordinator instance

        Raises:
            ValueError: If topology is not registered
        """
        topology = config.topology
        if topology not in cls._registry:
            from marie.agent.coordination.chain import ChainCoordinator
            from marie.agent.coordination.fan_out import FanOutCoordinator

            cls._registry["parallel"] = FanOutCoordinator
            cls._registry["sequential"] = ChainCoordinator

        coordinator_cls = cls._registry.get(topology)
        if not coordinator_cls:
            raise ValueError(f"Unknown topology: {topology}")

        return coordinator_cls(config)
