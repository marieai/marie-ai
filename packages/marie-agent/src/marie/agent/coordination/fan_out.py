"""Fan-out coordinator for parallel agent execution."""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List

from marie.agent.coordination.topology import (
    AgentResult,
    BaseCoordinator,
    CoordinationResult,
)

if TYPE_CHECKING:
    from marie.agent.base import BaseAgent
    from marie.agent.coordination.config import CoordinationConfig
    from marie.agent.message import Message

logger = logging.getLogger("marie.agent.coordination.fan_out")


class FanOutCoordinator(BaseCoordinator):
    """Coordinator for parallel (fan-out) agent execution.

    Executes multiple agents concurrently using asyncio.gather,
    respecting max_concurrent limits via semaphore. Results are
    collected and merged according to the configured strategy.

    Aligned with claude-flow parallel coordination pattern.
    """

    def __init__(self, config: "CoordinationConfig"):
        super().__init__(config)
        self._semaphore: asyncio.Semaphore | None = None

    async def run(
        self,
        messages: List["Message"],
        **kwargs: Any,
    ) -> CoordinationResult:
        """Execute all agents in parallel.

        Args:
            messages: Input messages to process
            **kwargs: Additional arguments passed to agents

        Returns:
            CoordinationResult with merged outputs from all agents
        """
        if not self._agents:
            return CoordinationResult(
                results=[],
                merged_output=None,
                topology="parallel",
                merge_strategy=self.config.merge_strategy,
                total_duration_ms=0.0,
            )

        start_time = time.perf_counter()
        started_at = datetime.now(timezone.utc)

        self._semaphore = asyncio.Semaphore(self.config.max_concurrent)

        tasks = [
            self._run_agent_with_semaphore(agent, messages, **kwargs)
            for agent in self._agents
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        agent_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                agent_results.append(
                    AgentResult(
                        agent_name=self._agents[i].name or f"agent_{i}",
                        output=None,
                        status="failed",
                        error=str(result),
                    )
                )
            else:
                agent_results.append(result)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        merged = self._merge_results(agent_results)

        return CoordinationResult(
            results=agent_results,
            merged_output=merged,
            topology="parallel",
            merge_strategy=self.config.merge_strategy,
            total_duration_ms=elapsed_ms,
            started_at=started_at,
            completed_at=datetime.now(timezone.utc),
        )

    async def _run_agent_with_semaphore(
        self,
        agent: "BaseAgent",
        messages: List["Message"],
        **kwargs: Any,
    ) -> AgentResult:
        """Run a single agent with semaphore-based concurrency control."""
        async with self._semaphore:
            return await self._run_single_agent(agent, messages, **kwargs)

    async def _run_single_agent(
        self,
        agent: "BaseAgent",
        messages: List["Message"],
        **kwargs: Any,
    ) -> AgentResult:
        """Execute a single agent and capture its result."""
        start_time = time.perf_counter()
        agent_name = agent.name or "unnamed_agent"

        try:
            result = await asyncio.wait_for(
                self._execute_agent(agent, messages, **kwargs),
                timeout=self.config.timeout,
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            return AgentResult(
                agent_name=agent_name,
                output=result.get("output") if isinstance(result, dict) else result,
                messages=result.get("messages", []) if isinstance(result, dict) else [],
                status="completed",
                duration_ms=elapsed_ms,
                metadata=result.get("metadata", {}) if isinstance(result, dict) else {},
            )

        except asyncio.TimeoutError:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.warning(f"Agent {agent_name} timed out after {self.config.timeout}s")
            return AgentResult(
                agent_name=agent_name,
                output=None,
                status="timeout",
                error=f"Timeout after {self.config.timeout}s",
                duration_ms=elapsed_ms,
            )

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"Agent {agent_name} failed: {e}")
            return AgentResult(
                agent_name=agent_name,
                output=None,
                status="failed",
                error=str(e),
                duration_ms=elapsed_ms,
            )

    async def _execute_agent(
        self,
        agent: "BaseAgent",
        messages: List["Message"],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Execute the agent's run method.

        Handles both sync and async agent implementations.
        """
        if hasattr(agent, "arun"):
            result = await agent.arun(messages, **kwargs)
        elif hasattr(agent, "run"):
            result = agent.run(messages, **kwargs)
            if asyncio.iscoroutine(result):
                result = await result
        else:
            raise ValueError(f"Agent {agent.name} has no run or arun method")

        if isinstance(result, dict):
            return result
        return {"output": result, "messages": [], "metadata": {}}
