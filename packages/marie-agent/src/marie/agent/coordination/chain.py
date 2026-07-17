"""Chain coordinator for sequential agent execution."""

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

logger = logging.getLogger("marie.agent.coordination.chain")


class ChainCoordinator(BaseCoordinator):
    """Coordinator for sequential (chain) agent execution.

    Executes agents one after another, passing the output of each
    agent as context to the next. Useful for pipelines where each
    agent builds on the previous agent's work.

    Aligned with claude-flow sequential coordination pattern.
    """

    def __init__(self, config: "CoordinationConfig"):
        super().__init__(config)
        self._stop_on_failure: bool = True

    async def run(
        self,
        messages: List["Message"],
        **kwargs: Any,
    ) -> CoordinationResult:
        """Execute agents sequentially in chain order.

        Args:
            messages: Input messages to process
            **kwargs: Additional arguments passed to agents

        Returns:
            CoordinationResult with final output from the chain
        """
        if not self._agents:
            return CoordinationResult(
                results=[],
                merged_output=None,
                topology="sequential",
                merge_strategy=self.config.merge_strategy,
                total_duration_ms=0.0,
            )

        start_time = time.perf_counter()
        started_at = datetime.now(timezone.utc)

        agent_results: List[AgentResult] = []
        current_context = self._build_initial_context(messages)

        for i, agent in enumerate(self._agents):
            result = await self._run_single_agent(
                agent,
                current_context,
                previous_results=agent_results,
                **kwargs,
            )
            agent_results.append(result)

            if not result.is_success and self._stop_on_failure:
                logger.warning(
                    f"Chain stopped at agent {agent.name} due to failure: {result.error}"
                )
                break

            if result.is_success and i < len(self._agents) - 1:
                current_context = self._update_context_for_next(current_context, result)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        merged = self._merge_results(agent_results)

        return CoordinationResult(
            results=agent_results,
            merged_output=merged,
            topology="sequential",
            merge_strategy=self.config.merge_strategy,
            total_duration_ms=elapsed_ms,
            started_at=started_at,
            completed_at=datetime.now(timezone.utc),
        )

    async def _run_single_agent(
        self,
        agent: "BaseAgent",
        messages: List["Message"],
        previous_results: List[AgentResult],
        **kwargs: Any,
    ) -> AgentResult:
        """Execute a single agent in the chain."""
        start_time = time.perf_counter()
        agent_name = agent.name or "unnamed_agent"

        try:
            result = await asyncio.wait_for(
                self._execute_agent(agent, messages, previous_results, **kwargs),
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
        previous_results: List[AgentResult],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Execute the agent with chain context."""
        chain_context = {
            "chain_position": len(previous_results),
            "previous_outputs": [r.output for r in previous_results if r.is_success],
        }
        kwargs["chain_context"] = chain_context

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

    def _build_initial_context(
        self,
        messages: List["Message"],
    ) -> List["Message"]:
        """Build initial context from input messages."""
        return list(messages)

    def _update_context_for_next(
        self,
        current_context: List["Message"],
        result: AgentResult,
    ) -> List["Message"]:
        """Update context with previous agent's output for the next agent."""
        from marie.agent.message import Message

        updated = list(current_context)

        if result.output:
            output_str = (
                str(result.output)
                if not isinstance(result.output, str)
                else result.output
            )
            updated.append(
                Message(
                    role="assistant",
                    content=f"[{result.agent_name}]: {output_str}",
                    name=result.agent_name,
                )
            )

        return updated

    def set_stop_on_failure(self, stop: bool) -> None:
        """Configure whether chain stops on agent failure."""
        self._stop_on_failure = stop
