"""Agent execution utilities for coordination layer.

Provides async wrappers for executing agents that may have sync iterator-based
run() methods or async arun() methods.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any, Dict, List

from marie.agent.backends.base import AgentStatus
from marie.agent.coordination.topology import AgentResult

if TYPE_CHECKING:
    from marie.agent.base import BaseAgent

logger = logging.getLogger("marie.agent.coordination.execution")


async def execute_agent_async(
    agent: "BaseAgent",
    messages: List[Dict[str, Any]],
    **kwargs: Any,
) -> AgentResult:
    """Execute an agent and collect output into AgentResult.

    Handles both sync iterator-based run() and async arun() methods.

    Args:
        agent: Agent to execute
        messages: Input messages
        **kwargs: Additional arguments passed to agent

    Returns:
        AgentResult with collected output
    """
    start_time = time.perf_counter()
    agent_name = getattr(agent, "name", None) or "unnamed_agent"

    all_messages: List[Dict[str, Any]] = []
    final_output = ""
    metadata: Dict[str, Any] = {}
    error = None
    status = AgentStatus.COMPLETED

    try:
        # Try async method first
        if hasattr(agent, "arun"):
            result = await agent.arun(messages, **kwargs)
            if isinstance(result, dict):
                final_output = result.get("output", "")
                all_messages = result.get("messages", [])
                metadata = result.get("metadata", {})
            else:
                final_output = result
        # Fall back to sync iterator method
        elif hasattr(agent, "run"):
            # Run sync iterator in executor to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, lambda: _collect_iterator_output(agent, messages, **kwargs)
            )
            final_output = result["output"]
            all_messages = result["messages"]
            metadata = result.get("metadata", {})
        else:
            raise ValueError(f"Agent {agent_name} has no run or arun method")

    except Exception as e:
        status = AgentStatus.FAILED
        error = str(e)
        logger.error(f"Agent {agent_name} execution failed: {e}")

    duration_ms = (time.perf_counter() - start_time) * 1000

    return AgentResult(
        agent_name=agent_name,
        output=final_output,
        messages=all_messages,
        status=status.value,
        error=error,
        duration_ms=duration_ms,
        metadata=metadata,
    )


async def execute_agent_with_timeout(
    agent: "BaseAgent",
    messages: List[Dict[str, Any]],
    timeout: float,
    **kwargs: Any,
) -> AgentResult:
    """Execute agent with timeout.

    Args:
        agent: Agent to execute
        messages: Input messages
        timeout: Timeout in seconds
        **kwargs: Additional arguments

    Returns:
        AgentResult (may have status="timeout" on timeout)
    """
    agent_name = getattr(agent, "name", None) or "unnamed_agent"
    start_time = time.perf_counter()

    try:
        return await asyncio.wait_for(
            execute_agent_async(agent, messages, **kwargs),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.warning(f"Agent {agent_name} timed out after {timeout}s")
        return AgentResult(
            agent_name=agent_name,
            output=None,
            messages=[],
            status="timeout",
            error=f"Timeout after {timeout}s",
            duration_ms=duration_ms,
        )


def _collect_iterator_output(
    agent: "BaseAgent",
    messages: List[Dict[str, Any]],
    **kwargs: Any,
) -> Dict[str, Any]:
    """Collect output from sync iterator-based agent.run().

    This runs in a thread executor to avoid blocking the event loop.
    """
    all_messages = []
    final_output = ""

    try:
        for chunk in agent.run(messages, **kwargs):
            if chunk:
                all_messages.extend(chunk)

        # Extract final output from last assistant message
        for msg in reversed(all_messages):
            if hasattr(msg, "role") and msg.role == "assistant":
                if hasattr(msg, "text_content"):
                    final_output = msg.text_content
                elif hasattr(msg, "content"):
                    final_output = (
                        msg.content
                        if isinstance(msg.content, str)
                        else str(msg.content)
                    )
                break
            elif isinstance(msg, dict) and msg.get("role") == "assistant":
                final_output = msg.get("content", "")
                break

    except Exception as e:
        raise e

    # Convert Message objects to dicts
    msg_dicts = []
    for msg in all_messages:
        if hasattr(msg, "model_dump"):
            msg_dicts.append(msg.model_dump())
        elif hasattr(msg, "dict"):
            msg_dicts.append(msg.dict())
        elif isinstance(msg, dict):
            msg_dicts.append(msg)
        else:
            msg_dicts.append({"content": str(msg)})

    return {"output": final_output, "messages": msg_dicts}
