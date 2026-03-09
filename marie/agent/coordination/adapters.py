"""Adapters for converting between coordination and backend result types."""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from marie.agent.coordination.topology import AgentResult as CoordAgentResult
    from marie.agent.coordination.topology import CoordinationResult

from marie.agent.backends.base import AgentResult, AgentStatus, ToolCallRecord


def coordination_result_to_agent_result(
    coord_result: "CoordinationResult",
) -> AgentResult:
    """Convert CoordinationResult (dataclass) to AgentResult (Pydantic) for API compatibility.

    Args:
        coord_result: Result from coordinator execution

    Returns:
        AgentResult compatible with backend API
    """
    return AgentResult(
        output=coord_result.merged_output,
        messages=_flatten_messages(coord_result.results),
        tool_calls=_collect_tool_calls(coord_result.results),
        status=_map_coordination_status(coord_result),
        error=_collect_errors(coord_result) if not coord_result.all_succeeded else None,
        iterations=len(coord_result.results),
        is_complete=True,
        metadata={
            "coordination": {
                "topology": coord_result.topology,
                "merge_strategy": coord_result.merge_strategy,
                "total_duration_ms": coord_result.total_duration_ms,
                "success_count": coord_result.success_count,
                "failure_count": coord_result.failure_count,
                "started_at": (
                    coord_result.started_at.isoformat()
                    if coord_result.started_at
                    else None
                ),
                "completed_at": (
                    coord_result.completed_at.isoformat()
                    if coord_result.completed_at
                    else None
                ),
                "agent_results": [asdict(r) for r in coord_result.results],
            }
        },
    )


def _map_coordination_status(coord_result: "CoordinationResult") -> AgentStatus:
    """Map coordination outcome to AgentStatus.

    Uses COMPLETED + metadata flag for partial success (avoids adding new enum).
    """
    if coord_result.all_succeeded:
        return AgentStatus.COMPLETED
    elif coord_result.success_count == 0:
        return AgentStatus.FAILED
    else:
        # Partial success: return COMPLETED, caller checks metadata.coordination.failure_count
        return AgentStatus.COMPLETED


def _flatten_messages(results: List["CoordAgentResult"]) -> List[Dict[str, Any]]:
    """Flatten messages from all agent results."""
    messages = []
    for r in results:
        if r.messages:
            messages.extend(r.messages)
    return messages


def _collect_tool_calls(results: List["CoordAgentResult"]) -> List[ToolCallRecord]:
    """Collect tool calls from all agent results."""
    calls = []
    for r in results:
        if r.metadata and "tool_calls" in r.metadata:
            for tc in r.metadata["tool_calls"]:
                if isinstance(tc, dict):
                    calls.append(ToolCallRecord(**tc))
                elif isinstance(tc, ToolCallRecord):
                    calls.append(tc)
    return calls


def _collect_errors(coord_result: "CoordinationResult") -> str:
    """Collect error messages from failed agent results."""
    errors = []
    for r in coord_result.results:
        if r.error:
            errors.append(f"{r.agent_name}: {r.error}")
    return "; ".join(errors) if errors else None
