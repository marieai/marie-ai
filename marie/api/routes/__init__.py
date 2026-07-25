"""API routes for Marie services."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from marie.api.routes.workflow import AgentWorkflowRouter

__all__ = [
    "rag_router",
    "create_kb_index_router",
    "create_mcp_router",
    "AgentWorkflowRouter",
    "create_agent_workflow_router",
]


def __getattr__(name: str) -> Any:
    if name == "create_kb_index_router":
        from marie.api.routes.kb_index import create_fastapi_router

        return create_fastapi_router
    if name == "create_mcp_router":
        from marie.api.routes.mcp import create_fastapi_router

        return create_fastapi_router
    if name == "rag_router":
        from marie.api.routes.rag import router

        return router
    if name in {"AgentWorkflowRouter", "create_agent_workflow_router"}:
        from marie.api.routes.workflow import (
            AgentWorkflowRouter,
            create_agent_workflow_router,
        )

        return {
            "AgentWorkflowRouter": AgentWorkflowRouter,
            "create_agent_workflow_router": create_agent_workflow_router,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
