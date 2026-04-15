"""API routes for Marie services."""

from marie.api.routes.mcp import create_fastapi_router as create_mcp_router
from marie.api.routes.rag import router as rag_router
from marie.api.routes.rag_index import create_fastapi_router as create_rag_index_router
from marie.api.routes.workflow import (
    AgentWorkflowRouter,
    create_agent_workflow_router,
)

__all__ = [
    "rag_router",
    "create_rag_index_router",
    "create_mcp_router",
    "AgentWorkflowRouter",
    "create_agent_workflow_router",
]
