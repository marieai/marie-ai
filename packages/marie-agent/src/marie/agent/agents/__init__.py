"""Agent implementations for Marie agent framework."""

from marie.agent.agents.assistant import (
    ChatAgent,
    FunctionCallingAgent,
    PlanAndExecuteAgent,
    ReactAgent,
)
from marie.agent.agents.router import MultiAgentHub, Router
from marie.agent.agents.vision_document_agent import (
    DocumentExtractionAgent,
    DocumentQAAgent,
    VisionDocumentAgent,
)

__all__ = [
    # ReAct and Plan-and-Execute agents
    "ReactAgent",
    "PlanAndExecuteAgent",
    # Other agents
    "ChatAgent",
    "FunctionCallingAgent",
    # Router
    "Router",
    "MultiAgentHub",
    # Vision Document Agents
    "VisionDocumentAgent",
    "DocumentExtractionAgent",
    "DocumentQAAgent",
]
