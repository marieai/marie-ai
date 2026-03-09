"""A2A Executor for exposing Marie agents via A2A protocol.

This module provides A2AMarieExecutor that wraps Marie agents and exposes
them via the A2A protocol using the official a2a-sdk package.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

from docarray import DocList
from docarray.documents import TextDoc

from marie import Executor, requests
from marie.agent.a2a import (
    A2AExecutor,
    AgentCardBuilder,
    MarieA2AExecutor,
)
from marie.logging_core.logger import MarieLogger
from marie.serve.executors import __dry_run_endpoint__

if TYPE_CHECKING:
    from marie.agent.base import BaseAgent

logger = logging.getLogger(__name__)


class A2AMarieExecutor(Executor):
    """Marie Executor exposing agents via A2A protocol.

    This executor wraps a Marie agent and exposes it through standard
    A2A protocol endpoints:
    - `/.well-known/agent.json` - Agent card discovery
    - `/a2a` - JSON-RPC endpoint for A2A operations

    Example:
        # In executor config YAML:
        jtype: A2AMarieExecutor
        with:
          agent_class: marie.agent.backends.QwenAgentBackend
          agent_config:
            name: "My Qwen Agent"
            model: "qwen2.5-72b"
          a2a_config:
            url: "http://localhost:8000"
            streaming: true
    """

    def __init__(
        self,
        agent_class: Optional[str] = None,
        agent_config: Optional[Dict[str, Any]] = None,
        a2a_config: Optional[Dict[str, Any]] = None,
        agent: Optional["BaseAgent"] = None,
        **kwargs,
    ):
        """Initialize the A2A executor.

        Args:
            agent_class: Fully qualified class name of the agent to instantiate.
            agent_config: Configuration dict to pass to the agent constructor.
            a2a_config: A2A-specific configuration (url, streaming, etc.).
            agent: Pre-instantiated agent instance (alternative to agent_class).
            **kwargs: Additional arguments for the base Executor.
        """
        super().__init__(**kwargs)
        self.logger = MarieLogger(self.__class__.__name__)
        self.logger.info("A2AMarieExecutor initializing")

        # Extract A2A config
        a2a_config = a2a_config or {}
        self._url = a2a_config.get("url", "http://localhost:8000")
        self._streaming = a2a_config.get("streaming", True)
        self._push_notifications = a2a_config.get("push_notifications", False)
        self._version = a2a_config.get("version", "1.0.0")

        # Instantiate or use provided agent
        if agent is not None:
            self._agent = agent
        elif agent_class:
            self._agent = self._instantiate_agent(agent_class, agent_config or {})
        else:
            raise ValueError("Either agent or agent_class must be provided")

        # Initialize A2A executor
        self._a2a_executor = None
        self._initialized = False

    def _instantiate_agent(
        self, agent_class: str, config: Dict[str, Any]
    ) -> "BaseAgent":
        """Dynamically instantiate an agent from class name."""
        from marie.helper import import_class

        cls = import_class(agent_class)
        return cls(**config)

    def _ensure_initialized(self) -> None:
        """Lazy initialization of A2A executor."""
        if self._initialized:
            return

        self._a2a_executor = A2AExecutor(
            agent=self._agent,
            name=getattr(self._agent, "name", None),
            url=self._url,
            version=self._version,
            description=getattr(self._agent, "description", None),
            streaming=self._streaming,
            push_notifications=self._push_notifications,
        )
        self._initialized = True
        self.logger.info(
            f"A2A executor initialized for agent: {self._a2a_executor.agent_card.name}"
        )

    @property
    def agent_card(self):
        """Get the A2A agent card."""
        self._ensure_initialized()
        return self._a2a_executor.agent_card

    @requests(on="/.well-known/agent.json")
    async def agent_card_endpoint(
        self,
        docs: DocList[TextDoc],
        parameters: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> DocList[TextDoc]:
        """Return the agent card for A2A discovery.

        Returns:
            DocList containing the agent card as JSON.
        """
        self._ensure_initialized()
        card_json = self._a2a_executor.agent_card.model_dump(
            by_alias=True, exclude_none=True
        )
        return DocList[TextDoc]([TextDoc(text=json.dumps(card_json))])

    @requests(on="/a2a")
    async def a2a_endpoint(
        self,
        docs: DocList[TextDoc],
        parameters: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> DocList[TextDoc]:
        """Handle A2A JSON-RPC requests.

        Accepts JSON-RPC requests in the request body (via docs[0].text)
        and returns the JSON-RPC response.

        Args:
            docs: DocList containing the JSON-RPC request.
            parameters: Additional parameters.

        Returns:
            DocList containing the JSON-RPC response.
        """
        self._ensure_initialized()

        # Extract request from first doc
        if not docs or not docs[0].text:
            error_response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {
                    "code": -32600,
                    "message": "Invalid request: empty body",
                },
            }
            return DocList[TextDoc]([TextDoc(text=json.dumps(error_response))])

        try:
            request = json.loads(docs[0].text)
        except json.JSONDecodeError as e:
            error_response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {
                    "code": -32700,
                    "message": f"Parse error: {e}",
                },
            }
            return DocList[TextDoc]([TextDoc(text=json.dumps(error_response))])

        # Handle request via SDK
        response = await self._a2a_executor.handle_request(request)

        # Convert response to JSON
        if hasattr(response, "model_dump"):
            response_json = response.model_dump(by_alias=True, exclude_none=True)
        else:
            response_json = response

        return DocList[TextDoc]([TextDoc(text=json.dumps(response_json))])

    @requests(on="/")
    async def default_endpoint(
        self,
        docs: DocList[TextDoc],
        parameters: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> DocList[TextDoc]:
        """Default endpoint returning agent info."""
        self._ensure_initialized()
        info = {
            "name": self._a2a_executor.agent_card.name,
            "description": self._a2a_executor.agent_card.description,
            "version": self._a2a_executor.agent_card.version,
            "a2a_endpoint": "/a2a",
            "agent_card_endpoint": "/.well-known/agent.json",
        }
        return DocList[TextDoc]([TextDoc(text=json.dumps(info))])

    @requests(on=__dry_run_endpoint__)
    async def dry_run_func(
        self,
        docs: DocList[TextDoc],
        parameters: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """Dry run endpoint for health checks."""
        self._ensure_initialized()
        self.logger.debug("A2A executor dry run completed")
