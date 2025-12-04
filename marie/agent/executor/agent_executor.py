"""AgentExecutor - Marie Executor for running agents with pluggable backends.

This module provides the AgentExecutor which integrates the agent framework
with Marie's executor system, enabling agents to be deployed as services.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, Dict, List, Optional, Type, Union

from docarray import DocList
from docarray.documents import TextDoc
from pydantic import Field

from marie.agent.backends import (
    AgentBackend,
    AgentResult,
    AgentStatus,
    BackendConfig,
    QwenAgentBackend,
)
from marie.agent.config import AgentConfig, load_config
from marie.agent.message import Message
from marie.agent.state.conversation import ConversationStore
from marie.agent.tools import resolve_tools
from marie.agent.tools.base import AgentTool
from marie.executor.marie_executor import MarieExecutor
from marie.logging_core.logger import MarieLogger
from marie.serve.runtimes.gateway.streamer import GatewayStreamer

try:
    from marie.serve.executors import requests
except ImportError:
    # Fallback for testing
    def requests(on: str = "/"):
        def decorator(func):
            func.__requests__ = {"on": on}
            return func

        return decorator


logger = MarieLogger("marie.agent.executor")


class AgentExecutor(MarieExecutor):
    """Marie Executor for running agents with pluggable backends.

    Provides HTTP/gRPC endpoints for agent interactions with support for:
    - Multiple backend types (Qwen, Haystack, AutoGen)
    - Tool registration and management
    - Conversation state persistence
    - Streaming responses
    - DAG task spawning

    Example YAML configuration:
        ```yaml
        jtype: AgentExecutor
        with:
          backend: qwen_agent
          backend_config:
            engine_name: qwen2_5_vl_7b
            max_iterations: 10
          tools:
            - search
            - calculator
          system_message: "You are a helpful assistant."
        ```

    Example usage:
        ```python
        executor = AgentExecutor(
            backend="qwen_agent",
            backend_config={"engine_name": "qwen2_5_vl_7b"},
            tools=["search", "calculator"],
        )

        # Via endpoint
        result = await executor.chat_endpoint(
            docs=[TextDoc(text="Hello, what can you do?")],
            parameters={"conversation_id": "conv-123"},
        )
        ```
    """

    # Backend registry mapping names to backend classes
    BACKEND_REGISTRY: Dict[str, Type[AgentBackend]] = {
        "qwen_agent": QwenAgentBackend,
    }

    def __init__(
        self,
        backend: str = "qwen_agent",
        backend_config: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Union[str, Dict, AgentTool]]] = None,
        system_message: Optional[str] = None,
        config_path: Optional[str] = None,
        enable_conversation_store: bool = True,
        **kwargs: Any,
    ):
        """Initialize the AgentExecutor.

        Args:
            backend: Backend type ('qwen_agent', 'haystack', 'autogen')
            backend_config: Backend-specific configuration
            tools: List of tool names, configs, or instances
            system_message: System message for the agent
            config_path: Path to YAML configuration file
            enable_conversation_store: Enable conversation persistence
            **kwargs: Additional MarieExecutor arguments
        """
        super().__init__(**kwargs)

        # Load config from file if provided
        if config_path:
            config = load_config(path=config_path)
            backend = config.backend
            backend_config = config.llm.model_dump()
            tools = config.get_tool_list()
            system_message = config.system_message

        self._backend_name = backend
        self._backend_config = backend_config or {}
        self._system_message = system_message
        self._tool_specs = tools or []

        # Initialize components
        self._backend: Optional[AgentBackend] = None
        self._tools: Dict[str, AgentTool] = {}
        self._conversation_store: Optional[ConversationStore] = None

        if enable_conversation_store:
            self._conversation_store = ConversationStore()

        # Lazy initialization
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Ensure backend and tools are initialized."""
        if self._initialized:
            return

        # Initialize tools
        if self._tool_specs:
            self._tools = resolve_tools(self._tool_specs)
            logger.info(
                f"Initialized {len(self._tools)} tools: {list(self._tools.keys())}"
            )

        # Initialize backend
        self._backend = self._create_backend()
        logger.info(f"Initialized backend: {self._backend_name}")

        self._initialized = True

    def _create_backend(self) -> AgentBackend:
        """Create the agent backend.

        Returns:
            Configured AgentBackend instance

        Raises:
            ValueError: If backend type is unknown
        """
        # Try to import additional backends lazily
        self._register_optional_backends()

        if self._backend_name not in self.BACKEND_REGISTRY:
            available = ", ".join(self.BACKEND_REGISTRY.keys())
            raise ValueError(
                f"Unknown backend '{self._backend_name}'. "
                f"Available backends: {available}"
            )

        backend_cls = self.BACKEND_REGISTRY[self._backend_name]

        # Add system message to config
        config = dict(self._backend_config)
        if self._system_message:
            config["system_message"] = self._system_message

        return backend_cls(config=BackendConfig(**config))

    def _register_optional_backends(self) -> None:
        """Register optional backends if their dependencies are available."""
        # Haystack backend
        if "haystack" not in self.BACKEND_REGISTRY:
            try:
                from marie.agent.backends.haystack_backend import HaystackAgentBackend

                self.BACKEND_REGISTRY["haystack"] = HaystackAgentBackend
            except ImportError:
                pass

        # AutoGen backend
        if "autogen" not in self.BACKEND_REGISTRY:
            try:
                from marie.agent.backends.autogen_backend import AutoGenAgentBackend

                self.BACKEND_REGISTRY["autogen"] = AutoGenAgentBackend
            except ImportError:
                pass

    @requests(on="/chat")
    async def chat_endpoint(
        self,
        docs: DocList[TextDoc],
        parameters: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> DocList[TextDoc]:
        """Main chat endpoint for agent interaction.

        Args:
            docs: Input documents (text content)
            parameters: Request parameters including:
                - conversation_id: Optional conversation ID for continuity
                - max_iterations: Override max iterations
                - stream: Whether to stream response
            **kwargs: Additional arguments

        Returns:
            DocList containing agent response
        """
        self._ensure_initialized()
        parameters = parameters or {}

        # Extract conversation ID
        conversation_id = parameters.get("conversation_id", str(uuid.uuid4()))

        # Build messages from docs and conversation history
        messages = await self._build_messages(docs, conversation_id, parameters)

        # Get timeout from backend config (default 300 seconds)
        timeout_seconds = self._backend_config.get("timeout_seconds", 300.0)

        # Run agent with timeout
        start_time = time.time()
        try:
            result = await asyncio.wait_for(
                self._backend.run(
                    messages=messages,
                    tools=self._tools,
                    config=None,
                    **parameters,
                ),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError:
            logger.error(f"Agent execution timed out after {timeout_seconds}s")
            from marie.agent.backends.base import AgentResult, AgentStatus

            result = AgentResult(
                output=f"Request timed out after {timeout_seconds} seconds.",
                status=AgentStatus.ERROR,
                error="Execution timeout",
            )

        # Update conversation store
        if self._conversation_store:
            # Add user message
            user_content = docs[0].text if docs else ""
            await self._conversation_store.add_message(
                conversation_id,
                Message.user(user_content),
            )
            # Add assistant response
            if result.output:
                await self._conversation_store.add_message(
                    conversation_id,
                    (
                        result.output
                        if isinstance(result.output, Message)
                        else Message.assistant(result.output_text)
                    ),
                )

        # Build response
        duration_ms = (time.time() - start_time) * 1000
        response_text = result.output_text

        response_doc = TextDoc(
            text=response_text,
            tags={
                "conversation_id": conversation_id,
                "status": result.status.value,
                "iterations": result.iterations,
                "duration_ms": duration_ms,
                "tool_calls": len(result.tool_calls),
            },
        )

        return DocList[TextDoc]([response_doc])

    @requests(on="/chat/stream")
    async def chat_stream_endpoint(
        self,
        docs: DocList[TextDoc],
        parameters: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> DocList[TextDoc]:
        """Streaming chat endpoint.

        Note:
            Streaming is not yet implemented. This endpoint currently
            returns the complete response. True streaming will require
            SSE (Server-Sent Events) or WebSocket support.
        """
        # Streaming not yet implemented - delegate to regular chat
        return await self.chat_endpoint(docs, parameters, **kwargs)

    @requests(on="/tools")
    async def list_tools_endpoint(
        self,
        docs: Optional[DocList[TextDoc]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> DocList[TextDoc]:
        """List available tools.

        Returns:
            DocList containing tool information as JSON
        """
        self._ensure_initialized()

        import json

        tools_info = []
        for name, tool in self._tools.items():
            tools_info.append(
                {
                    "name": name,
                    "description": tool.description,
                    "parameters": tool.metadata.get_parameters_dict(),
                }
            )

        return DocList[TextDoc]([TextDoc(text=json.dumps(tools_info, indent=2))])

    @requests(on="/conversations")
    async def list_conversations_endpoint(
        self,
        docs: Optional[DocList[TextDoc]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> DocList[TextDoc]:
        """List active conversations.

        Returns:
            DocList containing conversation IDs
        """
        import json

        if not self._conversation_store:
            return DocList[TextDoc]([TextDoc(text="[]")])

        conversations = await self._conversation_store.list_conversations()
        return DocList[TextDoc]([TextDoc(text=json.dumps(conversations, indent=2))])

    @requests(on="/conversation/clear")
    async def clear_conversation_endpoint(
        self,
        docs: Optional[DocList[TextDoc]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> DocList[TextDoc]:
        """Clear a conversation.

        Args:
            parameters: Must include 'conversation_id'

        Returns:
            Confirmation message
        """
        parameters = parameters or {}
        conversation_id = parameters.get("conversation_id")

        if not conversation_id:
            return DocList[TextDoc](
                [TextDoc(text="Error: conversation_id is required")]
            )

        if self._conversation_store:
            await self._conversation_store.clear(conversation_id)

        return DocList[TextDoc](
            [TextDoc(text=f"Conversation {conversation_id} cleared")]
        )

    async def _build_messages(
        self,
        docs: DocList[TextDoc],
        conversation_id: str,
        parameters: Dict[str, Any],
    ) -> List[Message]:
        """Build message list from docs and conversation history.

        Args:
            docs: Input documents
            conversation_id: Conversation ID for history lookup
            parameters: Request parameters

        Returns:
            List of Messages including history and new input
        """
        messages: List[Message] = []

        # Add conversation history
        if self._conversation_store:
            history = await self._conversation_store.get_messages(conversation_id)
            messages.extend(history)

        # Add new user message from docs
        if docs:
            user_content = "\n".join(doc.text for doc in docs if doc.text)
            if user_content:
                messages.append(Message.user(user_content))

        return messages

    def add_tool(self, tool: Union[str, Dict, AgentTool]) -> None:
        """Add a tool to the executor.

        Args:
            tool: Tool name, config, or instance
        """
        resolved = resolve_tools([tool])
        self._tools.update(resolved)

    def remove_tool(self, name: str) -> bool:
        """Remove a tool by name.

        Args:
            name: Tool name

        Returns:
            True if removed, False if not found
        """
        if name in self._tools:
            del self._tools[name]
            return True
        return False

    @classmethod
    def from_config(cls, config: AgentConfig, **kwargs: Any) -> "AgentExecutor":
        """Create an AgentExecutor from configuration.

        Args:
            config: AgentConfig instance
            **kwargs: Additional arguments

        Returns:
            Configured AgentExecutor
        """
        return cls(
            backend=config.backend,
            backend_config=config.llm.model_dump(),
            tools=config.get_tool_list(),
            system_message=config.system_message,
            **kwargs,
        )

    @classmethod
    def register_backend(
        cls,
        name: str,
        backend_cls: Type[AgentBackend],
    ) -> None:
        """Register a custom backend.

        Args:
            name: Backend name
            backend_cls: Backend class
        """
        cls.BACKEND_REGISTRY[name] = backend_cls
