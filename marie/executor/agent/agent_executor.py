"""AgentExecutor - Marie Executor for running agents with pluggable backends.

This module provides the AgentExecutor which integrates the agent framework
with Marie's executor system, enabling agents to be deployed as services.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type, Union

if TYPE_CHECKING:
    from marie.agent.base import BaseAgent
    from marie.agent.guardrails.chain import GuardrailChain

from docarray import DocList
from docarray.documents import TextDoc

from marie.agent.backends import (
    AgentBackend,
    AgentResult,
    AgentStatus,
    BackendConfig,
    OpenAIAgentBackend,
    OpenAIBackendConfig,
)
from marie.agent.config import AgentConfig, load_config
from marie.agent.coordination import (
    AgentExecutionContext,
    CoordinatorFactory,
    coordination_result_to_agent_result,
)
from marie.agent.message import ContentItem, Message
from marie.agent.state.conversation import ConversationStore
from marie.agent.tools import resolve_tools
from marie.agent.tools.base import AgentTool
from marie.engine.output_parser import parse_json_markdown
from marie.executor.marie_executor import MarieExecutor
from marie.logging_core.logger import MarieLogger

try:
    from marie.serve.executors import requests
except ImportError:
    # Fallback for testing
    def requests(on: str = "/"):
        def decorator(func):
            func.__requests__ = {"on": on}
            return func

        return decorator


logger = MarieLogger("marie.executor.agent.agent_executor")


class AgentExecutor(MarieExecutor):
    """Marie Executor for running agents with pluggable backends.

    Provides HTTP/gRPC endpoints for agent interactions with support for:
    - Multiple backend types (OpenAI, Haystack, AutoGen)
    - Tool registration and management
    - Conversation state persistence
    - Streaming responses
    - DAG task spawning

    Example YAML configuration:
        ```yaml
        jtype: AgentExecutor
        with:
          backend: openai
          backend_config:
            model: qwen2_5_vl_7b
            base_url: http://localhost:8000/v1
            max_iterations: 10
          tools:
            - search
            - calculator
          system_message: "You are a helpful assistant."
        ```

    Example usage:
        ```python
        executor = AgentExecutor(
            backend="openai",
            backend_config={
                "model": "qwen2_5_vl_7b",
                "base_url": "http://localhost:8000/v1",
            },
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
        "openai": OpenAIAgentBackend,
    }

    def __init__(
        self,
        backend: str = "openai",
        backend_config: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Union[str, Dict, AgentTool]]] = None,
        system_message: Optional[str] = None,
        config_path: Optional[str] = None,
        enable_conversation_store: bool = True,
        **kwargs: Any,
    ):
        """Initialize the AgentExecutor.

        Args:
            backend: Backend type ('openai', 'haystack', 'autogen')
            backend_config: Backend-specific configuration
            tools: List of tool names, configs, or instances
            system_message: System message for the agent
            config_path: Path to YAML configuration file
            enable_conversation_store: Enable conversation persistence
            **kwargs: Additional MarieExecutor arguments
        """
        super().__init__(**kwargs)

        # Load config from file if provided
        config: Optional[AgentConfig] = None
        if config_path:
            config = load_config(path=config_path)
            backend = config.backend
            backend_config = config.llm.model_dump()
            tools = config.get_tool_list()
            system_message = config.system_message

        # Store full config for coordination support
        self._config = config

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
        self._sub_agents: Dict[str, "BaseAgent"] = {}

        # Guardrail chains (built lazily in _ensure_initialized)
        self._before_chain: Optional["GuardrailChain"] = None
        self._after_chain: Optional["GuardrailChain"] = None
        self._tool_call_chain: Optional["GuardrailChain"] = None

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

        # Initialize guardrails from config
        if self._config and self._config.guardrails:
            from marie.agent.guardrails.chain import GuardrailChain
            from marie.agent.guardrails.registry import resolve_guardrails_for_phase

            gc = self._config.guardrails

            # Build before guardrails
            if gc.before:
                before_guards = resolve_guardrails_for_phase(
                    "before", [e.model_dump() for e in gc.before]
                )
                self._before_chain = GuardrailChain(before_guards)
                logger.info(f"Initialized {len(before_guards)} before-guardrails")

            # Build after guardrails
            if gc.after:
                after_guards = resolve_guardrails_for_phase(
                    "after", [e.model_dump() for e in gc.after]
                )
                self._after_chain = GuardrailChain(after_guards)
                logger.info(f"Initialized {len(after_guards)} after-guardrails")

            # Build tool-call guardrails and wrap tools
            if gc.tool_call:
                from marie.agent.guardrails.guarded_tool import GuardedTool

                tool_guards = resolve_guardrails_for_phase(
                    "tool_call", [e.model_dump() for e in gc.tool_call]
                )
                self._tool_call_chain = GuardrailChain(tool_guards)
                logger.info(f"Initialized {len(tool_guards)} tool-call guardrails")

                # Wrap all tools with guardrails
                self._tools = {
                    name: GuardedTool(tool, self._tool_call_chain)
                    for name, tool in self._tools.items()
                }
                logger.info("Wrapped tools with tool-call guardrails")

        # Initialize backend
        self._backend = self._create_backend()
        logger.info(f"Initialized backend: {self._backend_name}")

        self._initialized = True

    def _should_use_coordination(self) -> bool:
        """Check if coordination mode should be used."""
        return (
            self._config is not None
            and self._config.coordination is not None
            and self._config.coordination.topology
            in ("parallel", "sequential", "workflow")
            and self._config.sub_agents is not None
            and len(self._config.sub_agents) > 0
        )

    def _load_sub_agents(self) -> Dict[str, "BaseAgent"]:
        """Load sub-agents by name.

        Note: For now, this returns cached agents. In production, this would
        resolve agent names to actual agent instances from a registry.
        """
        if self._sub_agents:
            return self._sub_agents

        if not self._config or not self._config.sub_agents:
            return {}

        # TODO: Implement agent registry lookup
        # For now, log a warning that sub-agents need to be registered
        logger.warning(
            f"Sub-agents requested but not registered: {self._config.sub_agents}. "
            "Use register_sub_agent() to add agent instances."
        )
        return self._sub_agents

    def register_sub_agent(self, name: str, agent: "BaseAgent") -> None:
        """Register a sub-agent for coordination.

        Args:
            name: Agent name (must match name in config.sub_agents)
            agent: Agent instance
        """
        self._sub_agents[name] = agent
        logger.info(f"Registered sub-agent: {name}")

    async def _coordinated_chat(
        self,
        docs: DocList[TextDoc],
        parameters: Dict[str, Any],
    ) -> DocList[TextDoc]:
        """Execute chat using coordination mode.

        Args:
            docs: Input documents
            parameters: Request parameters

        Returns:
            DocList with coordinated response
        """
        conversation_id = parameters.get("conversation_id", str(uuid.uuid4()))

        # ── BEFORE GUARDRAILS ──
        docs, before_results, early_return = await self._run_before_guardrails(
            docs,
            conversation_id,
            parameters,
        )
        if early_return is not None:
            return early_return

        messages = await self._build_messages(docs, conversation_id, parameters)

        start_time = time.time()

        # Create execution context for tracing
        with AgentExecutionContext(
            workflow_id=f"coord-{conversation_id}",
            agent_name="coordinator",
            session_id=parameters.get("session_id"),
            user_id=parameters.get("user_id"),
        ):
            # Get coordinator
            coord_config = self._config.coordination
            coordinator = CoordinatorFactory.create(coord_config)

            # Load and add sub-agents
            sub_agents = self._load_sub_agents()
            if not sub_agents:
                logger.error("No sub-agents available for coordination")
                return DocList[TextDoc](
                    [
                        TextDoc(
                            text="Error: No sub-agents configured for coordination",
                            tags={"status": "failed", "error": "no_sub_agents"},
                        )
                    ]
                )

            for agent in sub_agents.values():
                coordinator.add_agent(agent)

            # Run coordination
            try:
                coord_result = await asyncio.wait_for(
                    coordinator.run(
                        messages,
                        session_id=parameters.get("session_id"),
                        user_id=parameters.get("user_id"),
                    ),
                    timeout=coord_config.timeout,
                )
            except asyncio.TimeoutError:
                logger.error(f"Coordination timed out after {coord_config.timeout}s")
                return DocList[TextDoc](
                    [
                        TextDoc(
                            text=f"Coordination timed out after {coord_config.timeout} seconds",
                            tags={
                                "status": "timeout",
                                "conversation_id": conversation_id,
                            },
                        )
                    ]
                )

            # Convert to AgentResult for API compatibility
            result = coordination_result_to_agent_result(coord_result)

        # ── AFTER GUARDRAILS ──
        response_text, after_results = await self._run_after_guardrails(
            result.output_text,
            conversation_id,
            parameters,
        )

        # Update conversation store with SANITIZED text
        if self._conversation_store:
            user_content = (
                "\n".join(doc.text for doc in docs if doc.text) if docs else ""
            )
            await self._conversation_store.add_message(
                conversation_id,
                Message.user(user_content),
            )
            if response_text:
                await self._conversation_store.add_message(
                    conversation_id,
                    Message.assistant(response_text),
                )

        # Build response
        duration_ms = (time.time() - start_time) * 1000

        response_doc = TextDoc(
            text=response_text,
            tags={
                "conversation_id": conversation_id,
                "status": result.status.value,
                "duration_ms": duration_ms,
                "coordination": {
                    "topology": coord_result.topology,
                    "success_count": coord_result.success_count,
                    "failure_count": coord_result.failure_count,
                },
                "guardrails": {
                    "before": before_results,
                    "after": after_results,
                },
            },
        )

        return DocList[TextDoc]([response_doc])

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

        if backend_cls is OpenAIAgentBackend:
            return backend_cls(config=OpenAIBackendConfig(**config))
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

    @requests(on="/agent/execute")
    async def execute_endpoint(
        self,
        docs: DocList[TextDoc],
        parameters: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> DocList[TextDoc]:
        """Execute one stateless agent task for a workflow node."""
        self._ensure_initialized()
        parameters = parameters or {}
        execution_id = str(parameters.get("execution_id") or uuid.uuid4())

        docs, before_results, early_return = await self._run_before_guardrails(
            docs,
            execution_id,
            parameters,
        )
        if early_return is not None:
            return early_return

        messages = await self._build_messages(
            docs,
            execution_id,
            parameters,
            include_history=False,
        )
        output_schema = parameters.get("output_schema")
        if output_schema is not None:
            if not isinstance(output_schema, dict):
                raise ValueError("output_schema must be an object")
            messages.append(
                Message.user(
                    "Return the final answer as one JSON object matching this JSON "
                    f"Schema. Do not include commentary.\n{json.dumps(output_schema)}"
                )
            )

        backend_parameters = {
            key: parameters[key]
            for key in ("agent_id", "seed", "skill_name", "user_id")
            if key in parameters
        }
        timeout_seconds = float(
            parameters.get(
                "timeout_seconds",
                self._backend_config.get("timeout_seconds", 300.0),
            )
        )
        structured_output_retries = int(parameters.get("structured_output_retries", 1))
        if structured_output_retries < 0 or structured_output_retries > 3:
            raise ValueError("structured_output_retries must be between 0 and 3")

        start_time = time.time()
        attempt_messages = list(messages)
        total_iterations = 0
        total_tool_calls = 0
        structured_output_attempts = 0
        with AgentExecutionContext(
            workflow_id=execution_id,
            session_id=parameters.get("session_id"),
            user_id=parameters.get("user_id"),
            agent_name=self._config.name if self._config else self._backend_name,
        ):
            for attempt in range(structured_output_retries + 1):
                structured_output_attempts = attempt + 1
                try:
                    result = await asyncio.wait_for(
                        self._backend.run(
                            messages=attempt_messages,
                            tools=self._tools,
                            config=None,
                            **backend_parameters,
                        ),
                        timeout=timeout_seconds,
                    )
                except asyncio.TimeoutError as error:
                    raise TimeoutError(
                        f"Agent execution timed out after {timeout_seconds}s"
                    ) from error

                if result.status != AgentStatus.COMPLETED:
                    raise RuntimeError(result.error or "Agent execution failed")
                total_iterations += result.iterations
                total_tool_calls += len(result.tool_calls)

                response_text, after_results = await self._run_after_guardrails(
                    result.output_text,
                    execution_id,
                    parameters,
                    extra_ctx={
                        "iterations": result.iterations,
                        "tool_calls_count": len(result.tool_calls),
                    },
                )
                if output_schema is None:
                    break

                try:
                    from jsonschema import Draft202012Validator

                    payload = parse_json_markdown(response_text)
                    if not isinstance(payload, dict):
                        raise ValueError("Agent structured output must be an object")
                    Draft202012Validator(output_schema).validate(payload)
                except Exception as error:
                    if attempt == structured_output_retries:
                        raise
                    attempt_messages.extend(
                        [
                            Message.assistant(response_text),
                            Message.user(
                                "The previous final answer did not satisfy the "
                                f"required JSON object schema: {error}. Return a "
                                "corrected JSON object only."
                            ),
                        ]
                    )
                    continue

                response_text = json.dumps(payload, separators=(",", ":"))
                break

        return DocList[TextDoc](
            [
                TextDoc(
                    text=response_text,
                    tags={
                        "execution_id": execution_id,
                        "status": result.status.value,
                        "iterations": total_iterations,
                        "structured_output_attempts": structured_output_attempts,
                        "duration_ms": (time.time() - start_time) * 1000,
                        "tool_calls": total_tool_calls,
                        "guardrails": {
                            "before": before_results,
                            "after": after_results,
                        },
                    },
                )
            ]
        )

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

        # Check for coordination mode
        if self._should_use_coordination():
            return await self._coordinated_chat(docs, parameters)

        # Extract conversation ID
        conversation_id = parameters.get("conversation_id", str(uuid.uuid4()))

        # ── BEFORE GUARDRAILS ──
        docs, before_results, early_return = await self._run_before_guardrails(
            docs,
            conversation_id,
            parameters,
        )
        if early_return is not None:
            return early_return

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
                status=AgentStatus.FAILED,
                error="Execution timeout",
            )

        # ── AFTER GUARDRAILS ──
        response_text, after_results = await self._run_after_guardrails(
            result.output_text,
            conversation_id,
            parameters,
            extra_ctx={
                "iterations": result.iterations,
                "tool_calls_count": len(result.tool_calls),
            },
        )

        # Update conversation store with SANITIZED text
        if self._conversation_store:
            # Add user message
            user_content = (
                "\n".join(doc.text for doc in docs if doc.text) if docs else ""
            )
            await self._conversation_store.add_message(
                conversation_id,
                Message.user(user_content),
            )
            # Add assistant response (sanitized)
            if response_text:
                await self._conversation_store.add_message(
                    conversation_id,
                    Message.assistant(response_text),
                )

        # Build response
        duration_ms = (time.time() - start_time) * 1000

        response_doc = TextDoc(
            text=response_text,
            tags={
                "conversation_id": conversation_id,
                "status": result.status.value,
                "iterations": result.iterations,
                "duration_ms": duration_ms,
                "tool_calls": len(result.tool_calls),
                "guardrails": {
                    "before": before_results,
                    "after": after_results,
                },
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

        Uses ``backend.run_stream()`` to stream token-level chunks.
        Currently collects all chunks and returns the assembled response
        as a single ``TextDoc``.  When SSE/WebSocket transport is
        available, the chunks can be forwarded directly.
        """
        self._ensure_initialized()
        parameters = parameters or {}

        conversation_id = parameters.get("conversation_id", str(uuid.uuid4()))

        # ── BEFORE GUARDRAILS ──
        docs, before_results, early_return = await self._run_before_guardrails(
            docs,
            conversation_id,
            parameters,
        )
        if early_return is not None:
            return early_return

        messages = await self._build_messages(docs, conversation_id, parameters)

        # Build abort signal from optional timeout
        from marie.agent.cancellation import AbortSignal
        from marie.agent.streaming import StreamChunk

        timeout_seconds = self._backend_config.get("timeout_seconds", 300.0)
        abort_signal = AbortSignal.timeout(timeout_seconds)

        start_time = time.time()
        collected_text_parts: List[str] = []
        final_result: Optional[AgentResult] = None

        try:
            async for item in self._backend.run_stream(
                messages=messages,
                tools=self._tools,
                abort_signal=abort_signal,
                **parameters,
            ):
                if isinstance(item, AgentResult):
                    final_result = item
                elif isinstance(item, StreamChunk):
                    if item.content:
                        collected_text_parts.append(item.content)
        except Exception as e:
            logger.error(f"Streaming execution failed: {e}")
            return DocList[TextDoc](
                [
                    TextDoc(
                        text=f"Streaming error: {e}",
                        tags={
                            "conversation_id": conversation_id,
                            "status": "failed",
                        },
                    )
                ]
            )

        duration_ms = (time.time() - start_time) * 1000
        response_text = "".join(collected_text_parts)

        # ── AFTER GUARDRAILS ──
        response_text, after_results = await self._run_after_guardrails(
            response_text,
            conversation_id,
            parameters,
        )

        # Update conversation store with SANITIZED text
        if self._conversation_store:
            user_content = (
                "\n".join(doc.text for doc in docs if doc.text) if docs else ""
            )
            await self._conversation_store.add_message(
                conversation_id, Message.user(user_content)
            )
            if response_text:
                await self._conversation_store.add_message(
                    conversation_id, Message.assistant(response_text)
                )

        status = final_result.status.value if final_result else "completed"

        return DocList[TextDoc](
            [
                TextDoc(
                    text=response_text,
                    tags={
                        "conversation_id": conversation_id,
                        "status": status,
                        "duration_ms": duration_ms,
                        "streamed": True,
                        "guardrails": {
                            "before": before_results,
                            "after": after_results,
                        },
                    },
                )
            ]
        )

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
        *,
        include_history: bool = True,
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
        if include_history and self._conversation_store:
            history = await self._conversation_store.get_messages(conversation_id)
            messages.extend(history)

        content: List[ContentItem] = []
        user_content = "\n".join(doc.text for doc in docs if doc.text)
        if user_content:
            content.append(ContentItem(text=user_content))

        content_items = parameters.get("content_items")
        if content_items is not None:
            if not isinstance(content_items, list):
                raise ValueError("content_items must be an array")
            content.extend(ContentItem.model_validate(item) for item in content_items)

        if content:
            messages.append(Message.user(content))

        return messages

    async def _run_before_guardrails(
        self,
        docs: DocList[TextDoc],
        conversation_id: str,
        parameters: Dict[str, Any],
    ) -> tuple[DocList[TextDoc], List[Dict[str, Any]], Optional[DocList[TextDoc]]]:
        """Run before-guardrails on input docs.

        Args:
            docs: Input documents
            conversation_id: Conversation identifier
            parameters: Request parameters

        Returns:
            Tuple of (possibly_modified_docs, guardrail_results, early_return_or_None).
            If early_return is not None, the caller should return it immediately.
        """
        from marie.agent.guardrails.result import GuardrailAction

        results: List[Dict[str, Any]] = []

        if not self._before_chain or self._before_chain.is_empty:
            return docs, results, None

        # Guard the FULL joined input, matching _build_messages
        user_text = "\n".join(doc.text for doc in docs if doc.text)

        ctx = {
            "phase": "before",
            "agent_name": self._backend_name,
            "conversation_id": conversation_id,
            "user_id": parameters.get("user_id"),
        }

        chain_result = await self._before_chain.run(user_text, ctx)
        results = [
            {"name": r.guardrail_name, "action": r.action.value, "score": r.score}
            for r in chain_result.results
        ]

        if chain_result.action == GuardrailAction.BLOCK:
            blocked = chain_result.results[-1] if chain_result.results else None
            return (
                docs,
                results,
                DocList[TextDoc](
                    [
                        TextDoc(
                            text=(
                                blocked.message
                                if blocked
                                else "Request blocked by safety policy."
                            ),
                            tags={
                                "conversation_id": conversation_id,
                                "status": "blocked",
                                "guardrail": (
                                    blocked.guardrail_name if blocked else "unknown"
                                ),
                            },
                        )
                    ]
                ),
            )

        if chain_result.action == GuardrailAction.ESCALATE:
            escalated = chain_result.results[-1] if chain_result.results else None
            return (
                docs,
                results,
                DocList[TextDoc](
                    [
                        TextDoc(
                            text="This request requires human review.",
                            tags={
                                "conversation_id": conversation_id,
                                "status": "escalated",
                                "guardrail": (
                                    escalated.guardrail_name if escalated else "unknown"
                                ),
                            },
                        )
                    ]
                ),
            )

        # MODIFY: rebuild docs with sanitized content
        if chain_result.final_content != user_text:
            docs = DocList[TextDoc]([TextDoc(text=chain_result.final_content)])

        return docs, results, None

    async def _run_after_guardrails(
        self,
        response_text: str,
        conversation_id: str,
        parameters: Dict[str, Any],
        extra_ctx: Optional[Dict[str, Any]] = None,
    ) -> tuple[str, List[Dict[str, Any]]]:
        """Run after-guardrails on response text.

        Args:
            response_text: Agent response text
            conversation_id: Conversation identifier
            parameters: Request parameters
            extra_ctx: Additional context (iterations, tool_calls_count, etc.)

        Returns:
            Tuple of (possibly_modified_response_text, guardrail_results)
        """
        from marie.agent.guardrails.result import GuardrailAction

        results: List[Dict[str, Any]] = []

        if not self._after_chain or self._after_chain.is_empty:
            return response_text, results

        ctx = {
            "phase": "after",
            "agent_name": self._backend_name,
            "conversation_id": conversation_id,
            "user_id": parameters.get("user_id"),
        }
        if extra_ctx:
            ctx.update(extra_ctx)

        chain_result = await self._after_chain.run(response_text, ctx)
        results = [
            {"name": r.guardrail_name, "action": r.action.value, "score": r.score}
            for r in chain_result.results
        ]

        if chain_result.action == GuardrailAction.BLOCK:
            blocked = chain_result.results[-1] if chain_result.results else None
            logger.warning(
                "After-guardrail blocked: guardrail=%s score=%.2f",
                blocked.guardrail_name if blocked else "unknown",
                blocked.score if blocked else 0.0,
            )
            return "I'm unable to provide that response.", results

        if chain_result.final_content != response_text:
            return chain_result.final_content, results

        return response_text, results

    def add_tool(self, tool: Union[str, Dict, AgentTool]) -> None:
        """Add a tool to the executor.

        Args:
            tool: Tool name, config, or instance
        """
        resolved = resolve_tools([tool])

        # Wrap with tool-call guardrails if active
        if self._tool_call_chain:
            from marie.agent.guardrails.guarded_tool import GuardedTool

            resolved = {
                name: GuardedTool(t, self._tool_call_chain)
                for name, t in resolved.items()
            }

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
        instance = cls(
            backend=config.backend,
            backend_config=config.llm.model_dump(),
            tools=config.get_tool_list(),
            system_message=config.system_message,
            **kwargs,
        )
        # Preserve full config for coordination AND guardrails.
        # __init__ only sets self._config when config_path is provided,
        # so from_config must set it explicitly.
        instance._config = config
        return instance

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
