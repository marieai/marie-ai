"""Base agent class for Marie agent framework.

This module provides the abstract base class for all agents following
the Qwen-Agent template method pattern with marie.engine integration.

All agents inherit skill support, enabling Claude Code-like slash commands
and automatic skill routing.
"""

from __future__ import annotations

import copy
import re
from abc import ABC, abstractmethod

# Import SearchableToolset for type checking
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

from openinference.semconv.trace import SpanAttributes
from opentelemetry import context as otel_context
from opentelemetry import trace as trace_api
from opentelemetry.trace import StatusCode

from marie.agent.message import (
    ASSISTANT,
    CONTENT,
    DEFAULT_SYSTEM_MESSAGE,
    ROLE,
    SYSTEM,
    ContentItem,
    FunctionCall,
    Message,
    format_messages,
)
from marie.agent.tools.base import AgentTool, ToolOutput
from marie.agent.tools.registry import resolve_tools
from marie.instrumentation import MAX_FIELD_BYTES, start_span
from marie.logging_core.logger import MarieLogger

if TYPE_CHECKING:
    from marie.agent.tools.searchable import SearchableToolset

from marie.agent.cancellation import AbortSignal  # isort:skip circular with marie.agent
from marie.agent.streaming import StreamChunk  # isort:skip circular with marie.agent

if TYPE_CHECKING:
    from marie_mem0 import Mem0Config

    from marie.agent.emitter import Emitter
    from marie.agent.llm_wrapper import BaseLLMWrapper
    from marie.agent.middleware import MiddlewareList, RunMiddlewareProtocol
    from marie.agent.skills import Skill, SkillContext, SkillRouter

logger = MarieLogger("marie.agent.base")

# Pattern for explicit skill invocation: /skill-name [args]
SLASH_COMMAND_PATTERN = re.compile(
    r"^/([a-z0-9][a-z0-9-]*[a-z0-9]|[a-z0-9])(?:\s+(.*))?$"
)


def _message_debug_payload(message: Any) -> Dict[str, Any]:
    if hasattr(message, "model_dump"):
        return message.model_dump(mode="json")
    if isinstance(message, dict):
        return dict(message)
    return {
        "role": getattr(message, "role", None),
        "content": getattr(message, "content", None),
        "name": getattr(message, "name", None),
        "function_call": getattr(message, "function_call", None),
        "tool_calls": getattr(message, "tool_calls", None),
        "tool_call_id": getattr(message, "tool_call_id", None),
        "metadata": getattr(message, "metadata", None),
    }


def _tool_debug_payload(function_map: Dict[str, AgentTool]) -> List[Dict[str, Any]]:
    tools: List[Dict[str, Any]] = []
    for name, tool in sorted(function_map.items()):
        metadata = getattr(tool, "metadata", None)
        tools.append(
            {
                "name": name,
                "class": type(tool).__name__,
                "description": getattr(metadata, "description", None),
                "parameters": metadata.get_parameters_dict() if metadata else None,
                "return_direct": getattr(metadata, "return_direct", None),
            }
        )
    return tools


def _llm_debug_name(llm: Any) -> Optional[str]:
    if llm is None:
        return None
    return (
        getattr(llm, "model", None)
        or getattr(llm, "engine_name", None)
        or getattr(llm, "name", None)
        or type(llm).__name__
    )


class BaseAgent(ABC):
    """Abstract base class for Marie agents.

    Implements the template method pattern where `run()` handles normalization
    and `_run()` contains the core agent logic. This design follows Qwen-Agent
    architecture for consistency and extensibility.

    Subclasses must implement:
        - `_run()`: Core agent execution logic

    Example:
        ```python
        class MyAgent(BaseAgent):
            def _run(self, messages, lang="en", **kwargs):
                # Process messages and generate response
                for response in self._call_llm(messages):
                    yield [response]


        agent = MyAgent(
            llm=MarieEngineLLMWrapper(engine_name="qwen2_5_vl_7b"),
            function_list=["search", "calculator"],
            system_message="You are a helpful assistant.",
        )

        for responses in agent.run([{"role": "user", "content": "Hello"}]):
            print(responses[-1].content)
        ```
    """

    def __init__(
        self,
        function_list: Optional[List[Union[str, Dict, AgentTool, Callable]]] = None,
        llm: Optional["BaseLLMWrapper"] = None,
        system_message: Optional[str] = DEFAULT_SYSTEM_MESSAGE,
        name: Optional[str] = None,
        description: Optional[str] = None,
        memory: Optional["Mem0Config"] = None,
        skills_enabled: bool = True,
        auto_match_skills: bool = True,
        default_skills: Optional[List[str]] = None,
        middlewares: Optional["MiddlewareList"] = None,
        tools: Optional["SearchableToolset"] = None,
        **kwargs: Any,
    ):
        """Initialize the agent.

        Args:
            function_list: List of tools available to the agent. Can be:
                - Tool name strings (looked up from registry)
                - Configuration dicts with 'name' key
                - AgentTool instances
                - Callable functions
            llm: LLM wrapper for generating responses
            system_message: System message prepended to conversations
            name: Agent name (used in multi-agent scenarios)
            description: Agent description (used for delegation decisions)
            memory: Optional Mem0 configuration for memory integration
            skills_enabled: Enable skill routing (slash commands, auto-matching)
            auto_match_skills: Auto-match skills from message content
            default_skills: Skills to always load (by name)
            middlewares: Optional list of middleware to apply to runs
            tools: Optional SearchableToolset for dynamic tool discovery.
                When provided, enables BM25-based tool search instead of
                exposing all tools. See SearchableToolset documentation.
            **kwargs: Additional configuration
        """
        self.llm = llm
        self.system_message = system_message
        self.name = name
        self.description = description
        self.extra_generate_cfg: Dict[str, Any] = kwargs.get("extra_generate_cfg", {})
        self.middlewares: List["RunMiddlewareProtocol"] = list(middlewares or [])
        self._emitter: Optional["Emitter"] = None

        # Skill system configuration
        self.skills_enabled = skills_enabled
        self.auto_match_skills = auto_match_skills
        self.default_skills = default_skills or []
        self._skill_router: Optional["SkillRouter"] = None

        # Searchable toolset (passed directly, Haystack-style)
        self._searchable_toolset: Optional["SearchableToolset"] = tools

        # Track tools dirty state for schema refresh
        self._tools_dirty = False

        # Initialize memory
        self._mem0 = None
        if memory and memory.enabled:
            self._init_memory(memory)

        # Initialize tools
        self.function_map: Dict[str, AgentTool] = {}

        if self._searchable_toolset is not None:
            # Use SearchableToolset - bind it and populate function_map
            self._init_searchable_toolset()
        elif function_list:
            # Use traditional function_list
            self._init_tools(function_list)

        # Initialize skill router if enabled
        if self.skills_enabled:
            self._init_skills()

    def _init_tools(
        self,
        function_list: List[Union[str, Dict, AgentTool, Callable]],
    ) -> None:
        """Initialize tools from the function list.

        Args:
            function_list: List of tool specifications
        """
        resolved = resolve_tools(function_list)
        for name, tool in resolved.items():
            if name in self.function_map:
                logger.warning(
                    f"Repeatedly adding tool {name}, will use the newest tool"
                )
            self.function_map[name] = tool

    def _init_skills(self) -> None:
        """Initialize the skill routing system."""
        try:
            from marie.agent.skills import SKILL_REGISTRY, SkillRouter

            self._skill_router = SkillRouter(registry=SKILL_REGISTRY)
            logger.debug(f"Skill router initialized with {len(SKILL_REGISTRY)} skills")
        except ImportError:
            logger.debug("Skills module not available, skill routing disabled")
            self.skills_enabled = False
        except Exception as e:
            logger.warning(f"Failed to initialize skill router: {e}")
            self.skills_enabled = False

    def _init_searchable_toolset(self) -> None:
        """Initialize the searchable toolset for dynamic tool discovery."""
        from marie.agent.tools.searchable import SearchableToolset

        if self._searchable_toolset is None:
            return

        # Bind the toolset to this agent
        self._searchable_toolset.bind(
            register_callback=self._register_discovered_tool,
            tools_dirty_callback=self._mark_tools_dirty,
        )

        # Populate function_map with all tools from the toolset
        # This allows tool execution even before discovery
        self.function_map = self._searchable_toolset.get_all_tools()

        logger.debug(
            f"SearchableToolset bound with {self._searchable_toolset.tool_count} tools "
            f"(passthrough={self._searchable_toolset.is_passthrough})"
        )

    def _register_discovered_tool(self, tool: AgentTool) -> None:
        """Callback to register a tool discovered via search_tools.

        This is called by SearchableToolset when a tool is discovered.
        The tool is added to function_map so it can be executed.

        Args:
            tool: Tool to register for execution
        """
        if tool.metadata.name not in self.function_map:
            self.function_map[tool.metadata.name] = tool
            logger.debug(f"Registered discovered tool: {tool.metadata.name}")

    def _mark_tools_dirty(self) -> None:
        """Mark tools as dirty to trigger schema refresh."""
        self._tools_dirty = True

    @property
    def emitter(self) -> Optional["Emitter"]:
        """Get or create the agent's event emitter.

        The emitter is lazily created on first access with namespace "agent".
        """
        if self._emitter is None:
            from marie.agent.emitter import Emitter, EmitterOptions

            namespace = f"agent.{self.name}" if self.name else "agent"
            self._emitter = Emitter(EmitterOptions(namespace=namespace))
        return self._emitter

    def _bind_middlewares(self, emitter: "Emitter") -> None:
        """Bind all middlewares to the emitter.

        Middlewares are sorted by priority (highest first) before binding.
        """
        sorted_middlewares = sorted(
            self.middlewares,
            key=lambda m: -m.priority,
        )
        for middleware in sorted_middlewares:
            middleware.bind(emitter)

    def _parse_slash_command(self, message: str) -> tuple[Optional[str], str]:
        """Extract /skill-name from message.

        Args:
            message: User message

        Returns:
            Tuple of (skill_name, remaining_message) or (None, original_message)
        """
        message = message.strip()
        match = SLASH_COMMAND_PATTERN.match(message)

        if match:
            skill_name = match.group(1)
            remaining = match.group(2) or ""
            return skill_name, remaining.strip()

        return None, message

    async def _route_to_skill(
        self,
        message: str,
        explicit_skill: Optional[str] = None,
    ) -> Optional["SkillContext"]:
        """Route message to a skill if applicable.

        Args:
            message: User message
            explicit_skill: Explicitly requested skill name

        Returns:
            SkillContext if skill matched, None otherwise
        """
        if not self.skills_enabled or not self._skill_router:
            return None

        # Check for explicit skill first
        if explicit_skill:
            context = await self._skill_router.route(
                message,
                explicit_skill=explicit_skill,
                auto_match=False,
            )
            return context if context.has_skill else None

        # Check for /skill-name in message
        parsed_skill, _ = self._parse_slash_command(message)
        if parsed_skill:
            context = await self._skill_router.route(
                message,
                explicit_skill=parsed_skill,
                auto_match=False,
            )
            return context if context.has_skill else None

        # Auto-match if enabled
        if self.auto_match_skills:
            context = await self._skill_router.route(
                message,
                auto_match=True,
            )
            return context if context.has_skill else None

        return None

    def _build_skill_system_prompt(self, skill: "Skill") -> str:
        """Build enhanced system prompt with skill instructions.

        Args:
            skill: Active skill

        Returns:
            Combined system prompt with skill context
        """
        base_system = self.system_message or ""
        skill_injection = skill.to_system_prompt_injection()

        return f"{base_system}\n\n{skill_injection}"

    def _resolve_skill_tools(self, skill: "Skill") -> Dict[str, AgentTool]:
        """Resolve tools specified in skill's allowed-tools.

        Args:
            skill: Skill with tool requirements

        Returns:
            Dict of tool name to AgentTool
        """
        if not skill.metadata.allowed_tools:
            return {}

        resolved: Dict[str, AgentTool] = {}
        unresolved: List[Union[str, Dict[str, Any], AgentTool, Callable]] = []
        for spec in skill.metadata.allowed_tools:
            if isinstance(spec, str) and spec in self.function_map:
                resolved[spec] = self.function_map[spec]
                continue
            if isinstance(spec, dict):
                name = spec.get("name")
                if isinstance(name, str) and name in self.function_map:
                    resolved[name] = self.function_map[name]
                    continue
            unresolved.append(spec)

        if unresolved:
            resolved.update(resolve_tools(unresolved))
        return resolved

    def _init_memory(self, memory_config: "Mem0Config") -> None:
        """Initialize Mem0 memory integration.

        Args:
            memory_config: Mem0 configuration
        """
        try:
            from marie_mem0 import Mem0Memory

            self._mem0 = Mem0Memory(memory_config)
            logger.info("Mem0 memory integration initialized")
        except ImportError:
            logger.warning(
                "marie-mem0 package not installed, memory integration disabled. "
                "Install with: uv add marie-mem0"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Mem0 memory: {e}")

    def _augment_with_memories(
        self,
        messages: List[Message],
        user_id: str,
        agent_id: Optional[str] = None,
        limit: int = 5,
    ) -> List[Message]:
        """Search and prepend relevant memories to context.

        Args:
            messages: Current conversation messages
            user_id: User identifier for memory scoping
            agent_id: Optional agent identifier
            limit: Maximum memories to retrieve

        Returns:
            Messages with memory context prepended
        """
        if not self._mem0 or not self._mem0.is_enabled:
            return messages

        # Get last user message as query
        query = ""
        for msg in reversed(messages):
            if msg.role == "user":
                query = msg.text_content or ""
                break

        if not query:
            return messages

        # Search for relevant memories
        memories = self._mem0.search(
            query=query,
            user_id=user_id,
            agent_id=agent_id,
            limit=limit,
        )

        if not memories:
            return messages

        # Format memories as context
        memory_lines = []
        for m in memories:
            memory_text = m.get("memory", "")
            if memory_text:
                memory_lines.append(f"- {memory_text}")

        if not memory_lines:
            return messages

        memory_context = "\n".join(memory_lines)

        # Create memory context message
        memory_msg = Message(
            role=SYSTEM,
            content=f"Relevant memories from previous interactions:\n{memory_context}",
        )

        # Insert after the first system message or at the beginning
        result = list(messages)
        insert_idx = 0
        if result and result[0].role == SYSTEM:
            insert_idx = 1

        result.insert(insert_idx, memory_msg)
        return result

    def _store_interaction(
        self,
        messages: List[Message],
        response: str,
        user_id: str,
        agent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store the interaction in memory.

        Args:
            messages: Conversation messages
            response: Agent response content
            user_id: User identifier for memory scoping
            agent_id: Optional agent identifier
            metadata: Optional metadata to attach
        """
        if not self._mem0 or not self._mem0.is_enabled:
            return

        # Convert messages to dict format for mem0
        msg_dicts = []
        for msg in messages:
            if msg.role != SYSTEM:  # Skip system messages
                msg_dicts.append(
                    {
                        "role": msg.role,
                        "content": msg.text_content or "",
                    }
                )

        # Add the assistant response
        msg_dicts.append({"role": ASSISTANT, "content": response})

        # Store in memory
        self._mem0.add(
            messages=msg_dicts,
            user_id=user_id,
            agent_id=agent_id or self.name,
            metadata=metadata,
        )

    def add_tool(self, tool: Union[str, Dict, AgentTool, Callable]) -> None:
        """Add a tool to the agent.

        Note: If using SearchableToolset, adding tools at runtime is not
        recommended as the BM25 index won't be updated. Create a new
        SearchableToolset with all tools instead.

        Args:
            tool: Tool specification (name, config, instance, or callable)
        """
        self._init_tools([tool])
        self._tools_dirty = True

    def remove_tool(self, name: str) -> bool:
        """Remove a tool from the agent.

        Note: If using SearchableToolset, removing tools at runtime is not
        recommended. Create a new SearchableToolset without the tool instead.

        Args:
            name: Tool name to remove

        Returns:
            True if tool was removed, False if not found
        """
        if name in self.function_map:
            del self.function_map[name]
            self._tools_dirty = True
            return True
        return False

    def _get_exposed_tools(self) -> List[AgentTool]:
        """Get tools to expose to the LLM.

        When searchable toolset is enabled and not in passthrough mode,
        returns only the search_tools meta-function. Otherwise returns
        all tools in function_map.

        Returns:
            List of tools to include in LLM prompt
        """
        if self._searchable_toolset and not self._searchable_toolset.is_passthrough:
            return self._searchable_toolset.get_exposed_tools()
        return list(self.function_map.values())

    def run(
        self,
        messages: List[Union[Dict, Message]],
        skill_name: Optional[str] = None,
        **kwargs: Any,
    ) -> Iterator[List[Message]]:
        """Execute the agent with the given messages.

        This is the public entry point that normalizes input and delegates
        to `_run()` for core logic. Supports skill routing via explicit
        skill_name parameter or /skill-name slash commands in messages.

        Emits events via the emitter if middlewares are configured:
        - agent.start: When execution begins
        - agent.success: When execution completes successfully
        - agent.error: When an error occurs
        - agent.finish: Always emitted when execution ends

        Args:
            messages: Input messages (can be dicts or Message objects)
            skill_name: Explicit skill to use (bypasses auto-matching)
            **kwargs: Additional arguments passed to `_run()`

        Yields:
            Lists of response Messages (streaming, yields partial results)

        Example:
            ```python
            # Normal execution
            for responses in agent.run([{"role": "user", "content": "Hello"}]):
                print(responses[-1].content)

            # With explicit skill
            for responses in agent.run(messages, skill_name="document-extraction"):
                print(responses[-1].content)

            # With slash command (skill parsed from message)
            for responses in agent.run([{"role": "user", "content": "/code-review main.py"}]):
                print(responses[-1].content)

            # With middleware
            from marie.agent.middleware.trajectory import TrajectoryMiddleware

            agent = MyAgent(middlewares=[TrajectoryMiddleware()])
            for responses in agent.run(messages):
                print(responses[-1].content)
            ```
        """
        import time

        from marie.agent.emitter import emit_sync

        start_time = time.perf_counter()

        # Deep copy to avoid mutation
        messages = copy.deepcopy(messages)

        # Track original message types for return format
        _return_message_type = "dict"
        new_messages: List[Message] = []

        if not messages:
            _return_message_type = "message"
        else:
            for msg in messages:
                if isinstance(msg, dict):
                    new_messages.append(Message(**msg))
                else:
                    new_messages.append(msg)
                    _return_message_type = "message"

        # Determine system message (may be enhanced by skill)
        system_message = self.system_message
        skill_tools: Dict[str, AgentTool] = {}

        # Skill routing (if enabled)
        if self.skills_enabled and new_messages:
            # Get last user message for skill matching
            user_message = ""
            for msg in reversed(new_messages):
                if msg.role == "user":
                    user_message = msg.text_content or ""
                    break

            if user_message:
                # Check for explicit skill or slash command
                explicit_skill = skill_name
                if not explicit_skill:
                    parsed_skill, _ = self._parse_slash_command(user_message)
                    explicit_skill = parsed_skill

                # Try to route to skill
                skill_context = self._sync_route_to_skill(
                    user_message,
                    explicit_skill=explicit_skill,
                )

                if skill_context and skill_context.has_skill:
                    skill = skill_context.skill
                    logger.debug(
                        f"Using skill '{skill.name}' "
                        f"(explicit={skill_context.explicit_invocation})"
                    )

                    # Enhance system message with skill instructions
                    system_message = self._build_skill_system_prompt(skill)

                    # Resolve skill tools
                    skill_tools = self._resolve_skill_tools(skill)

        # Prepend system message
        if system_message:
            if not new_messages or new_messages[0][ROLE] != SYSTEM:
                new_messages.insert(0, Message(role=SYSTEM, content=system_message))
            else:
                # Merge with existing system message
                existing_content = new_messages[0][CONTENT]
                if isinstance(existing_content, str):
                    new_messages[0][CONTENT] = (
                        system_message + "\n\n" + existing_content
                    )
                elif isinstance(existing_content, list):
                    new_messages[0][CONTENT] = [
                        ContentItem(text=system_message + "\n\n")
                    ] + existing_content

        # Save original tools for cleanup (always, for both skill and searchable tools)
        original_tools = self.function_map.copy()

        # Clear searchable toolset tracking for this request
        if self._searchable_toolset:
            self._searchable_toolset.clear_dynamic_tools()

        # Merge skill tools with agent tools
        if skill_tools:
            self.function_map.update(skill_tools)

        # Reset tools dirty flag for this run
        self._tools_dirty = False

        # Set up emitter and bind middleware
        run_emitter = None
        if self.middlewares:
            run_emitter = self.emitter
            self._bind_middlewares(run_emitter)

        # Emit start event
        emit_sync(
            run_emitter,
            "agent.start",
            {
                "agent_name": self.name,
                "message_count": len(new_messages),
            },
            source=self.name,
        )
        emit_sync(
            run_emitter,
            "agent.input",
            {
                "agent_name": self.name,
                "model_name": _llm_debug_name(self.llm),
                "messages": [
                    _message_debug_payload(message) for message in new_messages
                ],
                "tools": _tool_debug_payload(self.function_map),
            },
            source=self.name,
        )

        # OTel AGENT span — manual lifecycle because run() is a generator
        _otel_tracer = trace_api.get_tracer("marie.agent")
        _otel_span = start_span(
            _otel_tracer,
            f"agent:{self.name}",
            span_kind="agent",
        )
        _otel_span.set_attribute(SpanAttributes.AGENT_NAME, self.name or "")
        _session_id = kwargs.get("session_id")
        _user_id = kwargs.get("user_id")
        if _session_id:
            _otel_span.set_attribute(SpanAttributes.SESSION_ID, _session_id)
        if _user_id:
            _otel_span.set_attribute(SpanAttributes.USER_ID, _user_id)

        # Capture agent input: last user message as query preview
        _agent_input = {"agent": self.name, "session_id": _session_id}
        if new_messages:
            _last_msg = new_messages[-1]
            _content = (
                _last_msg.get("content")
                if isinstance(_last_msg, dict)
                else getattr(_last_msg, "content", None)
            )
            if isinstance(_content, str):
                _agent_input["query"] = _content[:MAX_FIELD_BYTES]
        _otel_span.set_input(_agent_input)

        _otel_span_token = otel_context.attach(
            trace_api.set_span_in_context(_otel_span)
        )

        last_responses: List[Message] = []
        response_iteration = 0
        success = False
        error_exc: Optional[Exception] = None

        try:
            # Execute core logic
            for responses in self._run(messages=new_messages, **kwargs):
                # Set agent name on responses
                for resp in responses:
                    if not resp.name and self.name:
                        resp.name = self.name

                last_responses = responses
                response_iteration += 1
                emit_sync(
                    run_emitter,
                    "agent.response",
                    {
                        "agent_name": self.name,
                        "iteration": response_iteration,
                        "messages": [
                            _message_debug_payload(response) for response in responses
                        ],
                    },
                    source=self.name,
                )

                # Convert output format based on input format
                if _return_message_type == "message":
                    yield [
                        Message(**r) if isinstance(r, dict) else r for r in responses
                    ]
                else:
                    yield [
                        r.model_dump() if isinstance(r, Message) else r
                        for r in responses
                    ]

            success = True
            # Capture agent output summary
            _output_summary = {"result_count": len(last_responses)}
            if last_responses:
                _last = last_responses[-1]
                _resp_content = (
                    _last.get("content")
                    if isinstance(_last, dict)
                    else getattr(_last, "content", None)
                )
                if isinstance(_resp_content, str):
                    _output_summary["response_preview"] = _resp_content[
                        :MAX_FIELD_BYTES
                    ]
            _otel_span.set_output(_output_summary)
            _otel_span.set_status(StatusCode.OK)

            # Emit success event
            duration_ms = (time.perf_counter() - start_time) * 1000
            emit_sync(
                run_emitter,
                "agent.success",
                {
                    "agent_name": self.name,
                    "result_count": len(last_responses),
                    "duration_ms": duration_ms,
                },
                source=self.name,
            )

        except GeneratorExit:
            # Consumer stopped iterating — clean exit, not an error
            success = True
            _otel_span.set_status(StatusCode.OK)

        except Exception as e:
            error_exc = e
            _otel_span.set_status(StatusCode.ERROR, str(e))
            _otel_span.record_exception(e)
            # Emit error event
            emit_sync(
                run_emitter,
                "agent.error",
                {
                    "agent_name": self.name,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                },
                source=self.name,
            )
            raise

        finally:
            otel_context.detach(_otel_span_token)
            _otel_span.end()

            # Emit finish event
            duration_ms = (time.perf_counter() - start_time) * 1000
            emit_sync(
                run_emitter,
                "agent.finish",
                {
                    "agent_name": self.name,
                    "success": success,
                    "duration_ms": duration_ms,
                },
                source=self.name,
            )

            # Always restore original tools to prevent leakage across requests
            # This handles both skill tools and dynamically discovered tools
            self.function_map = original_tools
            self._tools_dirty = False

    def _sync_route_to_skill(
        self,
        message: str,
        explicit_skill: Optional[str] = None,
    ) -> Optional["SkillContext"]:
        """Synchronous skill routing wrapper.

        Args:
            message: User message
            explicit_skill: Explicitly requested skill name

        Returns:
            SkillContext if skill matched, None otherwise
        """
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Already in async context, use thread
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(
                        asyncio.run, self._route_to_skill(message, explicit_skill)
                    )
                    return future.result(timeout=5.0)
            else:
                return loop.run_until_complete(
                    self._route_to_skill(message, explicit_skill)
                )
        except Exception as e:
            logger.debug(f"Skill routing failed: {e}")
            return None

    def run_nonstream(
        self,
        messages: List[Union[Dict, Message]],
        **kwargs: Any,
    ) -> List[Message]:
        """Execute the agent and return the final response.

        Same as `run()` but returns only the final result instead of streaming.

        Args:
            messages: Input messages
            **kwargs: Additional arguments

        Returns:
            Final list of response Messages
        """
        *_, last_responses = self.run(messages, **kwargs)
        return last_responses

    @abstractmethod
    def _run(
        self,
        messages: List[Message],
        lang: str = "en",
        **kwargs: Any,
    ) -> Iterator[List[Message]]:
        """Core agent execution logic.

        Subclasses must implement this method to define the agent's behavior.

        Args:
            messages: Normalized list of Messages with system message prepended
            lang: Language code ('en' or 'zh')
            **kwargs: Additional arguments

        Yields:
            Lists of response Messages
        """
        raise NotImplementedError

    def _call_llm(
        self,
        messages: List[Message],
        functions: Optional[List[Dict]] = None,
        stream: bool = False,
        extra_generate_cfg: Optional[Dict[str, Any]] = None,
    ) -> Iterator[List[Message]]:
        """Call the LLM with the given messages.

        Args:
            messages: Messages to send to the LLM
            functions: Optional function definitions for function calling
            stream: Whether to stream the response (not yet implemented)
            extra_generate_cfg: Additional generation configuration

        Yields:
            LLM response Messages

        Raises:
            ValueError: If LLM is not configured

        Note:
            Streaming is not yet implemented. Responses are returned complete.
        """
        if self.llm is None:
            raise ValueError("LLM is not configured for this agent")

        # Merge generation configs
        merged_cfg = {**self.extra_generate_cfg}
        if extra_generate_cfg:
            merged_cfg.update(extra_generate_cfg)

        import time

        from marie.agent.emitter import emit_sync

        run_emitter = self.emitter if self.middlewares else None
        model_name = _llm_debug_name(self.llm)
        start_time = time.perf_counter()
        emit_sync(
            run_emitter,
            "llm.start",
            {
                "model_name": model_name,
                "message_count": len(messages),
                "tool_count": len(functions or []),
                "stream": stream,
                "extra_generate_cfg": merged_cfg,
            },
            source="llm",
        )

        try:
            response_iter = self.llm.chat(
                messages=messages,
                functions=functions,
                stream=stream,
                extra_generate_cfg=merged_cfg,
            )
        except Exception as e:
            logger.error(f"LLM call failed: {type(e).__name__}: {e}")
            duration_ms = (time.perf_counter() - start_time) * 1000
            emit_sync(
                run_emitter,
                "llm.error",
                {
                    "model_name": model_name,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "duration_ms": duration_ms,
                },
                source="llm",
            )
            emit_sync(
                run_emitter,
                "llm.finish",
                {
                    "model_name": model_name,
                    "success": False,
                    "duration_ms": duration_ms,
                },
                source="llm",
            )
            # Yield an error message so the agent can handle it gracefully
            error_msg = Message.assistant(
                f"I encountered an error while processing: {type(e).__name__}. "
                "Please try again or rephrase your request."
            )

            # Return a generator that yields the error message
            def error_generator():
                yield [error_msg]

            return error_generator()

        def instrumented_generator() -> Iterator[List[Message]]:
            response_count = 0
            has_tool_calls = False
            success = False
            try:
                for responses in response_iter:
                    response_count += len(responses)
                    has_tool_calls = has_tool_calls or any(
                        bool(getattr(response, "tool_calls", None))
                        for response in responses
                    )
                    yield responses
                success = True
                duration_ms = (time.perf_counter() - start_time) * 1000
                emit_sync(
                    run_emitter,
                    "llm.success",
                    {
                        "model_name": model_name,
                        "response_count": response_count,
                        "has_tool_calls": has_tool_calls,
                        "duration_ms": duration_ms,
                    },
                    source="llm",
                )
            except Exception as e:
                logger.error(f"LLM call failed: {type(e).__name__}: {e}")
                duration_ms = (time.perf_counter() - start_time) * 1000
                emit_sync(
                    run_emitter,
                    "llm.error",
                    {
                        "model_name": model_name,
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "duration_ms": duration_ms,
                    },
                    source="llm",
                )
                error_msg = Message.assistant(
                    f"I encountered an error while processing: {type(e).__name__}. "
                    "Please try again or rephrase your request."
                )
                yield [error_msg]
            finally:
                duration_ms = (time.perf_counter() - start_time) * 1000
                emit_sync(
                    run_emitter,
                    "llm.finish",
                    {
                        "model_name": model_name,
                        "success": success,
                        "duration_ms": duration_ms,
                    },
                    source="llm",
                )

        return instrumented_generator()

    async def _call_llm_stream(
        self,
        messages: List[Message],
        functions: Optional[List[Dict]] = None,
        abort_signal: Optional["AbortSignal"] = None,
        extra_generate_cfg: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator["StreamChunk", None]:
        """Stream LLM response as chunks.

        Requires the LLM wrapper to implement ``achat_stream()``.
        Falls back gracefully if the wrapper only has the default implementation.

        Args:
            messages: Messages to send to the LLM
            functions: Optional function definitions
            abort_signal: Optional cancellation signal
            extra_generate_cfg: Additional generation configuration

        Yields:
            StreamChunk deltas
        """

        if self.llm is None:
            raise ValueError("LLM is not configured for this agent")

        merged_cfg = {**self.extra_generate_cfg}
        if extra_generate_cfg:
            merged_cfg.update(extra_generate_cfg)

        try:
            async for chunk in self.llm.achat_stream(
                messages=messages,
                functions=functions,
                abort_signal=abort_signal,
                extra_generate_cfg=merged_cfg,
            ):
                yield chunk
        except Exception as e:
            logger.error(f"LLM stream failed: {type(e).__name__}: {e}")
            yield StreamChunk.error(str(e))

    def _call_tool(
        self,
        tool_name: str,
        tool_args: Union[str, Dict] = "{}",
        **kwargs: Any,
    ) -> Union[str, List[ContentItem]]:
        """Execute a tool by name.

        Args:
            tool_name: Name of the tool to call
            tool_args: Arguments for the tool (string or dict)
            **kwargs: Additional arguments passed to the tool

        Returns:
            Tool output (string or list of ContentItems for multimodal)
        """
        import time

        from marie.agent.emitter import emit_sync

        run_emitter = self.emitter if self.middlewares else None
        start_time = time.perf_counter()
        emit_sync(
            run_emitter,
            "tool.start",
            {"tool_name": tool_name, "arguments": tool_args},
            source=tool_name,
        )
        if tool_name not in self.function_map:
            message = f"Tool '{tool_name}' does not exist."
            duration_ms = (time.perf_counter() - start_time) * 1000
            emit_sync(
                run_emitter,
                "tool.error",
                {"tool_name": tool_name, "error": message, "duration_ms": duration_ms},
                source=tool_name,
            )
            emit_sync(
                run_emitter,
                "tool.finish",
                {"tool_name": tool_name, "success": False, "duration_ms": duration_ms},
                source=tool_name,
            )
            return message

        tool = self.function_map[tool_name]
        result = tool.safe_call(tool_args, **kwargs)
        duration_ms = (time.perf_counter() - start_time) * 1000

        if result.is_error:
            logger.warning(f"Tool '{tool_name}' failed: {result.content}")
            emit_sync(
                run_emitter,
                "tool.error",
                {
                    "tool_name": tool_name,
                    "error": result.content,
                    "duration_ms": duration_ms,
                },
                source=tool_name,
            )
        else:
            emit_sync(
                run_emitter,
                "tool.success",
                {
                    "tool_name": tool_name,
                    "result_length": len(str(result.content)),
                    "result": result.content,
                    "duration_ms": duration_ms,
                },
                source=tool_name,
            )
        emit_sync(
            run_emitter,
            "tool.finish",
            {
                "tool_name": tool_name,
                "success": not result.is_error,
                "duration_ms": duration_ms,
            },
            source=tool_name,
        )

        return result.content

    async def _acall_tool(
        self,
        tool_name: str,
        tool_args: Union[str, Dict] = "{}",
        **kwargs: Any,
    ) -> Union[str, List[ContentItem]]:
        """Execute a tool asynchronously.

        Args:
            tool_name: Name of the tool to call
            tool_args: Arguments for the tool
            **kwargs: Additional arguments

        Returns:
            Tool output
        """
        import time

        run_emitter = self.emitter if self.middlewares else None
        start_time = time.perf_counter()
        if run_emitter is not None:
            await run_emitter.emit(
                "tool.start",
                {"tool_name": tool_name, "arguments": tool_args},
                source=tool_name,
            )
        if tool_name not in self.function_map:
            message = f"Tool '{tool_name}' does not exist."
            duration_ms = (time.perf_counter() - start_time) * 1000
            if run_emitter is not None:
                await run_emitter.emit(
                    "tool.error",
                    {
                        "tool_name": tool_name,
                        "error": message,
                        "duration_ms": duration_ms,
                    },
                    source=tool_name,
                )
                await run_emitter.emit(
                    "tool.finish",
                    {
                        "tool_name": tool_name,
                        "success": False,
                        "duration_ms": duration_ms,
                    },
                    source=tool_name,
                )
            return message

        tool = self.function_map[tool_name]
        result = await tool.safe_acall(tool_args, **kwargs)
        duration_ms = (time.perf_counter() - start_time) * 1000

        if result.is_error:
            logger.warning(f"Tool '{tool_name}' failed: {result.content}")
            if run_emitter is not None:
                await run_emitter.emit(
                    "tool.error",
                    {
                        "tool_name": tool_name,
                        "error": result.content,
                        "duration_ms": duration_ms,
                    },
                    source=tool_name,
                )
        elif run_emitter is not None:
            await run_emitter.emit(
                "tool.success",
                {
                    "tool_name": tool_name,
                    "result_length": len(str(result.content)),
                    "result": result.content,
                    "duration_ms": duration_ms,
                },
                source=tool_name,
            )
        if run_emitter is not None:
            await run_emitter.emit(
                "tool.finish",
                {
                    "tool_name": tool_name,
                    "success": not result.is_error,
                    "duration_ms": duration_ms,
                },
                source=tool_name,
            )

        return result.content

    def _detect_tool_call(
        self, message: Message
    ) -> Tuple[bool, str, str, str, Optional[str]]:
        """Detect if a message contains a tool/function call.

        Args:
            message: Message to analyze

        Returns:
            Tuple of (has_call, tool_name, tool_args, text_content, tool_call_id)
        """
        func_name: Optional[str] = None
        func_args: Optional[str] = None
        tool_call_id: Optional[str] = None

        # Check legacy function_call format
        if message.function_call:
            func_name = message.function_call.name
            func_args = message.function_call.get_arguments_str()
        # Check newer tool_calls format (OpenAI)
        elif message.tool_calls and len(message.tool_calls) > 0:
            tool_call = message.tool_calls[0]
            if isinstance(tool_call, dict):
                tool_call_id = tool_call.get("id")
                func_info = tool_call.get("function", {})
                func_name = func_info.get("name")
                args = func_info.get("arguments", {})
                # arguments can be dict or string
                if isinstance(args, dict):
                    import json

                    func_args = json.dumps(args)
                else:
                    func_args = str(args)
            else:
                # Handle object-style tool_call
                tool_call_id = getattr(tool_call, "id", None)
                func_name = getattr(tool_call.function, "name", None)
                func_args = getattr(tool_call.function, "arguments", "{}")

        text = message.text_content or ""

        return (
            (func_name is not None),
            func_name or "",
            func_args or "{}",
            text,
            tool_call_id,
        )

    def _get_tool_definitions(
        self,
        use_exposed: bool = True,
    ) -> List[Dict[str, Any]]:
        """Get OpenAI-compatible function definitions for tools.

        Args:
            use_exposed: If True and searchable toolset is active, returns
                only exposed tools (search_tools in non-passthrough mode).
                If False, returns all tools in function_map.

        Returns:
            List of function definitions
        """
        if use_exposed:
            tools = self._get_exposed_tools()
        else:
            tools = list(self.function_map.values())
        return [tool.get_function_definition() for tool in tools]

    def _get_tool_definitions_openai(
        self,
        use_exposed: bool = True,
    ) -> List[Dict[str, Any]]:
        """Get OpenAI tool format definitions.

        Args:
            use_exposed: If True and searchable toolset is active, returns
                only exposed tools. If False, returns all tools.

        Returns:
            List of tool definitions in OpenAI format
        """
        if use_exposed:
            tools = self._get_exposed_tools()
        else:
            tools = list(self.function_map.values())
        return [tool.to_openai_tool() for tool in tools]

    def _check_tools_dirty(self) -> bool:
        """Check if tools have been modified and need schema refresh.

        Call this in agent loops before making LLM calls. If True,
        recompute tool schemas for the next LLM call.

        Returns:
            True if tools have changed since last check
        """
        dirty = self._tools_dirty
        if dirty:
            self._tools_dirty = False
        return dirty


class BasicAgent(BaseAgent):
    """Simple agent that just calls the LLM without tools.

    The most basic form of an agent - passes messages directly to the LLM
    without any tool augmentation or complex workflows.

    Example:
        ```python
        agent = BasicAgent(
            llm=MarieEngineLLMWrapper(engine_name="qwen2_5_vl_7b"),
            system_message="You are a helpful assistant.",
        )

        for responses in agent.run([{"role": "user", "content": "Hello"}]):
            print(responses[-1].content)
        ```
    """

    def _run(
        self,
        messages: List[Message],
        lang: str = "en",
        **kwargs: Any,
    ) -> Iterator[List[Message]]:
        """Simply forward messages to the LLM.

        Args:
            messages: Input messages
            lang: Language code
            **kwargs: Additional arguments (seed, etc.)

        Yields:
            LLM responses
        """
        extra_generate_cfg = {"lang": lang}
        if kwargs.get("seed") is not None:
            extra_generate_cfg["seed"] = kwargs["seed"]

        return self._call_llm(messages, extra_generate_cfg=extra_generate_cfg)
