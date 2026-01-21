---
sidebar_position: 7
---

# API Reference

Complete API reference for the Marie-AI Agent Framework.

## Agents

### BaseAgent

Abstract base class for all agents.

```python
from marie.agent.base import BaseAgent

class BaseAgent:
    def __init__(
        self,
        llm: BaseLLMWrapper,
        function_list: List[Union[str, AgentTool, Callable]] = None,
        name: str = "Agent",
        description: str = "",
        system_message: str = "",
        max_iterations: int = 10,
        **kwargs
    ):
        """
        Initialize a base agent.

        Args:
            llm: LLM backend wrapper
            function_list: List of tools (names, instances, or callables)
            name: Agent name for identification
            description: Agent description
            system_message: System prompt for the LLM
            max_iterations: Maximum tool-call iterations
        """

    def run(
        self,
        messages: List[Union[Dict, Message]],
        **kwargs
    ) -> Generator[List[Message], None, None]:
        """
        Run the agent on a conversation.

        Args:
            messages: Conversation history

        Yields:
            List of response messages
        """

    def _run(self, messages: List[Message], **kwargs) -> Generator:
        """Abstract method for agent-specific logic."""
        raise NotImplementedError

    def _call_tool(
        self,
        tool_name: str,
        tool_args: Union[str, Dict]
    ) -> ToolOutput:
        """Execute a tool by name."""
```

### ReactAgent

General-purpose ReAct-style agent.

```python
from marie.agent import ReactAgent

class ReactAgent(BaseAgent):
    """
    Agent using ReAct pattern for reasoning and acting.

    Features:
    - Multi-step tool calling
    - Text-based tool call detection
    - Configurable iteration limits
    """

    def _run(self, messages: List[Message], **kwargs):
        """
        Execute ReAct loop:
        1. Call LLM with tools
        2. Detect tool calls in response
        3. Execute tools and add results
        4. Repeat until no tool call or max iterations
        """
```

### PlanAndExecuteAgent

Agent with explicit planning phase.

```python
from marie.agent.agents import PlanAndExecuteAgent

class PlanAndExecuteAgent(BaseAgent):
    """
    Agent that creates plans before execution.

    Expected response format:
    PLAN:
    1. Step one
    2. Step two

    STEP 1: [action]
    ...

    FINAL ANSWER:
    [summary]
    """
```

### ChatAgent

Simple conversational agent without tools.

```python
from marie.agent.agents import ChatAgent

class ChatAgent(BaseAgent):
    """Pure conversational agent, no tool support."""
```

### VisionDocumentAgent

Document understanding specialist.

```python
from marie.agent.agents import VisionDocumentAgent

class VisionDocumentAgent(BaseAgent):
    """
    Specialized for visual document understanding.

    Features:
    - Task categorization
    - Tool suggestion
    - Pattern-based fallbacks
    """

    def get_task_info(self, task: str) -> Dict:
        """
        Analyze task and suggest tools.

        Returns:
            {
                "category": str,
                "suggested_tools": List[str],
                "available_tools": List[str],
                "pattern": Optional[str]
            }
        """
```

### Router

Multi-agent task router.

```python
from marie.agent.agents.router import Router

class Router(BaseAgent):
    """
    Routes tasks to specialized sub-agents.

    Args:
        agents: Dict[str, BaseAgent] - name to agent mapping
    """

    def __init__(
        self,
        llm: BaseLLMWrapper,
        agents: Dict[str, BaseAgent],
        name: str = "Router",
        system_message: str = "",
        **kwargs
    ):
        """Initialize router with sub-agents."""
```

## Messages

### Message

Unified message format.

```python
from marie.agent import Message

class Message(BaseModel):
    role: str  # "system", "user", "assistant", "tool", "function"
    content: Union[str, List[ContentItem]] = ""
    name: Optional[str] = None
    function_call: Optional[FunctionCall] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None

    def model_dump(self) -> Dict:
        """Convert to dictionary."""

    @classmethod
    def from_dict(cls, data: Dict) -> "Message":
        """Create from dictionary."""
```

### ContentItem

Multimodal content item.

```python
from marie.agent.message import ContentItem, ContentItemType

class ContentItem(BaseModel):
    text: Optional[str] = None
    image: Optional[str] = None  # path, URL, or base64
    file: Optional[str] = None
    audio: Optional[str] = None
    video: Optional[str] = None

    @property
    def type(self) -> ContentItemType:
        """Get content type."""
```

### ToolCall

Tool call representation.

```python
from marie.agent.message import ToolCall, FunctionCall

class FunctionCall(BaseModel):
    name: str
    arguments: str  # JSON string

class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: FunctionCall
```

## Tools

### AgentTool

Base class for tools.

```python
from marie.agent import AgentTool, ToolMetadata, ToolOutput

class AgentTool(ABC):
    """Abstract base class for agent tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name."""

    @property
    @abstractmethod
    def metadata(self) -> ToolMetadata:
        """Tool metadata including schema."""

    @abstractmethod
    def call(self, **kwargs) -> ToolOutput:
        """Execute the tool synchronously."""

    async def acall(self, **kwargs) -> ToolOutput:
        """Execute the tool asynchronously."""

    def safe_call(self, args: Union[str, Dict]) -> ToolOutput:
        """
        Safe execution with input validation.

        Args:
            args: Tool arguments (JSON string or dict)

        Returns:
            ToolOutput with is_error=True on failure
        """
```

### ToolMetadata

Tool interface description.

```python
from marie.agent import ToolMetadata

class ToolMetadata(BaseModel):
    name: str
    description: str  # Max 1024 chars for OpenAI
    fn_schema: Optional[Type[BaseModel]] = None  # Pydantic model
    parameters: Optional[Dict] = None  # JSON schema
    return_direct: bool = False

    def get_parameters_dict(self) -> Dict:
        """Get OpenAI-compatible parameters schema."""

    def to_openai_tool(self) -> Dict:
        """Convert to OpenAI tool format."""

    def to_openai_function(self) -> Dict:
        """Convert to OpenAI function format (legacy)."""
```

### ToolOutput

Tool execution result.

```python
from marie.agent.tools import ToolOutput

class ToolOutput(BaseModel):
    content: str  # String representation
    tool_name: str
    raw_input: Optional[Dict] = None
    raw_output: Any = None
    is_error: bool = False
```

### register_tool

Decorator for registering function-based tools.

```python
from marie.agent import register_tool

def register_tool(
    name_or_func: Union[str, Callable] = None,
    description: str = None
) -> Callable:
    """
    Register a function as a tool.

    Usage:
        @register_tool
        def my_tool(x: int) -> str:
            '''Tool description.'''
            return str(x)

        @register_tool("custom_name", description="Custom description")
        def another_tool(x: int) -> str:
            return str(x)
    """
```

### TOOL_REGISTRY

Global tool registry.

```python
from marie.agent.tools import TOOL_REGISTRY

class ToolRegistry:
    def register(
        self,
        name: str,
        tool: Union[AgentTool, Type[AgentTool], Callable],
        overwrite: bool = False
    ) -> None:
        """Register a tool."""

    def get(
        self,
        name: str,
        config: Dict = None
    ) -> AgentTool:
        """Get tool instance by name."""

    def has(self, name: str) -> bool:
        """Check if tool exists."""

    def list_tools(self) -> List[str]:
        """List all registered tool names."""

    def get_all_metadata(self) -> Dict[str, ToolMetadata]:
        """Get metadata for all tools."""

    def unregister(self, name: str) -> bool:
        """Remove a tool."""

    def clear(self) -> None:
        """Remove all tools."""
```

### resolve_tools

Resolve tool specifications to instances.

```python
from marie.agent.tools import resolve_tools

def resolve_tools(
    tools: List[Union[str, Dict, AgentTool, Callable]]
) -> Dict[str, AgentTool]:
    """
    Resolve tool specifications to AgentTool instances.

    Accepts:
    - str: Tool name from registry
    - Dict: {"name": "...", "config": {...}}
    - AgentTool: Direct instance
    - Callable: Function to wrap

    Returns:
        Dict mapping tool names to instances
    """
```

## LLM Wrappers

### BaseLLMWrapper

Abstract base for LLM backends.

```python
from marie.agent.llm_wrapper import BaseLLMWrapper

class BaseLLMWrapper(ABC):
    """Abstract LLM wrapper interface."""

    @abstractmethod
    def chat(
        self,
        messages: List[Message],
        functions: List[Dict] = None,
        **kwargs
    ) -> List[Message]:
        """Synchronous chat completion."""

    async def achat(
        self,
        messages: List[Message],
        functions: List[Dict] = None,
        **kwargs
    ) -> List[Message]:
        """Async chat completion."""
```

### MarieEngineLLMWrapper

Wrapper for local models via Marie Engine.

```python
from marie.agent.llm_wrapper import MarieEngineLLMWrapper

class MarieEngineLLMWrapper(BaseLLMWrapper):
    """
    Local model inference via marie.engine.

    Features:
    - vLLM integration
    - Tool call detection (XML and Action formats)
    - Guided generation support
    """

    def __init__(
        self,
        engine_name: str,
        provider: str = "vllm",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ):
        """
        Initialize Marie Engine wrapper.

        Args:
            engine_name: Model name in Marie (e.g., "qwen2_5_vl_7b")
            provider: Backend provider (vllm, openai)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
```

### OpenAICompatibleWrapper

Wrapper for OpenAI-compatible APIs.

```python
from marie.agent.llm_wrapper import OpenAICompatibleWrapper

class OpenAICompatibleWrapper(BaseLLMWrapper):
    """
    OpenAI-compatible API wrapper.

    Works with: OpenAI, Azure OpenAI, Claude, etc.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ):
        """
        Initialize OpenAI-compatible wrapper.

        Args:
            model: Model name (e.g., "gpt-4o-mini", "claude-3-sonnet")
            api_key: API key
            base_url: API endpoint URL
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
```

### get_llm_wrapper

Factory function for LLM wrappers.

```python
from marie.agent.llm_wrapper import get_llm_wrapper

def get_llm_wrapper(
    backend: str = "marie",
    **kwargs
) -> BaseLLMWrapper:
    """
    Create LLM wrapper by backend type.

    Args:
        backend: "marie" or "openai"
        **kwargs: Backend-specific arguments

    Returns:
        Configured LLM wrapper instance
    """
```

## Configuration

### AgentConfig

Agent configuration model.

```python
from marie.agent.config import AgentConfig, LLMConfig

class LLMConfig(BaseModel):
    backend: str = "marie"  # "marie" or "openai"
    engine_name: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2048

class AgentConfig(BaseModel):
    name: str
    description: str = ""
    backend: str = "qwen_agent"
    system_message: str = ""
    max_iterations: int = 10
    llm: LLMConfig
    tools: List[Dict] = []
    memory: Optional[Dict] = None

    @classmethod
    def from_yaml(cls, path: str) -> "AgentConfig":
        """Load from YAML file."""

    @classmethod
    def from_yaml_string(cls, yaml_str: str) -> "AgentConfig":
        """Load from YAML string."""
```

### load_config

Configuration loader.

```python
from marie.agent.config import load_config

def load_config(
    path: str = None,
    config_dict: Dict = None,
    **kwargs
) -> AgentConfig:
    """
    Load agent configuration.

    Args:
        path: YAML file path
        config_dict: Configuration dictionary
        **kwargs: Override values

    Returns:
        AgentConfig instance
    """
```

## State Management

### ConversationState

Conversation state container.

```python
from marie.agent.state import ConversationState

class ConversationState:
    def __init__(self, conversation_id: str = None):
        """
        Initialize conversation state.

        Args:
            conversation_id: Unique ID (auto-generated if None)
        """

    @property
    def conversation_id(self) -> str:
        """Conversation identifier."""

    @property
    def messages(self) -> List[Message]:
        """All messages in conversation."""

    def add_message(self, message: Message) -> None:
        """Add message to conversation."""

    def cache_tool_result(
        self,
        tool_name: str,
        args: str,
        result: Any,
        ttl: int = 3600
    ) -> None:
        """
        Cache tool result.

        Args:
            tool_name: Tool name
            args: Arguments (as cache key)
            result: Result to cache
            ttl: Time-to-live in seconds
        """

    def get_cached_tool_result(
        self,
        tool_name: str,
        args: str
    ) -> Optional[Any]:
        """Get cached result if available and not expired."""
```

### ConversationStore

Thread-safe conversation storage.

```python
from marie.agent.state import ConversationStore

class ConversationStore:
    def store(self, state: ConversationState) -> None:
        """Store conversation state."""

    def get(self, conversation_id: str) -> Optional[ConversationState]:
        """Retrieve conversation by ID."""

    def list_conversations(self) -> List[str]:
        """List all conversation IDs."""

    def delete(self, conversation_id: str) -> bool:
        """Delete conversation."""

    def clear(self) -> None:
        """Delete all conversations."""
```

## Public Exports

All public APIs are available from `marie.agent`:

```python
from marie.agent import (
    # Agents
    BaseAgent,
    ReactAgent,
    PlanAndExecuteAgent,
    ChatAgent,
    VisionDocumentAgent,
    DocumentExtractionAgent,
    DocumentQAAgent,

    # Messages
    Message,
    ContentItem,

    # Tools
    AgentTool,
    ToolMetadata,
    ToolOutput,
    register_tool,
    TOOL_REGISTRY,
    resolve_tools,

    # LLM Wrappers
    BaseLLMWrapper,
    MarieEngineLLMWrapper,
    OpenAICompatibleWrapper,
    get_llm_wrapper,

    # Configuration
    AgentConfig,
    LLMConfig,
    load_config,

    # State
    ConversationState,
    ConversationStore,
)
```

## Type Definitions

Common types used throughout the framework:

```python
from typing import (
    List, Dict, Union, Optional, Any,
    Callable, Generator, Type
)

# Message types
MessageDict = Dict[str, Any]
MessageList = List[Union[Message, MessageDict]]

# Tool types
ToolSpec = Union[str, Dict, AgentTool, Callable]
ToolList = List[ToolSpec]

# Function schema
FunctionSchema = Dict[str, Any]  # OpenAI function schema format
```

## Exceptions

```python
from marie.agent.exceptions import (
    AgentError,           # Base exception
    ToolNotFoundError,    # Tool not in registry
    ToolExecutionError,   # Tool failed to execute
    LLMError,             # LLM backend error
    ConfigurationError,   # Invalid configuration
)
```