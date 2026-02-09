"""Marie Agent Framework.

A state-of-the-art agent framework for Marie-AI following the Qwen-Agent
blueprint with Haystack Agent patterns, integrated with the existing
Executor system.

Architecture Overview:
    Qwen-Agent serves as the meta-planner that can delegate to Haystack
    and AutoGen as wrapped tools. The framework provides:

    - **BaseAgent**: Template method pattern for agent implementations
    - **AgentTool**: Unified tool interface with registry support
    - **LLM Wrappers**: Bridge to marie.engine and OpenAI-compatible APIs
    - **Configuration**: Support for both YAML and Python configuration

Example Usage:
    ```python
    from marie.agent import BaseAgent, register_tool, Message
    from marie.agent.llm_wrapper import MarieEngineLLMWrapper


    # Register a custom tool
    @register_tool("search")
    def search_documents(query: str) -> str:
        '''Search for documents.'''
        return json.dumps({"results": [...]})


    # Create an agent
    class MyAgent(BaseAgent):
        def _run(self, messages, lang="en", **kwargs):
            for response in self._call_llm(messages, self._get_tool_definitions()):
                # Check for tool calls
                has_call, name, args, text, tool_id = self._detect_tool_call(response[0])
                if has_call:
                    result = self._call_tool(name, args)
                    # Continue conversation with tool result...
                yield response


    agent = MyAgent(
        llm=MarieEngineLLMWrapper(engine_name="qwen2_5_vl_7b"),
        function_list=["search"],
        system_message="You are a helpful assistant.",
    )

    # Run the agent
    for responses in agent.run([{"role": "user", "content": "Hello"}]):
        print(responses[-1].content)
    ```

Configuration:
    ```yaml
    # agent_config.yaml
    agent:
      name: my_agent
      backend: qwen_agent
      llm:
        engine_name: qwen2_5_vl_7b
        provider: vllm
      tools:
        - search
        - calculator
    ```

    ```python
    from marie.agent.config import AgentConfig

    config = AgentConfig.from_yaml("agent_config.yaml")
    ```
"""

from marie.agent.agents import (
    ChatAgent,
    DocumentExtractionAgent,
    DocumentQAAgent,
    FunctionCallingAgent,
    MultiAgentHub,
    PlanAndExecuteAgent,
    ReactAgent,
    Router,
    VisionDocumentAgent,
)
from marie.agent.assistants import DocumentAssistant
from marie.agent.backends import (
    AgentBackend,
    AgentResult,
    AgentStatus,
    AutoGenAgentBackend,
    AutoGenBackendConfig,
    BackendConfig,
    CompositeBackend,
    HaystackAgentBackend,
    HaystackBackendConfig,
    QwenAgentBackend,
    QwenBackendConfig,
    SimpleHaystackBackend,
    SimpleQwenBackend,
    SingleAgentBackend,
    ToolCallRecord,
    create_coding_backend,
    create_research_backend,
)
from marie.agent.base import BaseAgent, BasicAgent
from marie.agent.config import (
    AgentConfig,
    ExecutorConfig,
    LLMConfig,
    MCPConfig,
    MemoryConfig,
    ToolConfig,
    load_config,
)
from marie.agent.executor import AgentExecutor
from marie.agent.llm_types import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    ContentBlock,
    ImageBlock,
    MessageRole,
    TextBlock,
)
from marie.agent.llm_wrapper import (
    BaseLLMWrapper,
    MarieEngineLLMWrapper,
    OpenAICompatibleWrapper,
    get_llm_wrapper,
)
from marie.agent.message import (
    ASSISTANT,
    CONTENT,
    DEFAULT_SYSTEM_MESSAGE,
    FUNCTION,
    NAME,
    ROLE,
    SYSTEM,
    TOOL,
    USER,
    ContentItem,
    ContentItemType,
    FunctionCall,
    Message,
    ToolCall,
    format_messages,
)
from marie.agent.state import (
    AgentMemoryBridge,
    ConversationState,
    ConversationStore,
)
from marie.agent.tools import (
    TOOL_REGISTRY,
    AgentTool,
    ComponentTool,
    DocumentSearchTool,
    FunctionTool,
    MultiDocumentSearchTool,
    ToolMetadata,
    ToolOutput,
    adapt_tool,
    get_tool,
    list_tools,
    register_tool,
    resolve_tools,
)
from marie.agent.utils import asyncio_run, get_or_reuse_loop, run_async

__all__ = [
    # Base classes
    "BaseAgent",
    "BasicAgent",
    # Agent implementations
    "ReactAgent",
    "PlanAndExecuteAgent",
    "ChatAgent",
    "FunctionCallingAgent",
    # Router
    "Router",
    "MultiAgentHub",
    # Vision Document Agents
    "VisionDocumentAgent",
    "DocumentExtractionAgent",
    "DocumentQAAgent",
    # Document Assistants (RAG)
    "DocumentAssistant",
    # Backends
    "AgentBackend",
    "AgentResult",
    "AgentStatus",
    "BackendConfig",
    "CompositeBackend",
    "ToolCallRecord",
    # Qwen backend
    "QwenAgentBackend",
    "QwenBackendConfig",
    "SimpleQwenBackend",
    # Haystack backend
    "HaystackAgentBackend",
    "HaystackBackendConfig",
    "SimpleHaystackBackend",
    # AutoGen backend
    "AutoGenAgentBackend",
    "AutoGenBackendConfig",
    "SingleAgentBackend",
    "create_research_backend",
    "create_coding_backend",
    # Executor
    "AgentExecutor",
    # State management
    "ConversationStore",
    "ConversationState",
    "AgentMemoryBridge",
    # Message types
    "Message",
    "ContentItem",
    "ContentItemType",
    "FunctionCall",
    "ToolCall",
    "format_messages",
    # Message constants
    "ROLE",
    "CONTENT",
    "NAME",
    "SYSTEM",
    "USER",
    "ASSISTANT",
    "FUNCTION",
    "TOOL",
    "DEFAULT_SYSTEM_MESSAGE",
    # Tool system
    "AgentTool",
    "ComponentTool",
    "DocumentSearchTool",
    "FunctionTool",
    "MultiDocumentSearchTool",
    "ToolMetadata",
    "ToolOutput",
    "TOOL_REGISTRY",
    "register_tool",
    "get_tool",
    "list_tools",
    "resolve_tools",
    "adapt_tool",
    # LLM wrappers
    "BaseLLMWrapper",
    "MarieEngineLLMWrapper",
    "OpenAICompatibleWrapper",
    "get_llm_wrapper",
    # Configuration
    "AgentConfig",
    "ExecutorConfig",
    "LLMConfig",
    "MCPConfig",
    "MemoryConfig",
    "ToolConfig",
    "load_config",
    # LLM types (native)
    "MessageRole",
    "TextBlock",
    "ImageBlock",
    "ContentBlock",
    "ChatMessage",
    "ChatResponse",
    "CompletionResponse",
    # Utilities
    "run_async",
    "asyncio_run",
    "get_or_reuse_loop",
]

__version__ = "0.1.0"
