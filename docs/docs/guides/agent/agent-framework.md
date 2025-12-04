---
sidebar_position: 1
---

# Agent Framework

The Marie Agent Framework provides a state-of-the-art agent system following the Qwen-Agent blueprint with Haystack Agent patterns, integrated with Marie's existing Executor system.

---

## Architecture Overview

The agent framework uses **Qwen-Agent as the meta-planner** that orchestrates work and can delegate to specialized backends like Haystack (for RAG) and AutoGen (for multi-agent teams) through tool calls.

```
┌─────────────────────────────────────────────────────────────┐
│              JOB SCHEDULER                                  │
│  Dispatches AGENT job to AgentExecutor                      │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   AGENT EXECUTOR                            │
│  ┌───────────────────────────────────────────────────────┐ │
│  │            QWEN-AGENT (Meta-Planner)                  │ │
│  │  Planning → Tool Selection → Execution                │ │
│  │       │            │             │                    │ │
│  │  ┌────▼────┐  ┌────▼────┐  ┌────▼────┐              │ │
│  │  │Haystack │  │AutoGen  │  │ Other   │              │ │
│  │  │RAG Tool │  │Team Tool│  │ Tools   │              │ │
│  │  └─────────┘  └─────────┘  └─────────┘              │ │
│  └───────────────────────────────────────────────────────┘ │
│  Returns final result when complete                        │
└─────────────────────────────────────────────────────────────┘
```

**Key Points:**
- Agent tasks execute to completion within the AgentExecutor
- Tool calls (Haystack RAG, AutoGen teams) happen as internal calls, not separate DAG jobs
- The scheduler dispatches agent jobs like any other executor job
- Agents maintain conversation state for multi-turn interactions

---

## Core Concepts

### Message Schema

The `Message` class provides a unified format for agent communication, compatible with both Qwen-Agent and OpenAI patterns.

```python
from marie.agent import Message

# Create messages
system_msg = Message.system("You are a helpful assistant.")
user_msg = Message.user("What is 2 + 2?")
assistant_msg = Message.assistant("2 + 2 equals 4.")

# Tool result message
tool_result = Message.tool_result(
    tool_call_id="call_123",
    content="Result data",
    name="calculator",
)
```

### Tool System

Tools are registered globally and can be resolved by name. Use the `@register_tool` decorator to create custom tools.

```python
from marie.agent import register_tool, get_tool, resolve_tools

@register_tool("calculator")
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    return str(eval(expression))

# Get a registered tool
tool = get_tool("calculator")

# Resolve multiple tools
tools = resolve_tools(["calculator", "search"])
```

### Agent Backends

Backends provide different execution strategies. The framework includes:

| Backend | Description |
|---------|-------------|
| `QwenAgentBackend` | Native Qwen-style ReAct execution |
| `HaystackAgentBackend` | Wraps Haystack RAG pipelines |
| `AutoGenAgentBackend` | Wraps AutoGen multi-agent teams |
| `CompositeBackend` | Delegates to multiple sub-backends |

```python
from marie.agent import QwenAgentBackend, BackendConfig

backend = QwenAgentBackend(
    config=BackendConfig(
        max_iterations=10,
        timeout_seconds=300.0,
    )
)

result = await backend.run(
    messages=[Message.user("Hello")],
    tools={"calculator": calculator_tool},
)
```

---

## Agent Executor

The `AgentExecutor` integrates agents with Marie's executor system, providing HTTP/gRPC endpoints for agent interactions.

### YAML Configuration

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

### Python Usage

```python
from marie.agent import AgentExecutor

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

### Endpoints

| Endpoint | Description |
|----------|-------------|
| `/chat` | Main chat endpoint for agent interaction |
| `/chat/stream` | Streaming chat endpoint (SSE planned) |
| `/tools` | List available tools |
| `/conversations` | List active conversations |
| `/conversation/clear` | Clear a conversation |

---

## Conversation State

The `ConversationStore` manages conversation state with LRU eviction and TTL support.

```python
from marie.agent import ConversationStore, Message

store = ConversationStore(
    max_conversations=1000,
    max_messages_per_conversation=100,
    ttl_seconds=3600,
)

# Add messages
await store.add_message("conv-1", Message.user("Hello"))
await store.add_message("conv-1", Message.assistant("Hi there!"))

# Get history
messages = await store.get_messages("conv-1")

# Clear conversation
await store.clear("conv-1")
```

---

## Query Planners

The framework includes query planners for creating DAG workflows with agent execution nodes.

### Available Planners

| Planner | Description |
|---------|-------------|
| `agent_qwen_orchestrator` | Qwen-Agent as top-level planner |
| `agent_haystack_rag` | Haystack RAG workflow |
| `agent_autogen_team` | AutoGen team workflow |
| `agent_composite` | Composite with delegation |

### Query Definitions

```python
from marie.query_planner.agent_planner import (
    AgentQueryDefinition,
    RAGQueryDefinition,
    MultiAgentQueryDefinition,
)

# Agent execution
agent_def = AgentQueryDefinition(
    method="AGENT",
    agent_backend="qwen_agent",
    max_iterations=10,
    tools=["search", "calculator"],
)

# RAG execution
rag_def = RAGQueryDefinition(
    method="RAG",
    retriever_type="bm25",
    top_k=5,
)
```

---

## Creating Custom Tools

### Function Tool

```python
from marie.agent import register_tool

@register_tool("search_docs")
def search_documents(query: str, limit: int = 10) -> str:
    """Search documents by query.

    Args:
        query: Search query string
        limit: Maximum results to return

    Returns:
        JSON string of search results
    """
    results = perform_search(query, limit)
    return json.dumps(results)
```

### Class-Based Tool

```python
from marie.agent import AgentTool, ToolMetadata, ToolOutput

class DatabaseTool(AgentTool):
    def __init__(self, connection_string: str):
        self._conn = connect(connection_string)
        self._metadata = ToolMetadata(
            name="database",
            description="Query the database",
            parameters={
                "type": "object",
                "properties": {
                    "sql": {"type": "string", "description": "SQL query"},
                },
                "required": ["sql"],
            },
        )

    @property
    def name(self) -> str:
        return "database"

    @property
    def description(self) -> str:
        return "Query the database"

    @property
    def metadata(self) -> ToolMetadata:
        return self._metadata

    def call(self, sql: str) -> ToolOutput:
        result = self._conn.execute(sql)
        return ToolOutput(
            content=json.dumps(result.fetchall()),
            tool_name="database",
        )

    async def acall(self, sql: str) -> ToolOutput:
        return self.call(sql)
```

---

## Tool Wrappers

The framework provides wrappers for integrating external systems as tools.

### ComponentTool

Wrap any class with a `run()` or `__call__()` method as a tool, with automatic schema generation:

```python
from marie.agent import ComponentTool

# Wrap a class with run() method
class TextAnalyzer:
    def run(self, text: str, language: str = "en") -> dict:
        """Analyze text sentiment."""
        return {"sentiment": "positive", "language": language}

analyzer = TextAnalyzer()
tool = ComponentTool.from_component(analyzer)

# Wrap a callable class
class Calculator:
    def __call__(self, expression: str) -> float:
        return eval(expression)

calc = Calculator()
tool = ComponentTool.from_component(calc, name="calculator")

# Wrap with custom method name
class DataProcessor:
    def process(self, data: str, format: str = "json") -> str:
        return f"{format}: {data}"

processor = DataProcessor()
tool = ComponentTool.from_component(
    processor,
    method_name="process",
    name="data_processor",
)

# For pipeline-style components (dict output)
tool = ComponentTool.from_pipeline_component(
    component=retriever,
    name="search",
    description="Search documents",
)
```

### ExecutorTool

Wrap Marie executors as tools:

```python
from marie.agent.tools.wrappers import ExecutorTool

tool = ExecutorTool.from_executor(
    executor=my_executor,
    endpoint="/extract",
    name="document_extractor",
    description="Extract data from documents",
)
```

### HaystackPipelineTool

Wrap Haystack pipelines:

```python
from marie.agent.tools.wrappers import HaystackPipelineTool

tool = HaystackPipelineTool.from_pipeline(
    pipeline=rag_pipeline,
    name="rag_search",
    description="Search and retrieve relevant documents",
    input_keys=["query"],
    output_key="answers",
)
```

### AutoGenTeamTool

Wrap AutoGen teams:

```python
from marie.agent.tools.wrappers import AutoGenTeamTool

tool = AutoGenTeamTool.from_group_chat(
    group_chat=research_team,
    initiator=user_proxy,
    name="research_team",
    description="Research team for complex analysis",
)
```

---

## Configuration

### AgentConfig

```python
from marie.agent import AgentConfig, LLMConfig

config = AgentConfig(
    name="my_agent",
    backend="qwen_agent",
    llm=LLMConfig(
        engine_name="qwen2_5_vl_7b",
        provider="vllm",
        temperature=0.7,
    ),
    tools=["search", "calculator"],
    system_message="You are a helpful assistant.",
    max_iterations=10,
)

# Load from YAML
config = AgentConfig.from_yaml("agent_config.yaml")
```

### YAML Configuration File

```yaml
# agent_config.yaml
agent:
  name: my_agent
  backend: qwen_agent
  llm:
    engine_name: qwen2_5_vl_7b
    provider: vllm
    temperature: 0.7
    max_tokens: 4096
  tools:
    - search
    - calculator
    - document_extractor
  system_message: |
    You are a helpful assistant specialized in document processing.
    Use the available tools to help users with their requests.
  max_iterations: 10

memory:
  type: chat_buffer
  max_messages: 100
```

---

## Best Practices

1. **Tool Design**: Keep tools focused and single-purpose. Document parameters clearly.

2. **Error Handling**: Tools should return error information in the ToolOutput rather than raising exceptions.

3. **Conversation Management**: Use conversation IDs to maintain context across requests.

4. **Backend Selection**:
   - Use `qwen_agent` for general-purpose tasks
   - Use `haystack` for RAG-heavy workflows
   - Use `autogen` for complex multi-agent scenarios

5. **Resource Management**: Set appropriate `max_iterations` and `timeout_seconds` to prevent runaway executions.

---

## Module Reference

### Core Imports

```python
from marie.agent import (
    # Base classes
    BaseAgent,
    BasicAgent,

    # Agent implementations
    AssistantAgent,
    ChatAgent,
    FunctionCallingAgent,
    PlanningAgent,

    # Backends
    AgentBackend,
    AgentResult,
    AgentStatus,
    BackendConfig,
    QwenAgentBackend,
    HaystackAgentBackend,
    AutoGenAgentBackend,
    CompositeBackend,

    # Executor
    AgentExecutor,

    # State management
    ConversationStore,
    ConversationState,
    AgentMemoryBridge,

    # Message types
    Message,
    ContentItem,
    FunctionCall,
    ToolCall,

    # Tool system
    AgentTool,
    ComponentTool,
    FunctionTool,
    ToolMetadata,
    ToolOutput,
    TOOL_REGISTRY,
    register_tool,
    get_tool,
    resolve_tools,

    # Configuration
    AgentConfig,
    LLMConfig,
    load_config,
)
```
