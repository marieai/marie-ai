---
sidebar_position: 3
---

# Core Concepts

This guide explains the fundamental concepts of the Marie-AI Agent Framework.

## Agents

An **Agent** is an LLM-powered entity that can understand user requests, reason about them, and take actions using tools. Agents follow a loop:

1. Receive user message
2. Decide if a tool is needed
3. Execute tool (if needed)
4. Generate response
5. Repeat until task is complete

### Agent Lifecycle

```text
User Message
     │
     ▼
┌─────────────┐
│    LLM      │◄──────────────────────────┐
│  Reasoning  │                           │
└──────┬──────┘                           │
       │                                  │
       ▼                                  │
   Tool Call?                             │
    ┌──┴──┐                              │
   Yes    No                             │
    │      │                             │
    ▼      ▼                             │
┌───────┐  ┌────────────┐                │
│Execute│  │  Generate  │                │
│ Tool  │  │  Response  │                │
└───┬───┘  └────────────┘                │
    │                                     │
    └─────────────────────────────────────┘
```

### BaseAgent

All agents inherit from `BaseAgent`, which provides:

- Message normalization (dict ↔ Message objects)
- Tool execution infrastructure
- Iteration control
- Streaming response handling

```python
from marie.agent.base import BaseAgent

class MyCustomAgent(BaseAgent):
    def _run(self, messages, **kwargs):
        # Implement your agent logic
        yield [Message(role="assistant", content="Hello!")]
```

## Messages

Messages represent the conversation between user and agent. The framework uses a unified `Message` class compatible with OpenAI's format.

### Message Structure

```python
from marie.agent import Message, ContentItem

# Simple text message
msg = Message(role="user", content="Hello!")

# Multimodal message with image
msg = Message(
    role="user",
    content=[
        ContentItem(image="/path/to/image.jpg"),
        ContentItem(text="What's in this image?"),
    ]
)

# Message with tool call
msg = Message(
    role="assistant",
    content="I'll check the weather.",
    tool_calls=[
        ToolCall(
            id="call_123",
            function=FunctionCall(
                name="get_weather",
                arguments='{"city": "Tokyo"}'
            )
        )
    ]
)
```

### Message Roles

| Role | Description |
|------|-------------|
| `system` | System prompt/instructions for the agent |
| `user` | Human input |
| `assistant` | Agent/LLM response |
| `tool` | Result from tool execution |
| `function` | Legacy function result (OpenAI compatibility) |

### Content Types

Messages can contain multiple content types:

```python
from marie.agent.message import ContentItem, ContentItemType

# Text content
ContentItem(text="Hello world")

# Image content (file path, URL, or base64)
ContentItem(image="/path/to/image.png")
ContentItem(image="https://example.com/image.jpg")
ContentItem(image="data:image/png;base64,...")

# File content (documents)
ContentItem(file="/path/to/document.pdf")

# Audio/Video (placeholders)
ContentItem(audio="/path/to/audio.mp3")
ContentItem(video="/path/to/video.mp4")
```

### Message Normalization

The framework automatically handles conversion between dict and Message formats:

```python
# Dict format (convenient for users)
messages = [
    {"role": "user", "content": "Hello!"},
    {"role": "user", "content": [
        {"image": "photo.jpg"},
        {"text": "Describe this"}
    ]}
]

# Message format (used internally)
messages = [
    Message(role="user", content="Hello!"),
    Message(role="user", content=[
        ContentItem(image="photo.jpg"),
        ContentItem(text="Describe this")
    ])
]
```

## Tools

Tools extend agent capabilities by allowing them to execute functions, access APIs, or perform computations.

### Tool Components

```text
┌─────────────────────────────────────────────┐
│                    Tool                     │
├─────────────────────────────────────────────┤
│  ┌─────────────────────────────────────┐   │
│  │           ToolMetadata               │   │
│  │  • name: Unique identifier           │   │
│  │  • description: What it does         │   │
│  │  • fn_schema: Input validation       │   │
│  │  • parameters: JSON schema           │   │
│  └─────────────────────────────────────┘   │
│                                             │
│  ┌─────────────────────────────────────┐   │
│  │           call() method              │   │
│  │  • Receives validated arguments      │   │
│  │  • Executes tool logic               │   │
│  │  • Returns ToolOutput                │   │
│  └─────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
```

### ToolMetadata

Describes a tool's interface for the LLM:

```python
from marie.agent import ToolMetadata
from pydantic import BaseModel, Field

class WeatherInput(BaseModel):
    city: str = Field(description="City name to get weather for")
    units: str = Field(default="celsius", description="Temperature units")

metadata = ToolMetadata(
    name="get_weather",
    description="Get the current weather for a city",
    fn_schema=WeatherInput,  # Pydantic model for validation
)

# Get OpenAI-compatible schema
print(metadata.to_openai_tool())
```

### ToolOutput

Result from tool execution:

```python
from marie.agent.tools import ToolOutput

output = ToolOutput(
    content='{"temperature": 72, "condition": "sunny"}',
    tool_name="get_weather",
    raw_input={"city": "Tokyo"},
    raw_output={"temperature": 72, "condition": "sunny"},
    is_error=False,
)
```

### Tool Registry

Tools are managed through a global registry:

```python
from marie.agent import register_tool, TOOL_REGISTRY

# Register a tool
@register_tool("my_tool")
def my_tool(x: int) -> str:
    """Multiply by 2."""
    return str(x * 2)

# Access registered tools
TOOL_REGISTRY.list_tools()  # ["my_tool", ...]
TOOL_REGISTRY.get("my_tool")  # Returns AgentTool instance
TOOL_REGISTRY.has("my_tool")  # True
```

## LLM Backends

LLM backends handle communication with language models.

### LLM Wrapper Interface

```python
from marie.agent.llm_wrapper import BaseLLMWrapper

class CustomLLMWrapper(BaseLLMWrapper):
    def chat(self, messages, functions=None, **kwargs):
        """Synchronous chat completion."""
        # Return list of response messages
        pass

    async def achat(self, messages, functions=None, **kwargs):
        """Async chat completion."""
        pass
```

### MarieEngineLLMWrapper

For local models via Marie's engine system:

```python
from marie.agent.llm_wrapper import MarieEngineLLMWrapper

llm = MarieEngineLLMWrapper(
    engine_name="qwen2_5_vl_7b",  # Model name in Marie
    provider="vllm",              # vllm, openai, etc.
    temperature=0.7,
    max_tokens=2048,
)
```

Features:
- Integrates with `marie.engine.get_engine()`
- Supports vLLM for efficient local inference
- Detects tool calls in XML format: `<tool_call>{json}</tool_call>`
- Fallback to Action/Action Input format

### OpenAICompatibleWrapper

For cloud LLMs (OpenAI, Claude, etc.):

```python
from marie.agent.llm_wrapper import OpenAICompatibleWrapper

# OpenAI
llm = OpenAICompatibleWrapper(
    model="gpt-4o-mini",
    api_key="sk-...",
    base_url="https://api.openai.com/v1",
)

# Azure OpenAI
llm = OpenAICompatibleWrapper(
    model="gpt-4",
    api_key="...",
    base_url="https://your-resource.openai.azure.com/",
)

# Claude via Anthropic
llm = OpenAICompatibleWrapper(
    model="claude-3-sonnet",
    api_key="...",
    base_url="https://api.anthropic.com/v1/",
)
```

Features:
- Standard OpenAI API format
- Handles both `function_call` (legacy) and `tool_calls` (current)
- Async support with `achat()`

## Conversation State

The framework provides state management for persistent conversations.

### ConversationState

```python
from marie.agent.state import ConversationState, ConversationStore

# Create a conversation
state = ConversationState(conversation_id="conv-123")
state.add_message(Message(role="user", content="Hello"))
state.add_message(Message(role="assistant", content="Hi there!"))

# Cache tool results (with TTL)
state.cache_tool_result("get_weather", "Tokyo", {"temp": 72})
cached = state.get_cached_tool_result("get_weather", "Tokyo")
```

### ConversationStore

Thread-safe storage for multiple conversations:

```python
store = ConversationStore()

# Store conversation
store.store(state)

# Retrieve by ID
state = store.get("conv-123")

# List all conversations
all_ids = store.list_conversations()
```

## Execution Flow

### ReAct Pattern (ReactAgent)

The ReactAgent uses the ReAct (Reasoning + Acting) pattern:

```text
User: "What's 15% of 85?"
          │
          ▼
┌─────────────────────┐
│     LLM thinks:     │
│ "I need to use the  │
│ calculator tool"    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Tool Call Detected │
│  calculator(        │
│    "85 * 0.15"      │
│  )                  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Execute Tool      │
│   Result: "12.75"   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   LLM generates     │
│   final response:   │
│   "15% of 85 is     │
│    12.75"           │
└─────────────────────┘
```

### Tool Call Detection

The framework detects tool calls in multiple formats:

**XML Format (Primary)**
```
<tool_call>
{"name": "calculator", "arguments": {"expression": "85 * 0.15"}}
</tool_call>
```

**Action Format (Fallback)**
```
Action: calculator
Action Input: {"expression": "85 * 0.15"}
```

**OpenAI Native Format**
```json
{
  "tool_calls": [{
    "id": "call_123",
    "function": {
      "name": "calculator",
      "arguments": "{\"expression\": \"85 * 0.15\"}"
    }
  }]
}
```

## Configuration

### YAML Configuration

Agents can be configured via YAML:

```yaml
name: DocumentAnalyzer
description: Analyzes documents using OCR and classification
backend: qwen_agent

llm:
  backend: marie
  engine_name: qwen2_5_vl_7b
  provider: vllm
  temperature: 0.7
  max_tokens: 2048

tools:
  - name: document_ocr
    config:
      api_url: http://localhost:51000/api
  - name: document_classifier

memory:
  type: buffer
  max_messages: 100

system_message: |
  You are a document analysis assistant.
  Use the available tools to extract and analyze document content.
```

### Loading Configuration

```python
from marie.agent.config import AgentConfig

# From YAML file
config = AgentConfig.from_yaml("agent_config.yaml")

# From dict
config = AgentConfig(
    name="MyAgent",
    llm={"backend": "openai", "model": "gpt-4o-mini"},
    tools=[{"name": "calculator"}],
)
```

## Next Steps

- **[Tool Development](./tool-development.md)**: Create custom tools
- **[Built-in Agents](./built-in-agents.md)**: Explore available agent types
- **[Examples](./examples.md)**: See complete working examples