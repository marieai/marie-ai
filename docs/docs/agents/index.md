---
sidebar_position: 1
---

# Agent Framework

Marie-AI's Agent Framework provides a production-ready system for building LLM-powered agents with tool calling capabilities. It supports multiple LLM backends, streaming responses, and integrates seamlessly with Marie's document processing infrastructure.

## Overview

The Agent Framework enables you to:

- **Build conversational AI agents** with custom tools and system prompts
- **Process documents intelligently** using vision-language models
- **Execute multi-step workflows** with planning agents
- **Integrate with multiple LLM backends** including local models (via vLLM) and cloud providers (OpenAI, Claude)

## Key Features

| Feature | Description |
|---------|-------------|
| **Multiple Agent Types** | ReactAgent (ReAct), PlanAndExecuteAgent, ChatAgent, VisionDocumentAgent |
| **Tool System** | Registry-based tools with schema validation and automatic discovery |
| **LLM Backends** | Support for Marie Engine (vLLM), OpenAI, and compatible APIs |
| **Multimodal Support** | Handle text, images, and documents natively |
| **Streaming** | Yield-based streaming API for real-time responses |
| **State Management** | Conversation persistence and memory |

## Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│                        Agent Framework                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Assistant    │  │  Planning    │  │   Vision     │          │
│  │   Agent      │  │   Agent      │  │ Doc Agent    │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                 │                 │                   │
│         └────────────────┬┴─────────────────┘                   │
│                          ▼                                      │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                      BaseAgent                            │  │
│  │  • Message normalization                                  │  │
│  │  • Tool execution                                         │  │
│  │  • Iteration control                                      │  │
│  └───────────────────────────────────────────────────────────┘  │
│                          │                                      │
│         ┌────────────────┼────────────────┐                     │
│         ▼                ▼                ▼                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │    Tools    │  │ LLM Wrapper │  │   State     │             │
│  │  Registry   │  │  (Backend)  │  │  Manager    │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

Here's a minimal example of creating an agent with a custom tool:

```python
from marie.agent import ReactAgent, register_tool
from marie.agent.llm_wrapper import OpenAICompatibleWrapper
import os

# Define a tool using the decorator
@register_tool("get_weather")
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    # In practice, call a weather API
    return f"The weather in {city} is sunny, 72°F"

# Create the LLM backend
llm = OpenAICompatibleWrapper(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
)

# Create the agent
agent = ReactAgent(
    llm=llm,
    function_list=["get_weather"],
    name="WeatherBot",
    description="A helpful assistant that provides weather information",
    system_message="You are a helpful weather assistant.",
)

# Run the agent
messages = [{"role": "user", "content": "What's the weather in Tokyo?"}]
for responses in agent.run(messages=messages):
    if responses:
        print(responses[-1].content)
```

## Agent Types

### ReactAgent

General-purpose agent using the ReAct (Reasoning + Acting) pattern. Best for:
- Multi-step reasoning tasks
- Tool-augmented conversations
- Complex problem solving

### PlanAndExecuteAgent

Creates explicit plans before execution. Best for:
- Complex multi-step workflows
- Tasks requiring coordination
- Long-horizon planning

### VisionDocumentAgent

Specialized for document understanding. Best for:
- OCR and text extraction
- Document classification
- Form and invoice processing
- Visual question answering

### ChatAgent

Simple conversational agent without tools. Best for:
- Pure Q&A scenarios
- Lightweight chatbots
- Information retrieval

## LLM Backend Options

| Backend | Use Case | Configuration |
|---------|----------|---------------|
| **MarieEngineLLMWrapper** | Local models via vLLM | `engine_name="qwen2_5_vl_7b"` |
| **OpenAICompatibleWrapper** | OpenAI, Claude, etc. | `model="gpt-4o-mini"` |

## Directory Structure

```
marie/agent/
├── __init__.py          # Public exports
├── base.py              # BaseAgent class
├── message.py           # Message types
├── config.py            # Configuration
├── llm_wrapper.py       # LLM backends
├── agents/              # Agent implementations
│   ├── assistant.py
│   ├── planning.py
│   └── vision_document_agent.py
├── tools/               # Tool system
│   ├── base.py
│   └── registry.py
├── backends/            # Execution backends
└── state/               # State management
```

## Next Steps

1. **[Getting Started](./getting-started.md)**: Set up your first agent
2. **[Core Concepts](./core-concepts.md)**: Understand agents, tools, and messages
3. **[Tool Development](./tool-development.md)**: Create custom tools
4. **[Built-in Agents](./built-in-agents.md)**: Reference for all agent types
5. **[Examples](./examples.md)**: Complete working examples