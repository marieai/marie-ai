---
sidebar_position: 2
---

# Getting Started

This guide walks you through setting up and running your first Marie-AI agent.

## Prerequisites

- Python 3.10+
- Marie-AI installed (`pip install marie-ai`)
- An LLM backend (OpenAI API key or local model)

## Installation

The agent framework is included with Marie-AI:

```bash
pip install marie-ai
```

For development with all agent features:

```bash
pip install marie-ai[agent]
```

## Configuration

### Environment Variables

Create a `.env` file with your configuration:

```bash
# OpenAI (for cloud LLM backend)
OPENAI_API_KEY=your-openai-api-key
OPENAI_API_BASE=https://api.openai.com/v1

# Marie API (for document processing tools)
MARIE_API_URL=http://127.0.0.1:51000/api
MARIE_API_KEY=your-marie-api-key
```

### LLM Backend Setup

Choose your LLM backend based on your requirements:

**Option 1: OpenAI (Recommended for getting started)**

```python
from marie.agent.llm_wrapper import OpenAICompatibleWrapper
import os

llm = OpenAICompatibleWrapper(
    model="gpt-4o-mini",  # or "gpt-4o", "gpt-4-turbo"
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.openai.com/v1",
)
```

**Option 2: Local Model via Marie Engine**

```python
from marie.agent.llm_wrapper import MarieEngineLLMWrapper

llm = MarieEngineLLMWrapper(
    engine_name="qwen2_5_vl_7b",
    provider="vllm",
)
```

## Your First Agent

Let's create a simple calculator agent:

```python
from marie.agent import ReactAgent, register_tool
from marie.agent.llm_wrapper import OpenAICompatibleWrapper
import os
import json

# Step 1: Define tools
@register_tool("calculator")
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression.

    Args:
        expression: A math expression like "2 + 2" or "sqrt(16)"
    """
    import math

    # Safe evaluation with limited functions
    allowed = {
        'abs': abs, 'round': round, 'min': min, 'max': max,
        'sqrt': math.sqrt, 'pow': pow, 'sin': math.sin,
        'cos': math.cos, 'tan': math.tan, 'log': math.log,
        'pi': math.pi, 'e': math.e
    }

    try:
        result = eval(expression, {"__builtins__": {}}, allowed)
        return json.dumps({"result": result, "expression": expression})
    except Exception as e:
        return json.dumps({"error": str(e)})

@register_tool("get_time")
def get_time() -> str:
    """Get the current date and time."""
    from datetime import datetime
    now = datetime.now()
    return json.dumps({
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "day": now.strftime("%A")
    })

# Step 2: Create the LLM backend
llm = OpenAICompatibleWrapper(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
)

# Step 3: Create the agent
agent = ReactAgent(
    llm=llm,
    function_list=["calculator", "get_time"],
    name="CalculatorBot",
    description="A helpful assistant that can do math and tell time",
    system_message="""You are a helpful assistant with access to a calculator
    and time functions. Use the calculator for any math operations.
    Always show your work.""",
    max_iterations=5,
)

# Step 4: Run the agent
def chat(query: str):
    messages = [{"role": "user", "content": query}]

    for responses in agent.run(messages=messages):
        if responses:
            last_response = responses[-1]
            content = last_response.content if hasattr(last_response, 'content') else last_response.get('content', '')
            print(f"Assistant: {content}")

# Test it
chat("What's 15% of 85.50?")
chat("What time is it?")
chat("Calculate the square root of 144 plus 3 squared")
```

## Interactive Mode

For a conversational experience, implement an interactive loop:

```python
def run_interactive():
    messages = []
    print("Calculator Bot ready! Type 'quit' to exit.")

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() in ('quit', 'exit', 'q'):
            print("Goodbye!")
            break

        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})

        print("Assistant: ", end="", flush=True)
        for responses in agent.run(messages=messages):
            if responses:
                last = responses[-1]
                content = last.content if hasattr(last, 'content') else last.get('content', '')
                print(content)

        # Add assistant response to history
        if responses:
            for r in responses:
                msg = r if isinstance(r, dict) else r.model_dump()
                messages.append(msg)

if __name__ == "__main__":
    run_interactive()
```

## Running the Examples

Marie-AI includes ready-to-run examples in `examples/agents/`:

```bash
# Simple agent with basic tools
python examples/agents/agent_simple.py --task "Add 5 and 3"
python examples/agents/agent_simple.py --tui  # Interactive mode

# Full-featured assistant
python examples/agents/assistant_basic.py --query "What time is it in Tokyo?"

# Vision assistant (requires image)
python examples/agents/assistant_vision.py --image photo.jpg --query "Describe this"

# Planning agent for complex tasks
python examples/agents/planning_agent.py --task "Analyze the Python files in this directory"

# Document processing agent (requires Marie API)
python examples/agents/document_agent.py --document invoice.pdf --task "Extract all text"
```

## Testing Your Setup

Run the test suite to verify everything works:

```bash
# List available tests
python examples/agents/test_agents.py --list

# Quick smoke test
python examples/agents/test_agents.py --quick

# Test specific example
python examples/agents/test_agents.py --example agent_simple
```

## Common Patterns

### Adding Multiple Tools

```python
agent = ReactAgent(
    llm=llm,
    function_list=[
        "calculator",        # Registered tool by name
        "get_time",          # Another registered tool
        my_custom_tool,      # Tool instance
        some_function,       # Plain function (auto-wrapped)
    ],
    # ...
)
```

### Custom System Message

```python
agent = ReactAgent(
    llm=llm,
    function_list=tools,
    system_message="""You are a specialized financial analyst assistant.

    Guidelines:
    - Always verify calculations with the calculator tool
    - Format currency with two decimal places
    - Explain your reasoning step by step
    - Ask for clarification if the query is ambiguous
    """,
)
```

### Limiting Iterations

```python
agent = ReactAgent(
    llm=llm,
    function_list=tools,
    max_iterations=3,  # Stop after 3 tool-call cycles
)
```

## Troubleshooting

### "Tool not found" Error

Ensure tools are registered before creating the agent:

```python
# Register tool first
@register_tool("my_tool")
def my_tool():
    pass

# Then create agent
agent = ReactAgent(function_list=["my_tool"], ...)
```

### API Key Issues

Verify your environment variables:

```python
import os
print(f"API Key set: {bool(os.getenv('OPENAI_API_KEY'))}")
```

### Slow Responses

- Use `gpt-4o-mini` instead of `gpt-4o` for faster responses
- Reduce `max_iterations` if tool loops are unnecessary
- Consider local models for latency-sensitive applications

## Next Steps

- **[Core Concepts](./core-concepts.md)**: Understand the agent architecture
- **[Tool Development](./tool-development.md)**: Create custom tools
- **[Examples](./examples.md)**: Explore complete working examples