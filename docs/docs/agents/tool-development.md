---
sidebar_position: 4
---

# Tool Development

Tools extend agent capabilities by providing access to functions, APIs, and external systems. This guide covers creating custom tools for your agents.

## Overview

There are two approaches to creating tools:

1. **Function-based tools**: Quick and simple, using the `@register_tool` decorator
2. **Class-based tools**: More control, state management, and complex logic

## Function-Based Tools

The simplest way to create tools using the `@register_tool` decorator.

### Basic Example

```python
from marie.agent import register_tool
import json

@register_tool("greet")
def greet(name: str) -> str:
    """
    Greet a person by name.

    Args:
        name: The name of the person to greet
    """
    return json.dumps({"greeting": f"Hello, {name}!"})
```

### With Custom Name and Description

```python
@register_tool("weather_lookup", description="Get weather for any city worldwide")
def get_weather(city: str, units: str = "celsius") -> str:
    """
    Get current weather for a city.

    Args:
        city: City name (e.g., "Tokyo", "New York")
        units: Temperature units - "celsius" or "fahrenheit"
    """
    # Implementation
    return json.dumps({"city": city, "temp": 72, "units": units})
```

### Type Hints and Documentation

The framework extracts parameter information from:
- **Type hints**: Define parameter types
- **Docstrings**: Provide descriptions (Google or NumPy style)
- **Default values**: Mark parameters as optional

```python
from typing import Optional, List

@register_tool("search_files")
def search_files(
    directory: str,
    pattern: str = "*",
    recursive: bool = False,
    max_results: Optional[int] = None
) -> str:
    """
    Search for files matching a pattern.

    Args:
        directory: Directory to search in
        pattern: Glob pattern (e.g., "*.py", "**/*.md")
        recursive: Whether to search subdirectories
        max_results: Maximum number of results to return

    Returns:
        JSON list of matching file paths
    """
    import glob
    import os

    search_pattern = os.path.join(directory, "**" if recursive else "", pattern)
    files = glob.glob(search_pattern, recursive=recursive)

    if max_results:
        files = files[:max_results]

    return json.dumps({"files": files, "count": len(files)})
```

### Returning Structured Data

Always return JSON-serializable strings:

```python
@register_tool("analyze_text")
def analyze_text(text: str) -> str:
    """Analyze text and return statistics."""
    words = text.split()
    return json.dumps({
        "word_count": len(words),
        "char_count": len(text),
        "unique_words": len(set(words)),
        "avg_word_length": sum(len(w) for w in words) / len(words) if words else 0
    })
```

### Error Handling

Return errors in a consistent format:

```python
@register_tool("read_file")
def read_file(path: str) -> str:
    """Read contents of a file."""
    try:
        with open(path, 'r') as f:
            content = f.read()
        return json.dumps({"content": content, "path": path})
    except FileNotFoundError:
        return json.dumps({"error": f"File not found: {path}"})
    except PermissionError:
        return json.dumps({"error": f"Permission denied: {path}"})
    except Exception as e:
        return json.dumps({"error": str(e)})
```

## Class-Based Tools

For more complex tools requiring state, configuration, or custom behavior.

### Basic Class Tool

```python
from marie.agent import AgentTool, ToolMetadata, ToolOutput
from pydantic import BaseModel, Field

class CalculatorInput(BaseModel):
    expression: str = Field(description="Math expression to evaluate")

class CalculatorTool(AgentTool):
    """A calculator tool that evaluates math expressions."""

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name=self.name,
            description="Evaluate mathematical expressions safely",
            fn_schema=CalculatorInput,
        )

    def call(self, expression: str, **kwargs) -> ToolOutput:
        import math

        allowed = {
            'abs': abs, 'round': round, 'min': min, 'max': max,
            'sqrt': math.sqrt, 'pow': pow, 'sin': math.sin,
            'cos': math.cos, 'log': math.log, 'pi': math.pi, 'e': math.e
        }

        try:
            result = eval(expression, {"__builtins__": {}}, allowed)
            return ToolOutput(
                content=str(result),
                tool_name=self.name,
                raw_input={"expression": expression},
                raw_output=result,
            )
        except Exception as e:
            return ToolOutput(
                content=f"Error: {str(e)}",
                tool_name=self.name,
                raw_input={"expression": expression},
                is_error=True,
            )
```

### Stateful Tool

Tools can maintain state across calls:

```python
class CounterTool(AgentTool):
    """A counter that persists across calls."""

    def __init__(self):
        self._value = 0

    @property
    def name(self) -> str:
        return "counter"

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name=self.name,
            description="Manage a persistent counter",
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["increment", "decrement", "reset", "get"],
                        "description": "Action to perform"
                    },
                    "amount": {
                        "type": "integer",
                        "description": "Amount for increment/decrement",
                        "default": 1
                    }
                },
                "required": ["action"]
            }
        )

    def call(self, action: str, amount: int = 1, **kwargs) -> ToolOutput:
        if action == "increment":
            self._value += amount
        elif action == "decrement":
            self._value -= amount
        elif action == "reset":
            self._value = 0
        # "get" just returns current value

        return ToolOutput(
            content=str(self._value),
            tool_name=self.name,
            raw_output={"value": self._value, "action": action},
        )
```

### Configurable Tool

Accept configuration during initialization:

```python
from dataclasses import dataclass

@dataclass
class APIConfig:
    base_url: str
    api_key: str
    timeout: int = 30

class APITool(AgentTool):
    """Tool that calls an external API."""

    def __init__(self, config: APIConfig):
        self.config = config

    @property
    def name(self) -> str:
        return "api_call"

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name=self.name,
            description=f"Call API at {self.config.base_url}",
            parameters={
                "type": "object",
                "properties": {
                    "endpoint": {"type": "string", "description": "API endpoint"},
                    "method": {"type": "string", "enum": ["GET", "POST"]},
                    "data": {"type": "object", "description": "Request body"}
                },
                "required": ["endpoint"]
            }
        )

    def call(self, endpoint: str, method: str = "GET", data: dict = None, **kwargs) -> ToolOutput:
        import requests

        url = f"{self.config.base_url}/{endpoint}"
        headers = {"Authorization": f"Bearer {self.config.api_key}"}

        try:
            if method == "GET":
                resp = requests.get(url, headers=headers, timeout=self.config.timeout)
            else:
                resp = requests.post(url, headers=headers, json=data, timeout=self.config.timeout)

            resp.raise_for_status()
            return ToolOutput(
                content=resp.text,
                tool_name=self.name,
                raw_output=resp.json() if resp.headers.get('content-type', '').startswith('application/json') else resp.text,
            )
        except Exception as e:
            return ToolOutput(
                content=f"API Error: {str(e)}",
                tool_name=self.name,
                is_error=True,
            )
```

### Async Tool

For I/O-bound operations:

```python
class AsyncWebTool(AgentTool):
    """Async tool for fetching web content."""

    @property
    def name(self) -> str:
        return "fetch_url"

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name=self.name,
            description="Fetch content from a URL",
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to fetch"}
                },
                "required": ["url"]
            }
        )

    def call(self, url: str, **kwargs) -> ToolOutput:
        """Synchronous fallback."""
        import requests
        resp = requests.get(url, timeout=10)
        return ToolOutput(content=resp.text[:5000], tool_name=self.name)

    async def acall(self, url: str, **kwargs) -> ToolOutput:
        """Async implementation."""
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                text = await resp.text()
                return ToolOutput(content=text[:5000], tool_name=self.name)
```

## Tool Registration

### Automatic Registration (Decorator)

```python
@register_tool("my_tool")
def my_tool():
    pass
# Tool is automatically registered in TOOL_REGISTRY
```

### Manual Registration

```python
from marie.agent.tools import TOOL_REGISTRY

# Register an instance
tool = CalculatorTool()
TOOL_REGISTRY.register("calculator", tool)

# Or register a class (instantiated on first use)
TOOL_REGISTRY.register("counter", CounterTool)
```

### Using Tools with Agents

```python
from marie.agent import ReactAgent

# By name (from registry)
agent = ReactAgent(
    llm=llm,
    function_list=["calculator", "counter"],
)

# By instance
agent = ReactAgent(
    llm=llm,
    function_list=[CalculatorTool(), CounterTool()],
)

# Mixed
agent = ReactAgent(
    llm=llm,
    function_list=[
        "calculator",           # From registry
        CounterTool(),          # Instance
        my_custom_function,     # Auto-wrapped function
    ],
)
```

## Tool Schema Definition

### Using Pydantic Models

```python
from pydantic import BaseModel, Field
from typing import List, Optional

class SearchParams(BaseModel):
    query: str = Field(description="Search query")
    filters: Optional[List[str]] = Field(default=None, description="Filter tags")
    limit: int = Field(default=10, ge=1, le=100, description="Max results")

class SearchTool(AgentTool):
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="search",
            description="Search the database",
            fn_schema=SearchParams,  # Pydantic model
        )
```

### Using JSON Schema Directly

```python
class ManualSchemaTool(AgentTool):
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="manual_tool",
            description="Tool with manual schema",
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name parameter"
                    },
                    "count": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 100
                    },
                    "options": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["name"]
            }
        )
```

## Best Practices

### 1. Clear Descriptions

Write descriptions that help the LLM understand when to use the tool:

```python
# Good
@register_tool("convert_currency")
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """
    Convert an amount from one currency to another using live exchange rates.

    Use this tool when the user asks about currency conversion, exchange rates,
    or wants to know the value of money in a different currency.

    Args:
        amount: The amount to convert
        from_currency: Source currency code (e.g., "USD", "EUR", "JPY")
        to_currency: Target currency code

    Returns:
        JSON with converted amount and exchange rate
    """
```

### 2. Input Validation

Validate inputs before processing:

```python
@register_tool("process_file")
def process_file(path: str) -> str:
    """Process a file."""
    import os

    # Validate path
    if not path:
        return json.dumps({"error": "Path is required"})

    if not os.path.exists(path):
        return json.dumps({"error": f"File not found: {path}"})

    if not os.path.isfile(path):
        return json.dumps({"error": f"Not a file: {path}"})

    # Security check
    abs_path = os.path.abspath(path)
    if ".." in path or path.startswith("/etc") or path.startswith("/root"):
        return json.dumps({"error": "Access denied"})

    # Process file...
```

### 3. Consistent Return Format

Use a consistent JSON structure:

```python
# Success
return json.dumps({
    "success": True,
    "result": actual_result,
    "metadata": {"took_ms": 150}
})

# Error
return json.dumps({
    "success": False,
    "error": "Error message",
    "error_code": "INVALID_INPUT"
})
```

### 4. Security Considerations

```python
# Whitelist allowed operations
ALLOWED_COMMANDS = {"ls", "pwd", "date", "whoami"}

@register_tool("run_command")
def run_command(command: str) -> str:
    """Run a whitelisted shell command."""
    cmd_name = command.split()[0]

    if cmd_name not in ALLOWED_COMMANDS:
        return json.dumps({
            "error": f"Command '{cmd_name}' not allowed",
            "allowed": list(ALLOWED_COMMANDS)
        })

    # Execute safely...
```

### 5. Timeout Handling

```python
import signal

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")

@register_tool("slow_operation")
def slow_operation(data: str) -> str:
    """Operation with timeout protection."""
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(30)  # 30 second timeout

    try:
        result = do_slow_thing(data)
        return json.dumps({"result": result})
    except TimeoutError:
        return json.dumps({"error": "Operation timed out after 30 seconds"})
    finally:
        signal.alarm(0)  # Cancel alarm
```

### 6. Logging

Add logging for debugging:

```python
import logging

logger = logging.getLogger(__name__)

@register_tool("important_operation")
def important_operation(data: str) -> str:
    """Critical operation with logging."""
    logger.info(f"Starting operation with data length: {len(data)}")

    try:
        result = process(data)
        logger.info(f"Operation completed successfully")
        return json.dumps({"result": result})
    except Exception as e:
        logger.error(f"Operation failed: {e}", exc_info=True)
        return json.dumps({"error": str(e)})
```

## Integration with Marie APIs

### Document Processing Tool

```python
import requests
import base64

@register_tool("ocr_document")
def ocr_document(file_path: str, mode: str = "multiline") -> str:
    """
    Extract text from a document using Marie OCR.

    Args:
        file_path: Path to document (image or PDF)
        mode: OCR mode - "word", "line", "multiline", "sparse"
    """
    api_url = os.getenv("MARIE_API_URL", "http://127.0.0.1:51000/api")

    try:
        with open(file_path, "rb") as f:
            file_data = base64.b64encode(f.read()).decode()

        response = requests.post(
            f"{api_url}/document/extract",
            json={
                "data": file_data,
                "mode": mode,
                "features": ["ocr"]
            },
            timeout=60
        )
        response.raise_for_status()
        return json.dumps(response.json())
    except Exception as e:
        return json.dumps({"error": str(e)})
```

## Testing Tools

```python
import pytest
from marie.agent.tools import TOOL_REGISTRY

def test_calculator_tool():
    tool = TOOL_REGISTRY.get("calculator")

    # Test basic operation
    result = tool.call(expression="2 + 2")
    assert not result.is_error
    assert "4" in result.content

    # Test error handling
    result = tool.call(expression="invalid")
    assert result.is_error

def test_tool_metadata():
    tool = TOOL_REGISTRY.get("calculator")
    meta = tool.metadata

    assert meta.name == "calculator"
    assert meta.description
    assert "expression" in meta.get_parameters_dict()["properties"]
```

## Next Steps

- **[Built-in Agents](./built-in-agents.md)**: Learn about available agent types
- **[Examples](./examples.md)**: See complete tool implementations
- **[API Reference](./api-reference.md)**: Detailed API documentation