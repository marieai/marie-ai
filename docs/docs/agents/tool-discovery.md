---
sidebar_position: 5
---

# Dynamic Tool Discovery

When agents have access to large tool catalogs, exposing every tool to the LLM becomes expensive and can reduce accuracy. Marie-AI provides a hierarchical discovery system that uses BM25-based search to dynamically discover relevant tools at runtime.

## Overview

The hierarchical tool discovery system operates in two layers:

```text
User Message
      │
      ▼
┌─────────────────────────────────┐
│   Layer 1: Skill Discovery      │
│   SkillRouter with BM25 search  │
│   Matches message → skill       │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│   Layer 2: Tool Discovery       │
│   SearchableToolset with BM25   │
│   search_tools → relevant tools │
└──────────────┬──────────────────┘
               │
               ▼
        Agent Execution
```

### Key Benefits

- **Reduced context usage**: Only relevant tools are included in the LLM prompt
- **Improved accuracy**: Less noise from irrelevant tools
- **Scalable**: Works with 100s of tools without performance degradation
- **500x faster**: Uses bm25s library for efficient search

## SearchableToolset

Instead of exposing all tools upfront, agents start with a single `search_tools` function and dynamically discover relevant tools using BM25-based keyword search.

### Basic Usage

```python
from marie.agent import ReactAgent
from marie.agent.tools import SearchableToolset

# Create a searchable toolset from your tool catalog
toolset = SearchableToolset(
    tools=["calculator", "weather", "search", "email", ...],  # 100+ tools
    passthrough_threshold=5,
    top_k=3,
)

# Pass directly to agent - clean API, no extra parameters
agent = ReactAgent(
    llm=llm,
    tools=toolset,
)

# The agent discovers tools dynamically
for responses in agent.run([{"role": "user", "content": "Calculate 15% of 85"}]):
    print(responses[-1].content)
```

When using `SearchableToolset`:
- The agent receives only a `search_tools` function initially
- When it needs a tool, it calls `search_tools("capability I need")`
- Matching tools are discovered and made available for the next LLM call
- The agent then calls the discovered tool

### Configuration Options

```python
toolset = SearchableToolset(
    tools=my_tools,
    passthrough_threshold=5,   # Passthrough mode if ≤5 tools
    top_k=3,                   # Return top 3 matches per search
)

agent = ReactAgent(llm=llm, tools=toolset)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tools` | required | Tool catalog (names, instances, callables) |
| `passthrough_threshold` | `5` | Tool count for passthrough mode |
| `top_k` | `3` | Max tools returned per search |

### Passthrough Mode

When the tool catalog is small (≤ `passthrough_threshold`), all tools are exposed directly without the search layer:

```python
# Small catalog - all tools exposed directly
toolset = SearchableToolset(
    tools=["calculator", "weather"],  # Only 2 tools
    passthrough_threshold=5,
)
agent = ReactAgent(llm=llm, tools=toolset)
# Agent sees: calculator, weather (no search_tools needed)

# Large catalog - search_tools exposed
toolset = SearchableToolset(
    tools=[...100 tools...],
    passthrough_threshold=5,
)
agent = ReactAgent(llm=llm, tools=toolset)
# Agent sees: search_tools (discovers others dynamically)
```

### Using SearchableToolset Directly

For advanced use cases, you can use `SearchableToolset` independently:

```python
from marie.agent.tools import SearchableToolset

# Create toolset
toolset = SearchableToolset(
    tools=["calculator", "weather", "search", my_custom_tool],
    passthrough_threshold=5,
    top_k=3,
)

# Check mode
if toolset.is_passthrough:
    print("All tools exposed directly")
else:
    print("Using search_tools discovery")

# Get tools to expose to LLM
exposed = toolset.get_exposed_tools()

# Search directly
results = toolset.search("calculate math expression")
for tool, score in results:
    print(f"{tool.name}: {score:.3f}")

# Get a specific tool
calc = toolset.get_tool("calculator")
```

### How It Works

1. **Index Building**: Tool metadata (name, description, parameters) is tokenized and indexed using BM25

2. **Search**: When `search_tools` is called, the query is matched against the index

3. **Registration**: Discovered tools are registered via callback for execution

4. **Schema Refresh**: The agent detects new tools and includes them in the next LLM call

```python
# The search_tools function exposed to the LLM
def search_tools(query: str) -> str:
    """
    Search for available tools by describing what capability you need.
    Returns matching tools that you can then call.

    Args:
        query: Natural language description of needed capability

    Returns:
        JSON with found tools and their descriptions
    """
```

Example interaction:

```text
User: "What's the weather in Tokyo?"

Agent (turn 1):
  Thought: I need a weather tool
  Action: search_tools
  Action Input: "get weather forecast for city"

search_tools result:
  {
    "found": 2,
    "tools": [
      {"name": "get_weather", "description": "Get current weather...", "relevance_score": 0.847},
      {"name": "forecast", "description": "Get weather forecast...", "relevance_score": 0.623}
    ]
  }

Agent (turn 2):
  Thought: Found get_weather, I'll use it
  Action: get_weather
  Action Input: {"city": "Tokyo"}

get_weather result: {"temp": 22, "condition": "sunny"}

Agent (turn 3):
  The weather in Tokyo is 22°C and sunny.
```

## Skill Router with BM25

Layer 1 of the discovery system matches user messages to skills using BM25 search.

### Enabling BM25 Search

```python
from marie.agent.skills import SkillRouter, SKILL_REGISTRY

# Create router with BM25 (default)
router = SkillRouter(
    registry=SKILL_REGISTRY,
    auto_match_threshold=0.3,
    use_bm25=True,  # Default
)

# Route a message
context = await router.route("Extract data from this invoice")
if context.skill:
    print(f"Matched skill: {context.skill.name}")
    print(f"Score: {context.matched_score}")
```

### Search Skills

```python
# Search for skills by query
results = router.search_skills(
    query="document extraction OCR",
    top_k=5,
    tags=["document"],       # Optional tag filter
    provider="openai",       # Optional provider filter
)

for skill, score in results:
    print(f"{skill.name}: {score:.2f}")
```

### Skill Index Management

```python
# Rebuild index after adding skills
router.rebuild_index()

# Check index status
from marie.agent.skills import SkillSearchIndex

index = SkillSearchIndex()
print(f"BM25 available: {index.is_available}")
print(f"Indexed skills: {index.num_skills}")
```

## Installation

The BM25 search functionality requires the `bm25s` library:

```bash
pip install 'bm25s[core]>=0.2.0'
```

Or install with Marie-AI:

```bash
pip install 'marieai[standard]'
```

If `bm25s` is not installed, the system automatically falls back to linear keyword matching.

## Architecture

### Component Hierarchy

```text
BaseAgent
    │
    ├── SkillRouter (Layer 1)
    │   └── SkillSearchIndex (BM25)
    │
    └── SearchableToolset (Layer 2)
        └── BM25 Index
```

### Schema Refresh

When tools are discovered dynamically, the agent needs to include them in subsequent LLM calls:

```python
# Inside agent loop
while iteration < max_iterations:
    # Check if tools changed (e.g., via search_tools)
    if self._check_tools_dirty():
        # Refresh tool definitions for next LLM call
        functions = self._get_tool_definitions(use_exposed=False)

    # Continue with LLM call...
```

### Per-Request Cleanup

Discovered tools are automatically cleaned up after each request:

```python
# In BaseAgent.run()
try:
    original_tools = self.function_map.copy()

    if self._searchable_toolset:
        self._searchable_toolset.clear_dynamic_tools()

    # Execute agent logic...

finally:
    # Always restore original tools
    self.function_map = original_tools
```

## Complete Example

```python
from marie.agent import ReactAgent, register_tool
from marie.agent.tools import SearchableToolset
from marie.agent.llm_wrapper import OpenAICompatibleWrapper
import json

# Register many tools
@register_tool("calculator")
def calculator(expression: str) -> str:
    """Evaluate mathematical expressions."""
    return json.dumps({"result": eval(expression)})

@register_tool("get_weather")
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return json.dumps({"city": city, "temp": 22, "condition": "sunny"})

@register_tool("search_web")
def search_web(query: str) -> str:
    """Search the web for information."""
    return json.dumps({"results": [f"Result for: {query}"]})

@register_tool("send_email")
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email."""
    return json.dumps({"status": "sent", "to": to})

@register_tool("create_calendar_event")
def create_calendar_event(title: str, date: str) -> str:
    """Create a calendar event."""
    return json.dumps({"event_id": "evt_123", "title": title})

@register_tool("translate_text")
def translate_text(text: str, target_language: str) -> str:
    """Translate text to another language."""
    return json.dumps({"translated": f"[{target_language}] {text}"})

# ... imagine 100+ more tools ...

# Create LLM backend
llm = OpenAICompatibleWrapper(
    model="gpt-4o-mini",
    api_key="sk-...",
)

# Create searchable toolset (Haystack-style)
toolset = SearchableToolset(
    tools=[
        "calculator", "get_weather", "search_web",
        "send_email", "create_calendar_event", "translate_text",
        # ... all tools
    ],
    passthrough_threshold=5,
    top_k=3,
)

# Create agent with toolset
agent = ReactAgent(
    llm=llm,
    tools=toolset,  # Clean API - just pass the toolset
    system_message="You are a helpful assistant with access to many tools.",
)

# Run agent - it will discover tools as needed
messages = [{"role": "user", "content": "What's 15% of 85?"}]

for responses in agent.run(messages):
    print(responses[-1].content)
```

## Best Practices

### 1. Write Descriptive Tool Metadata

The search quality depends on tool descriptions:

```python
# Good - descriptive and specific
@register_tool("currency_converter")
def convert_currency(amount: float, from_curr: str, to_curr: str) -> str:
    """
    Convert money between currencies using live exchange rates.

    Use this for currency conversion, exchange rate queries,
    or calculating values in different currencies.

    Args:
        amount: Amount to convert
        from_curr: Source currency code (USD, EUR, GBP, JPY)
        to_curr: Target currency code
    """

# Bad - vague description
@register_tool("convert")
def convert(a, b, c):
    """Convert stuff."""
```

### 2. Use Appropriate Thresholds

```python
# Small catalog (< 10 tools) - passthrough is fine
toolset = SearchableToolset(tools=small_catalog, passthrough_threshold=10)
agent = ReactAgent(llm=llm, tools=toolset)

# Large catalog (100+ tools) - lower top_k for precision
toolset = SearchableToolset(
    tools=large_catalog,
    passthrough_threshold=5,
    top_k=3,
)
agent = ReactAgent(llm=llm, tools=toolset)
```

### 3. Group Related Tools

When tools are related, their combined descriptions improve search:

```python
# Calendar tools will match "schedule", "meeting", "event", etc.
calendar_tools = ["create_event", "update_event", "delete_event", "list_events"]

# Weather tools will match "weather", "forecast", "temperature", etc.
weather_tools = ["get_weather", "get_forecast", "get_alerts"]
```

### 4. Monitor Discovery

Log tool discovery for debugging:

```python
import logging
logging.getLogger("marie.agent.tools.searchable").setLevel(logging.DEBUG)

# Logs will show:
# DEBUG: Built BM25 index for 150 tools
# DEBUG: Registered discovered tool: calculator
# DEBUG: Refreshed tool definitions: 4 tools
```

## Comparison with Haystack

Marie-AI's `SearchableToolset` is inspired by [Haystack's SearchableToolset](https://haystack.deepset.ai/release-notes/2.25.0) but adds:

| Feature | Haystack | Marie-AI |
|---------|----------|----------|
| BM25 Search | rank-bm25 | bm25s (500x faster) |
| Skill Layer | - | SkillRouter (Layer 1) |
| Provider Filtering | - | Filter by LLM provider |
| Schema Refresh | - | `_tools_dirty` flag |
| Per-request Cleanup | - | Automatic |

## Next Steps

- **[Tool Development](./tool-development.md)**: Create tools with good descriptions
- **[Built-in Agents](./built-in-agents.md)**: Agent types that support tool discovery
- **[Examples](./examples.md)**: Complete working examples
