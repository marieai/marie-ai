---
sidebar_position: 8
---

# A2A Protocol Integration

Marie AI supports Google's Agent-to-Agent (A2A) protocol for bidirectional agent interoperability. This enables Marie agents to communicate with external A2A-compatible agents and expose Marie agents to external systems.

## Overview

The A2A integration provides:

- **Server Mode**: Expose Marie agents to external A2A clients via standard endpoints
- **Client Mode**: Call external A2A agents from Marie workflows
- **Discovery**: Automatic agent capability detection via `/.well-known/agent.json`
- **Streaming**: Real-time SSE responses for long-running tasks

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                          A2A Architecture                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   External A2A Agents          Marie AI Platform          Marie Agents   │
│   ┌─────────────────┐         ┌──────────────────┐      ┌────────────┐  │
│   │ Google ADK      │◀───────▶│                  │◀────▶│ ReactAgent │  │
│   │ Agent           │         │   A2AExecutor    │      └────────────┘  │
│   └─────────────────┘         │                  │      ┌────────────┐  │
│   ┌─────────────────┐         │   - Agent Card   │◀────▶│ Planning   │  │
│   │ Other A2A       │◀───────▶│   - JSON-RPC     │      │ Agent      │  │
│   │ Services        │         │   - SSE Streaming│      └────────────┘  │
│   └─────────────────┘         └──────────────────┘      ┌────────────┐  │
│                                       ▲                  │ Custom     │  │
│                                       │                  │ Agents     │  │
│                               ┌───────┴───────┐         └────────────┘  │
│                               │  A2AClient    │                          │
│                               │  - Discovery  │                          │
│                               │  - Messaging  │                          │
│                               │  - Tasks      │                          │
│                               └───────────────┘                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Exposing a Marie Agent via A2A

```python
from marie.agent import ReactAgent
from marie.agent.a2a import A2AExecutor
from marie.agent.llm_wrapper import OpenAICompatibleWrapper

# Create your Marie agent
llm = OpenAICompatibleWrapper(model="gpt-4o-mini")
agent = ReactAgent(
    llm=llm,
    name="WeatherBot",
    description="Provides weather information",
)

# Wrap with A2A executor
executor = A2AExecutor(
    agent=agent,
    url="http://localhost:8000",
    version="1.0.0",
)

# Start the server (using FastAPI/Starlette)
app = executor.create_app()

# Agent card available at: http://localhost:8000/.well-known/agent.json
# A2A endpoint at: http://localhost:8000/a2a
```

### Calling an External A2A Agent

```python
from marie.agent.a2a import A2AClient

async def call_external_agent():
    # Discover and connect to external agent
    async with await A2AClient.from_url("http://external-agent:9000") as client:
        # Send a message and get response
        result = await client.send_message("What's the weather in Tokyo?")
        print(result)

        # Or stream responses
        async for event in client.stream_message("Count from 1 to 10"):
            print(event)
```

### Using External Agents as Tools

```python
from marie.agent import ReactAgent
from marie.agent.tools import A2ARemoteAgentTool

# Create a tool that delegates to an external agent
weather_tool = A2ARemoteAgentTool(
    name="weather_agent",
    description="Gets weather information from external service",
    agent_url="http://weather-service:9000",
)

# Use in your agent
agent = ReactAgent(
    llm=llm,
    name="AssistantBot",
    function_list=[weather_tool],
)
```

## Components

| Component | Description |
|-----------|-------------|
| `A2AExecutor` | Wraps Marie agents for A2A protocol exposure |
| `A2AClient` | Client for calling external A2A agents |
| `A2AAgentDiscovery` | Discovers and caches agent cards |
| `AgentRegistry` | Named registry for managing multiple external agents |
| `A2ARemoteAgentTool` | Use external A2A agents as tools in Marie agents |

## Protocol Support

Marie's A2A implementation supports:

| Feature | Status |
|---------|--------|
| Agent Card Discovery | ✅ Full support |
| JSON-RPC 2.0 | ✅ Full support |
| Message Send | ✅ Full support |
| Message Stream (SSE) | ✅ Full support |
| Task Management | ✅ Full support |
| Push Notifications | ⚠️ Partial |
| Authentication | ⚠️ Basic |

## Configuration

### Environment Variables

```bash
# Default timeout for A2A requests (seconds)
A2A_REQUEST_TIMEOUT=60

# Default timeout for streaming requests (seconds)
A2A_STREAM_TIMEOUT=300

# Cache TTL for discovered agent cards (seconds)
A2A_DISCOVERY_CACHE_TTL=3600
```

### YAML Configuration

```yaml
# config/a2a.yml
a2a:
  server:
    enabled: true
    host: 0.0.0.0
    port: 8000

  client:
    default_timeout: 60
    stream_timeout: 300

  discovery:
    cache_ttl: 3600
    auto_refresh: true

  agents:
    - name: weather-service
      url: http://weather:9000
    - name: calculator
      url: http://calc:9001
```

## Studio Integration

Marie Studio provides a comprehensive UI for managing external A2A agents.

### Accessing External Agents

1. Navigate to **Agents** in the main sidebar
2. Click the **External A2A** tab

### Adding an External Agent

1. Click the **Add Agent** button
2. Enter the agent's base URL (e.g., `https://my-agent.example.com`)
3. Click **Discover** to fetch the agent card
4. Optionally override the agent name
5. Click **Register Agent** to add it to your repository

### Agent Card Display

Once registered, you can view detailed information about each agent:

- **Status**: Online, offline, or error state
- **Version**: Agent version from the card
- **Skills**: List of capabilities with descriptions and examples
- **Capabilities**: Streaming, push notifications, state history
- **Input/Output Modes**: Supported content types

### Health Monitoring

- **Individual Refresh**: Click the refresh icon on any agent row
- **Bulk Health Check**: Click **Health Check** to verify all agents
- **Status Indicators**: Visual badges showing connection status
- **Error Details**: View error messages when agents are unreachable

### Testing Agents

The detail panel includes a test interface:

1. Select an agent to view its details
2. Enter a test message in the **Test Connectivity** section
3. Click **Send Test Message**
4. View the response time and success/error status

For streaming-capable agents, the test interface shows real-time task status updates.

### Managing Agents

- **Refresh**: Update the agent card from the source
- **Documentation**: Quick link to agent documentation (if provided)
- **Remove**: Unregister the agent (does not affect the remote agent)

### Best Practices

- Use descriptive names when overriding auto-discovered names
- Regularly run health checks to identify connectivity issues
- Review agent capabilities before integrating into workflows
- Test agents with sample messages before production use

## Message Conversion Layer

When integrating with Marie's agent system, A2A messages are automatically converted:

| Marie Message | A2A Message |
|---------------|-------------|
| `role: "user"` | `role: "user"` |
| `role: "assistant"` | `role: "agent"` |
| `role: "system"` | `role: "user"` (with metadata) |
| `role: "tool"` | `role: "agent"` (with data part) |
| `content: str` | `parts: [TextPart]` |
| `content: [ContentItem]` | `parts: [TextPart \| FilePart \| DataPart]` |

This conversion is handled automatically by the `MessageConverter` class.

## Next Steps

- [A2A API Reference](./a2a-api-reference.md) - Detailed class documentation
- [A2A Examples](./a2a-examples.md) - Working code examples
- [Tool Development](./tool-development.md) - Creating custom tools
