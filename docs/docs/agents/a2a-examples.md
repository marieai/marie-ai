---
sidebar_position: 10
---

# A2A Examples

Working examples demonstrating A2A protocol integration with Marie AI.

## Example 1: Simple Echo Agent

A minimal A2A agent that echoes messages back.

```python
"""
Simple Echo Agent
Demonstrates basic A2A server implementation.

Run: python echo_agent.py
Test: curl http://localhost:9001/.well-known/agent.json
"""

from marie.agent import BasicAgent
from marie.agent.a2a import A2AExecutor
from marie.agent.llm_wrapper import OpenAICompatibleWrapper
import uvicorn
import os

# Create a simple agent
llm = OpenAICompatibleWrapper(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
)

agent = BasicAgent(
    llm=llm,
    name="Echo Agent",
    description="Echoes back your messages",
    system_message="You are an echo bot. Repeat what the user says, prefixed with 'Echo: '",
)

# Wrap with A2A executor
executor = A2AExecutor(
    agent=agent,
    url="http://localhost:9001",
    version="1.0.0",
)

# Create and run app
app = executor.create_app()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9001)
```

### Testing the Echo Agent

```bash
# Check agent card
curl http://localhost:9001/.well-known/agent.json | jq

# Send a message
curl -X POST http://localhost:9001/a2a \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "message/send",
    "params": {
      "message": {
        "role": "user",
        "parts": [{"kind": "text", "text": "Hello World!"}],
        "messageId": "msg-001"
      },
      "configuration": {"blocking": true}
    },
    "id": "1"
  }'
```

---

## Example 2: Multi-Agent Delegation

An orchestrator agent that delegates tasks to specialized agents.

```python
"""
Multi-Agent Orchestrator
Demonstrates calling multiple external A2A agents.
"""

import asyncio
from marie.agent import ReactAgent
from marie.agent.a2a import A2AClient, AgentRegistry
from marie.agent.llm_wrapper import OpenAICompatibleWrapper
from marie.agent.tools import tool
import os

# Initialize registry with external agents
registry = AgentRegistry()

async def setup_agents():
    await registry.register("weather", "http://weather-agent:9001")
    await registry.register("calculator", "http://calc-agent:9002")
    await registry.register("translator", "http://translate-agent:9003")

# Create tools that delegate to external agents
@tool("get_weather")
async def get_weather(city: str) -> str:
    """Get weather information for a city."""
    client = await registry.get_client("weather")
    response = await client.send_message(f"Weather in {city}")
    return response.parts[0].text

@tool("calculate")
async def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    client = await registry.get_client("calculator")
    response = await client.send_message(expression)
    return response.parts[0].text

@tool("translate")
async def translate(text: str, target_language: str) -> str:
    """Translate text to another language."""
    client = await registry.get_client("translator")
    response = await client.send_message(f"Translate to {target_language}: {text}")
    return response.parts[0].text

# Create orchestrator agent
llm = OpenAICompatibleWrapper(model="gpt-4o")

orchestrator = ReactAgent(
    llm=llm,
    name="Orchestrator",
    description="Coordinates multiple specialized agents",
    function_list=[get_weather, calculate, translate],
    system_message="""You are an orchestrator that delegates tasks to specialized agents.
    Use the available tools to handle weather queries, calculations, and translations.""",
)

async def main():
    await setup_agents()

    # Run a complex query
    messages = [{
        "role": "user",
        "content": "What's the weather in Paris? Also calculate 15% of 250 and translate 'Hello' to French."
    }]

    for responses in orchestrator.run(messages=messages):
        if responses:
            print(responses[-1].content)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Example 3: Streaming Responses

Handle streaming responses from A2A agents.

```python
"""
Streaming Client Example
Demonstrates SSE streaming with A2A agents.
"""

import asyncio
from marie.agent.a2a import A2AClient

async def stream_example():
    # Connect to a streaming-capable agent
    async with await A2AClient.from_url("http://streaming-agent:9003") as client:

        # Check streaming support
        if not client.supports_streaming:
            print("Agent doesn't support streaming")
            return

        print("Starting streaming request...")
        print("-" * 40)

        # Stream the response
        async for event in client.stream_message("Count from 1 to 10 slowly"):
            if event.kind == "status-update":
                print(f"[Status] {event.status.state}")

                # Check for messages in status
                if event.status.message:
                    for part in event.status.message.parts:
                        if part.kind == "text":
                            print(f"[Message] {part.text}")

            elif event.kind == "artifact-update":
                print(f"[Artifact] {event.artifact.name or 'unnamed'}")
                for part in event.artifact.parts:
                    if part.kind == "text":
                        print(f"  Content: {part.text}")

            # Check if final
            if hasattr(event, 'final') and event.final:
                print("-" * 40)
                print("Stream completed")
                break

if __name__ == "__main__":
    asyncio.run(stream_example())
```

### Streaming Server Implementation

```python
"""
Streaming Agent Server
Demonstrates SSE streaming from Marie agent.
"""

from marie.agent import ReactAgent
from marie.agent.a2a import A2AExecutor
from marie.agent.llm_wrapper import OpenAICompatibleWrapper
import uvicorn

llm = OpenAICompatibleWrapper(model="gpt-4o-mini")

agent = ReactAgent(
    llm=llm,
    name="Streaming Counter",
    description="Counts numbers with streaming updates",
)

executor = A2AExecutor(
    agent=agent,
    url="http://localhost:9003",
    capabilities={"streaming": True},
)

app = executor.create_app()
uvicorn.run(app, host="0.0.0.0", port=9003)
```

---

## Example 4: Tool Integration

Using A2A agents as tools within a Marie workflow.

```python
"""
A2A Remote Agent Tool Example
Use external A2A agents as tools in Marie agents.
"""

from marie.agent import ReactAgent
from marie.agent.tools import A2ARemoteAgentTool
from marie.agent.llm_wrapper import OpenAICompatibleWrapper
import os

# Create tools from external A2A agents
search_tool = A2ARemoteAgentTool(
    name="web_search",
    description="Search the web for current information",
    agent_url="http://search-agent:9000",
)

code_tool = A2ARemoteAgentTool(
    name="code_analyzer",
    description="Analyze code for bugs and improvements",
    agent_url="http://code-agent:9001",
    skill_id="analyze",  # Use specific skill
)

docs_tool = A2ARemoteAgentTool(
    name="doc_generator",
    description="Generate documentation for code",
    agent_url="http://docs-agent:9002",
)

# Create main agent with external tools
llm = OpenAICompatibleWrapper(
    model="gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY"),
)

assistant = ReactAgent(
    llm=llm,
    name="Developer Assistant",
    description="Helps with development tasks using specialized agents",
    function_list=[search_tool, code_tool, docs_tool],
    system_message="""You are a developer assistant with access to:
    - Web search for finding documentation and examples
    - Code analysis for reviewing code quality
    - Documentation generation for creating docs

    Use these tools to help developers with their tasks.""",
)

# Run example
messages = [{"role": "user", "content": "Analyze this Python code and generate documentation for it: def add(a, b): return a + b"}]

for responses in assistant.run(messages=messages):
    if responses:
        print(responses[-1].content)
```

---

## Example 5: Error Handling

Robust error handling for A2A operations.

```python
"""
Error Handling Example
Demonstrates proper error handling with A2A.
"""

import asyncio
from marie.agent.a2a import A2AClient, A2AAgentDiscovery
from marie.agent.a2a.errors import (
    A2AError,
    A2AClientError,
    AgentDiscoveryError,
    TaskNotFoundError,
)

async def robust_client():
    discovery = A2AAgentDiscovery()

    # Discovery with error handling
    try:
        card = await discovery.discover("http://agent:9000")
        print(f"Found agent: {card.name}")
    except AgentDiscoveryError as e:
        print(f"Discovery failed: {e}")
        return

    # Client operations with error handling
    try:
        async with await A2AClient.from_url("http://agent:9000") as client:

            # Send message with timeout handling
            try:
                response = await asyncio.wait_for(
                    client.send_message("Hello"),
                    timeout=30.0
                )
                print(f"Response: {response}")
            except asyncio.TimeoutError:
                print("Request timed out")

            # Task retrieval with not found handling
            try:
                task = await client.get_task("nonexistent-task-id")
            except TaskNotFoundError:
                print("Task not found")

    except A2AClientError as e:
        print(f"Client error: {e}")
    except A2AError as e:
        print(f"A2A error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

async def retry_with_backoff(
    func,
    max_retries: int = 3,
    base_delay: float = 1.0,
):
    """Retry function with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return await func()
        except A2AClientError as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            print(f"Attempt {attempt + 1} failed, retrying in {delay}s...")
            await asyncio.sleep(delay)

async def main():
    # Use retry wrapper
    async def make_request():
        async with await A2AClient.from_url("http://agent:9000") as client:
            return await client.send_message("Hello with retry")

    try:
        result = await retry_with_backoff(make_request)
        print(f"Success: {result}")
    except A2AClientError:
        print("All retries failed")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Example 6: Concurrent Agent Calls

Call multiple agents concurrently for parallel processing.

```python
"""
Concurrent Agent Calls
Demonstrates parallel A2A requests.
"""

import asyncio
from marie.agent.a2a import A2AClient, A2AAgentDiscovery

async def parallel_queries():
    # Discover multiple agents
    discovery = A2AAgentDiscovery()
    agents = await discovery.discover_many([
        "http://agent1:9001",
        "http://agent2:9002",
        "http://agent3:9003",
    ])

    print(f"Discovered {len(agents)} agents")

    # Create clients
    clients = [
        await A2AClient.from_url(agent.url)
        for agent in agents
    ]

    # Send queries in parallel
    queries = [
        "What is AI?",
        "Explain machine learning",
        "What is deep learning?",
    ]

    async def query_agent(client, query):
        try:
            response = await client.send_message(query)
            return {
                "agent": client.name,
                "query": query,
                "response": response.parts[0].text if response.parts else "No response",
            }
        except Exception as e:
            return {
                "agent": client.name,
                "query": query,
                "error": str(e),
            }

    # Run all queries concurrently
    tasks = [
        query_agent(client, query)
        for client, query in zip(clients, queries)
    ]

    results = await asyncio.gather(*tasks)

    # Print results
    for result in results:
        print(f"\n{result['agent']}:")
        print(f"  Query: {result['query']}")
        if 'error' in result:
            print(f"  Error: {result['error']}")
        else:
            print(f"  Response: {result['response'][:100]}...")

    # Cleanup
    for client in clients:
        await client.close()

if __name__ == "__main__":
    asyncio.run(parallel_queries())
```

---

## Running the Examples

### Prerequisites

```bash
# Install dependencies
pip install marie-ai[agent]

# Set API key for LLM
export OPENAI_API_KEY="your-key"
```

### Using Docker Compose

```yaml
# docker-compose.yml
version: '3.8'
services:
  echo-agent:
    build: .
    command: python examples/echo_agent.py
    ports:
      - "9001:9001"

  streaming-agent:
    build: .
    command: python examples/streaming_agent.py
    ports:
      - "9003:9003"

  orchestrator:
    build: .
    command: python examples/orchestrator.py
    depends_on:
      - echo-agent
      - streaming-agent
```

```bash
docker-compose up -d
```

### Testing with A2A Inspector

```bash
# Clone official inspector
git clone https://github.com/a2aproject/a2a-inspector
cd a2a-inspector
npm install && npm run dev

# Open http://localhost:3000
# Enter your agent URL to test
```

## Next Steps

- [A2A Integration Guide](./a2a-integration.md) - Overview and architecture
- [A2A API Reference](./a2a-api-reference.md) - Complete API documentation
- [Tool Development](./tool-development.md) - Creating custom tools
