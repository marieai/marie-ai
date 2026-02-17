---
sidebar_position: 9
---

# A2A API Reference

Complete API documentation for Marie AI's A2A protocol implementation.

## A2AExecutor

Exposes Marie agents via the A2A protocol.

```python
from marie.agent.a2a import A2AExecutor

class A2AExecutor:
    def __init__(
        self,
        agent: BaseAgent,
        url: str,
        version: str = "1.0.0",
        name: str | None = None,
        description: str | None = None,
        provider: AgentProvider | None = None,
        documentation_url: str | None = None,
        icon_url: str | None = None,
    ) -> None:
        """
        Initialize A2A executor for a Marie agent.

        Args:
            agent: The Marie agent to expose
            url: Base URL where the agent will be served
            version: Agent version string
            name: Override agent name (defaults to agent.name)
            description: Override description (defaults to agent.description)
            provider: Provider information
            documentation_url: Link to documentation
            icon_url: Agent icon URL
        """

    def create_app(self) -> Starlette:
        """Create a Starlette application with A2A endpoints."""

    async def handle_request(self, request: JSONRPCRequest) -> JSONRPCResponse:
        """Handle a JSON-RPC request."""

    def get_agent_card(self) -> AgentCard:
        """Generate the agent card for discovery."""
```

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/.well-known/agent.json` | GET | Agent card discovery |
| `/a2a` | POST | JSON-RPC endpoint (sync) |
| `/a2a/stream` | POST | SSE streaming endpoint |

### Example

```python
from marie.agent import ReactAgent
from marie.agent.a2a import A2AExecutor
import uvicorn

agent = ReactAgent(llm=llm, name="MyAgent")
executor = A2AExecutor(agent=agent, url="http://localhost:8000")
app = executor.create_app()

uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## A2AClient

Client for calling external A2A agents.

```python
from marie.agent.a2a import A2AClient

class A2AClient:
    @classmethod
    async def from_url(
        cls,
        url: str,
        timeout: float = 60.0,
        stream_timeout: float = 300.0,
    ) -> A2AClient:
        """
        Create client by discovering agent at URL.

        Args:
            url: Agent base URL
            timeout: Request timeout in seconds
            stream_timeout: Streaming request timeout

        Returns:
            Configured A2AClient instance
        """

    async def send_message(
        self,
        content: str | Message,
        context_id: str | None = None,
        task_id: str | None = None,
        blocking: bool = True,
    ) -> Message | Task:
        """
        Send a message to the agent.

        Args:
            content: Message text or Message object
            context_id: Optional conversation context ID
            task_id: Optional existing task ID
            blocking: Wait for completion (default: True)

        Returns:
            Agent's response message or task object
        """

    async def stream_message(
        self,
        content: str | Message,
        context_id: str | None = None,
        task_id: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """
        Stream responses from the agent.

        Args:
            content: Message text or Message object
            context_id: Optional conversation context ID
            task_id: Optional existing task ID

        Yields:
            Stream events (status updates, artifacts)
        """

    async def get_task(
        self,
        task_id: str,
        history_length: int | None = None,
    ) -> Task:
        """
        Retrieve a task by ID.

        Args:
            task_id: Task identifier
            history_length: Number of history messages to include

        Returns:
            Task object with status and history
        """

    async def cancel_task(self, task_id: str) -> Task:
        """
        Cancel a running task.

        Args:
            task_id: Task identifier

        Returns:
            Updated task object
        """

    @property
    def supports_streaming(self) -> bool:
        """Check if agent supports streaming."""

    @property
    def supports_push_notifications(self) -> bool:
        """Check if agent supports push notifications."""

    @property
    def name(self) -> str:
        """Agent name from card."""

    @property
    def description(self) -> str | None:
        """Agent description from card."""

    @property
    def skills(self) -> list[AgentSkill]:
        """Agent skills from card."""
```

### Example

```python
async with await A2AClient.from_url("http://agent:9000") as client:
    # Simple message
    response = await client.send_message("Hello!")
    print(response)

    # Streaming
    async for event in client.stream_message("Process this"):
        if event.kind == "status-update":
            print(f"Status: {event.status.state}")
        elif event.kind == "artifact-update":
            print(f"Artifact: {event.artifact.name}")
```

---

## A2AAgentDiscovery

Service for discovering and caching A2A agents.

```python
from marie.agent.a2a import A2AAgentDiscovery

class A2AAgentDiscovery:
    def __init__(self, cache_ttl: int = 3600) -> None:
        """
        Initialize discovery service.

        Args:
            cache_ttl: Cache time-to-live in seconds (default: 1 hour)
        """

    async def discover(
        self,
        url: str,
        force_refresh: bool = False,
    ) -> AgentCard:
        """
        Discover agent at URL.

        Args:
            url: Agent base URL
            force_refresh: Bypass cache

        Returns:
            Agent card

        Raises:
            AgentDiscoveryError: If discovery fails
        """

    async def get_client(self, url: str) -> A2AClient:
        """
        Get a client for the agent at URL.

        Args:
            url: Agent base URL

        Returns:
            Configured A2AClient
        """

    async def discover_many(
        self,
        urls: list[str],
        ignore_errors: bool = True,
    ) -> list[AgentCard]:
        """
        Discover multiple agents concurrently.

        Args:
            urls: List of agent URLs
            ignore_errors: Skip failed discoveries

        Returns:
            List of discovered agent cards
        """

    def list_agents(self) -> list[CachedAgent]:
        """List all cached agents."""

    def get_cached(self, url: str) -> AgentCard | None:
        """Get cached agent card without HTTP request."""

    def invalidate(self, url: str) -> None:
        """Remove agent from cache."""

    def clear_cache(self) -> None:
        """Clear all cached entries."""
```

---

## AgentRegistry

Named registry for managing external agents.

```python
from marie.agent.a2a import AgentRegistry

class AgentRegistry:
    def __init__(self, discovery: A2AAgentDiscovery | None = None) -> None:
        """Initialize registry with optional discovery service."""

    async def register(self, name: str, url: str) -> AgentCard:
        """
        Register an agent by name.

        Args:
            name: Friendly name for the agent
            url: Agent base URL

        Returns:
            Discovered agent card
        """

    def unregister(self, name: str) -> None:
        """Remove agent from registry."""

    async def get_card(self, name: str) -> AgentCard:
        """Get agent card by name."""

    async def get_client(self, name: str) -> A2AClient:
        """Get client for named agent."""

    def list_registered(self) -> list[str]:
        """List all registered agent names."""

    async def discover_all(self) -> dict[str, AgentCard]:
        """Refresh all registered agents."""
```

### Example

```python
registry = AgentRegistry()

# Register agents
await registry.register("weather", "http://weather:9000")
await registry.register("calculator", "http://calc:9001")

# Use by name
client = await registry.get_client("weather")
response = await client.send_message("Weather in Paris?")
```

---

## A2ARemoteAgentTool

Use external A2A agents as tools in Marie agents.

```python
from marie.agent.tools import A2ARemoteAgentTool

class A2ARemoteAgentTool:
    def __init__(
        self,
        name: str,
        description: str,
        agent_url: str,
        skill_id: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        """
        Create a tool that delegates to an external A2A agent.

        Args:
            name: Tool name
            description: Tool description for the LLM
            agent_url: External agent URL
            skill_id: Specific skill to invoke (optional)
            timeout: Request timeout
        """

    async def __call__(self, query: str) -> str:
        """Execute the tool with the given query."""
```

### Example

```python
from marie.agent import ReactAgent
from marie.agent.tools import A2ARemoteAgentTool

# Create remote agent tool
search_tool = A2ARemoteAgentTool(
    name="web_search",
    description="Search the web for information",
    agent_url="http://search-agent:9000",
)

# Use in agent
agent = ReactAgent(
    llm=llm,
    name="ResearchAssistant",
    function_list=[search_tool],
)
```

---

## Type Definitions

### AgentCard

```python
class AgentCard(BaseModel):
    name: str
    description: str | None = None
    url: str
    version: str = "1.0.0"
    skills: list[AgentSkill] | None = None
    capabilities: AgentCapabilities | None = None
    default_input_modes: list[str] | None = None
    default_output_modes: list[str] | None = None
    provider: AgentProvider | None = None
    documentation_url: str | None = None
    icon_url: str | None = None
```

### AgentSkill

```python
class AgentSkill(BaseModel):
    id: str
    name: str
    description: str | None = None
    tags: list[str] | None = None
    examples: list[str] | None = None
    input_modes: list[str] | None = None
    output_modes: list[str] | None = None
```

### AgentCapabilities

```python
class AgentCapabilities(BaseModel):
    streaming: bool = False
    push_notifications: bool = False
    state_transition_history: bool = False
```

### Task

```python
class Task(BaseModel):
    id: str
    context_id: str
    status: TaskStatus
    history: list[Message] | None = None
    artifacts: list[Artifact] | None = None
```

### TaskState

```python
class TaskState(str, Enum):
    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"
    COMPLETED = "completed"
    CANCELED = "canceled"
    FAILED = "failed"
    REJECTED = "rejected"
    AUTH_REQUIRED = "auth-required"
```

### Message

```python
class Message(BaseModel):
    role: Role  # "user" or "agent"
    parts: list[Part]
    message_id: str
    task_id: str | None = None
    context_id: str | None = None
```

### Part Types

```python
class TextPart(BaseModel):
    kind: Literal["text"] = "text"
    text: str

class FilePart(BaseModel):
    kind: Literal["file"] = "file"
    file: FileContent

class DataPart(BaseModel):
    kind: Literal["data"] = "data"
    data: dict[str, Any]
```

---

## Error Classes

```python
from marie.agent.a2a.errors import (
    A2AError,              # Base exception
    A2AClientError,        # Client-side errors
    A2AServerError,        # Server-side errors
    A2AProtocolError,      # Protocol violations
    TaskNotFoundError,     # Task doesn't exist
    TaskNotCancelableError,# Task can't be canceled
    AgentDiscoveryError,   # Discovery failed
)
```

### Error Codes

| Code | Name | Description |
|------|------|-------------|
| -32600 | Invalid Request | Malformed JSON-RPC request |
| -32601 | Method Not Found | Unknown method |
| -32602 | Invalid Params | Invalid method parameters |
| -32001 | Task Not Found | Task ID doesn't exist |
| -32002 | Task Not Cancelable | Task is in terminal state |
| -32004 | Unsupported Operation | Feature not supported |
| -32005 | Content Type Not Supported | Invalid content type |
