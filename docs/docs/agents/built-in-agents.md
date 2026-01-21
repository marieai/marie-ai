---
sidebar_position: 5
---

# Built-in Agents

Marie-AI provides several pre-built agent types for different use cases. Each agent follows the ReAct pattern with variations suited to specific tasks.

## Agent Hierarchy

```text
BaseAgent (abstract)
├── BasicAgent          # Pass-through to LLM
├── ReactAgent      # ReAct-style with tool calling
├── FunctionCallingAgent # OpenAI-style parallel tool calls
├── ChatAgent           # Conversational, no tools
├── PlanAndExecuteAgent       # Multi-step planning
└── VisionDocumentAgent # Document understanding
    ├── DocumentExtractionAgent
    └── DocumentQAAgent
```

## ReactAgent

The general-purpose agent using the ReAct (Reasoning + Acting) pattern. Best for complex reasoning tasks that require tool use.

### Features

- Multi-step reasoning with iterative tool calling
- Detects tool calls from LLM responses (XML or Action format)
- Configurable iteration limits
- Streaming support

### Usage

```python
from marie.agent import ReactAgent
from marie.agent.llm_wrapper import OpenAICompatibleWrapper

llm = OpenAICompatibleWrapper(model="gpt-4o-mini", api_key="...")

agent = ReactAgent(
    llm=llm,
    function_list=["calculator", "get_time", "web_search"],
    name="ResearchAssistant",
    description="A helpful research assistant",
    system_message="""You are a research assistant. Use available tools
    to find information and perform calculations. Always cite your sources.""",
    max_iterations=10,
)

messages = [{"role": "user", "content": "What's the population of Tokyo?"}]
for responses in agent.run(messages=messages):
    print(responses[-1].content)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `llm` | BaseLLMWrapper | required | LLM backend instance |
| `function_list` | List | `[]` | Tools (names, instances, or functions) |
| `name` | str | `"Assistant"` | Agent name |
| `description` | str | `""` | Agent description |
| `system_message` | str | `""` | System prompt |
| `max_iterations` | int | `10` | Max tool-call cycles |

### Execution Flow

```text
1. Receive user message
2. Call LLM with tools schema
3. Parse response for tool calls
4. If tool call found:
   a. Execute tool
   b. Add result to conversation
   c. Go to step 2 (until max_iterations)
5. Yield final response
```

## PlanAndExecuteAgent

Creates explicit plans before execution. Ideal for complex multi-step workflows.

### Features

- Explicit planning phase before action
- Step-by-step execution with progress tracking
- Stops on "FINAL ANSWER" marker
- Better for long-horizon tasks

### Usage

```python
from marie.agent.agents import PlanAndExecuteAgent

agent = PlanAndExecuteAgent(
    llm=llm,
    function_list=[
        "list_files",
        "read_file",
        "analyze_code",
        "generate_report",
    ],
    name="CodeAnalyzer",
    system_message="""You are a code analysis assistant. When given a task:
    1. First create a PLAN with numbered steps
    2. Execute each step, noting progress
    3. End with FINAL ANSWER summarizing results""",
    max_iterations=15,
)

messages = [{"role": "user", "content": "Analyze all Python files in ./src"}]
for responses in agent.run(messages=messages):
    print(responses[-1].content)
```

### Planning Pattern

The agent generates responses in this format:

```text
PLAN:
1. List all Python files in ./src
2. Read each file to understand structure
3. Analyze code metrics (functions, classes, imports)
4. Generate summary report

STEP 1: Listing Python files
[Tool call: list_files]
Found 15 Python files.

STEP 2: Reading files
[Tool calls for each file]
...

FINAL ANSWER:
Analysis complete. Found 15 Python files with:
- 45 functions
- 12 classes
- Average complexity: 3.2
```

### Best Practices

- Use clear system messages that explain the planning format
- Set higher `max_iterations` for complex plans
- Provide tools that support incremental progress

## ChatAgent

Simple conversational agent without tools. Lightweight and fast.

### Features

- Pure LLM conversation
- No tool overhead
- Fastest response time
- Good for Q&A and chat

### Usage

```python
from marie.agent.agents import ChatAgent

agent = ChatAgent(
    llm=llm,
    name="Chatbot",
    system_message="""You are a friendly assistant. Be helpful,
    concise, and accurate. If you don't know something, say so.""",
)

messages = [{"role": "user", "content": "Explain quantum computing simply"}]
for responses in agent.run(messages=messages):
    print(responses[-1].content)
```

### When to Use

- Simple Q&A without external data
- Creative writing assistance
- Explanations and summaries
- When tools aren't needed

## FunctionCallingAgent

Optimized for native function calling (OpenAI, Claude). Supports parallel tool execution.

### Features

- Uses native `tool_calls` format
- Parallel tool execution in single turn
- Tool call ID tracking
- Best with GPT-4, Claude 3+

### Usage

```python
from marie.agent.agents import FunctionCallingAgent

agent = FunctionCallingAgent(
    llm=llm,
    function_list=["get_weather", "get_time", "search_web"],
    name="MultiTool",
    system_message="You can use multiple tools simultaneously.",
)

# Agent can call multiple tools in one response
messages = [{"role": "user", "content": "What's the weather and time in Tokyo and London?"}]
```

### Parallel Execution

```text
User: "What's the weather in Tokyo and London?"

LLM Response:
{
  "tool_calls": [
    {"id": "call_1", "function": {"name": "get_weather", "arguments": "{\"city\": \"Tokyo\"}"}},
    {"id": "call_2", "function": {"name": "get_weather", "arguments": "{\"city\": \"London\"}"}}
  ]
}

[Both tools executed, results added]

LLM Final Response:
"Tokyo: 72°F, sunny. London: 58°F, cloudy."
```

## VisionDocumentAgent

Specialized for visual document understanding. Integrates with Marie's document processing APIs.

### Features

- Multimodal input (images, PDFs, text)
- Task categorization and tool suggestion
- Integration with Marie OCR, classification, extraction
- Pattern-based fallbacks for common queries

### Usage

```python
from marie.agent.agents import VisionDocumentAgent

agent = VisionDocumentAgent(
    llm=llm,  # Should be vision-capable (GPT-4V, Qwen-VL)
    function_list=[
        "ocr",
        "classify_document",
        "extract_table",
        "extract_key_value",
        "extract_entities",
    ],
    name="DocumentAnalyzer",
    system_message="You analyze documents using available tools.",
)

# With image
messages = [{
    "role": "user",
    "content": [
        {"image": "/path/to/invoice.png"},
        {"text": "Extract all line items from this invoice"}
    ]
}]

for responses in agent.run(messages=messages):
    print(responses[-1].content)
```

### Task Categorization

The agent categorizes tasks and suggests appropriate tools:

```python
task_info = agent.get_task_info("Extract tables from this document")
# Returns:
# {
#     "category": "table_extraction",
#     "suggested_tools": ["detect_tables", "extract_table_structure"],
#     "available_tools": [...],
#     "pattern": "table"
# }
```

### Document Types Supported

| Type | Tools Used |
|------|------------|
| Invoices | `ocr`, `extract_key_value`, `extract_entities` |
| Forms | `ocr`, `extract_key_value` |
| Tables | `detect_tables`, `extract_table_structure` |
| General docs | `ocr`, `classify_document` |
| Receipts | `ocr`, `extract_entities` |

## DocumentExtractionAgent

Focused variant for extraction tasks.

### Features

- Optimized for data extraction
- Returns structured data
- Better for batch processing

### Usage

```python
from marie.agent.agents import DocumentExtractionAgent

agent = DocumentExtractionAgent(
    llm=llm,
    function_list=["ocr", "extract_key_value", "extract_table"],
)

messages = [{
    "role": "user",
    "content": [
        {"file": "/path/to/form.pdf"},
        {"text": "Extract name, date, and amount fields"}
    ]
}]
```

## DocumentQAAgent

Optimized for question answering about documents.

### Features

- Visual question answering
- Context-aware responses
- Pattern matching for common queries

### Usage

```python
from marie.agent.agents import DocumentQAAgent

agent = DocumentQAAgent(
    llm=llm,
    function_list=["ocr", "vqa", "extract_entities"],
)

messages = [{
    "role": "user",
    "content": [
        {"image": "/path/to/document.png"},
        {"text": "What is the total amount due?"}
    ]
}]
```

## Router Agent

Routes tasks to specialized sub-agents.

### Features

- Task analysis and routing
- Multiple specialized agents
- Automatic agent selection
- Response attribution

### Usage

```python
from marie.agent.agents.router import Router

# Define specialized agents
time_agent = ReactAgent(
    llm=llm,
    function_list=["get_time"],
    name="TimeAssistant",
)

math_agent = ReactAgent(
    llm=llm,
    function_list=["calculator"],
    name="MathAssistant",
)

# Create router
router = Router(
    llm=llm,
    agents={
        "time": time_agent,
        "math": math_agent,
    },
    name="TaskRouter",
    system_message="""Route requests to the appropriate agent:
    - Time/date questions → time
    - Math/calculations → math
    - Other → respond directly""",
)

messages = [{"role": "user", "content": "What's 25% of 80?"}]
# Router selects math_agent, returns: "[MathAssistant] 25% of 80 is 20"
```

## Choosing an Agent

| Use Case | Recommended Agent |
|----------|-------------------|
| General tasks with tools | ReactAgent |
| Complex multi-step workflows | PlanAndExecuteAgent |
| Simple Q&A | ChatAgent |
| OpenAI/Claude with parallel tools | FunctionCallingAgent |
| Document processing | VisionDocumentAgent |
| Data extraction | DocumentExtractionAgent |
| Document Q&A | DocumentQAAgent |
| Multi-agent orchestration | Router |

## Custom Agents

Create custom agents by extending `BaseAgent`:

```python
from marie.agent.base import BaseAgent
from marie.agent.message import Message

class MyCustomAgent(BaseAgent):
    def __init__(self, llm, custom_param, **kwargs):
        super().__init__(llm=llm, **kwargs)
        self.custom_param = custom_param

    def _run(self, messages, **kwargs):
        # Custom logic here
        response = self.llm.chat(messages)

        # Process response
        processed = self._process_response(response)

        yield [Message(role="assistant", content=processed)]

    def _process_response(self, response):
        # Custom processing
        return response
```

## Next Steps

- **[Examples](./examples.md)**: See complete agent implementations
- **[Tool Development](./tool-development.md)**: Create tools for your agents
- **[API Reference](./api-reference.md)**: Detailed API documentation