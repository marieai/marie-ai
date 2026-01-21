---
sidebar_position: 6
---

# Examples

This guide covers the example agents included with Marie-AI. All examples are located in `examples/agents/`.

## Overview

| Example | Description | Use Case |
|---------|-------------|----------|
| `agent_simple.py` | Basic tools demo | Learning fundamentals |
| `assistant_basic.py` | Full-featured assistant | General-purpose tasks |
| `assistant_vision.py` | Vision-language agent | Image analysis |
| `planning_agent.py` | Multi-step planning | Complex workflows |
| `document_agent.py` | Marie API integration | Document processing |
| `vision_document_agent.py` | Advanced document agent | Document understanding |
| `multi_agent_router.py` | Task routing | Multi-agent orchestration |

## Running Examples

All examples support multiple modes:

```bash
# Task mode (single query)
python examples/agents/agent_simple.py --task "Add 5 and 3"

# Interactive mode
python examples/agents/agent_simple.py --tui

# With specific backend
python examples/agents/assistant_basic.py --backend openai --query "What time is it?"
```

## Simple Agent

**File:** `examples/agents/agent_simple.py`

The simplest example demonstrating basic tool creation and agent usage.

### Tools Included

```python
# Function-based tools
@register_tool("add")
def add(a: int, b: int) -> str:
    """Add two numbers."""
    return json.dumps({"result": a + b})

@register_tool("multiply")
def multiply(a: int, b: int) -> str:
    """Multiply two numbers."""
    return json.dumps({"result": a * b})

@register_tool("get_time")
def get_time() -> str:
    """Get current date and time."""
    now = datetime.now()
    return json.dumps({
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S")
    })

@register_tool("list_files")
def list_files(directory: str, pattern: str = "*") -> str:
    """List files matching a pattern."""
    # Implementation...

@register_tool("read_file")
def read_file(path: str) -> str:
    """Read file contents."""
    # Implementation...
```

### Class-Based Tool Example

```python
class CounterTool(AgentTool):
    """Stateful counter demonstration."""

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
                        "enum": ["increment", "decrement", "reset", "get"]
                    },
                    "amount": {"type": "integer", "default": 1}
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
        return ToolOutput(content=str(self._value), tool_name=self.name)
```

### Running

```bash
python examples/agents/agent_simple.py --task "Add 5 and 3"
python examples/agents/agent_simple.py --task "What time is it?"
python examples/agents/agent_simple.py --task "Increment counter by 5, then multiply by 2"
python examples/agents/agent_simple.py --tui
```

## Basic Assistant

**File:** `examples/agents/assistant_basic.py`

A comprehensive assistant with real-world tools.

### Tools Included

| Tool | Description |
|------|-------------|
| `get_current_time` | Get time in any timezone |
| `calculator` | Safe math evaluation |
| `run_shell_command` | Whitelisted shell commands |
| `read_file` | Read files with security checks |
| `write_file` | Write to allowed paths |
| `WebFetchTool` | Fetch and parse web content |
| `SystemInfoTool` | Get system information |

### Key Features

**Secure Calculator:**
```python
@register_tool("calculator")
def calculator(expression: str) -> str:
    """
    Evaluate math expressions safely.
    Supports: +, -, *, /, **, %, sqrt, sin, cos, tan, log
    Also handles: "15% of 85" format
    """
    # Percentage pattern handling
    match = re.match(r'(\d+(?:\.\d+)?)\s*%\s*of\s*(\d+(?:\.\d+)?)', expression)
    if match:
        pct, value = float(match.group(1)), float(match.group(2))
        return json.dumps({"result": (pct / 100) * value})

    # Safe eval with allowed functions only
    allowed = {'sqrt': math.sqrt, 'sin': math.sin, ...}
    result = eval(expression, {"__builtins__": {}}, allowed)
    return json.dumps({"result": result})
```

**Secure Shell Commands:**
```python
ALLOWED_COMMANDS = {"ls", "pwd", "whoami", "date", "cat", "head", "tail",
                    "wc", "grep", "find", "echo", "which", "uname"}

@register_tool("run_shell_command")
def run_shell_command(command: str, timeout: int = 30) -> str:
    """Run whitelisted shell commands only."""
    cmd_parts = shlex.split(command)
    if cmd_parts[0] not in ALLOWED_COMMANDS:
        return json.dumps({"error": f"Command not allowed: {cmd_parts[0]}"})
    # Execute with timeout...
```

### Running

```bash
python examples/agents/assistant_basic.py --query "What time is it in Tokyo?"
python examples/agents/assistant_basic.py --query "Calculate 15% tip on $85.50"
python examples/agents/assistant_basic.py --query "List files in current directory"
python examples/agents/assistant_basic.py --tui
```

## Vision Assistant

**File:** `examples/agents/assistant_vision.py`

Image analysis and manipulation using vision-language models.

### Tools Included

| Tool | Description |
|------|-------------|
| `image_info` | Get image metadata (size, format, EXIF) |
| `crop_image` | Crop by region or coordinates |
| `resize_image` | Resize with aspect ratio preservation |
| `convert_image` | Format conversion (PNG, JPG, WebP, etc.) |

### Multimodal Input

```python
# Build message with image
messages = [{
    "role": "user",
    "content": [
        {"image": image_path},
        {"text": query}
    ]
}]

# Agent processes both image and text
for responses in agent.run(messages=messages):
    print(responses[-1].content)
```

### Image Processing Example

```python
@register_tool("crop_image")
def crop_image(image_path: str, region: str, output_path: str = None) -> str:
    """
    Crop an image.

    Args:
        image_path: Source image path
        region: Predefined (top, bottom, left, right, center) or "x1,y1,x2,y2"
        output_path: Output path (optional)
    """
    from PIL import Image
    img = Image.open(image_path)
    w, h = img.size

    regions = {
        "top": (0, 0, w, h//2),
        "bottom": (0, h//2, w, h),
        "left": (0, 0, w//2, h),
        "right": (w//2, 0, w, h),
        "center": (w//4, h//4, 3*w//4, 3*h//4)
    }

    if region in regions:
        box = regions[region]
    else:
        box = tuple(map(int, region.split(",")))

    cropped = img.crop(box)
    out_path = output_path or f"cropped_{os.path.basename(image_path)}"
    cropped.save(out_path)

    return json.dumps({"output": out_path, "size": cropped.size})
```

### Running

```bash
python examples/agents/assistant_vision.py --image photo.jpg --query "Describe this image"
python examples/agents/assistant_vision.py --image doc.png --query "Crop the top half"
python examples/agents/assistant_vision.py --tui
```

## Planning Agent

**File:** `examples/agents/planning_agent.py`

Multi-step task planning and execution.

### Tools Included

| Tool | Description |
|------|-------------|
| `list_files` | List files with recursion support |
| `read_file` | Read file contents |
| `write_file` | Write files |
| `analyze_text` | Text statistics (words, patterns) |
| `analyze_code` | Code metrics (functions, classes) |
| `create_csv` | Create CSV files |
| `read_csv` | Read CSV files |
| `generate_report` | Generate markdown/text reports |
| `run_calculation` | Math with variables |

### Planning Pattern

```python
# System message encourages planning
system_message = """You are a planning assistant. For complex tasks:

1. First create a PLAN with numbered steps
2. Execute each step, noting STEP N: [action]
3. End with FINAL ANSWER summarizing results

Example:
PLAN:
1. List all Python files
2. Analyze each file
3. Generate summary report

STEP 1: Listing files
[Tool calls and results]

STEP 2: Analyzing code
[Tool calls and results]

FINAL ANSWER:
Found 10 Python files with 45 functions and 12 classes.
"""
```

### Code Analysis Tool

```python
@register_tool("analyze_code")
def analyze_code(file_path: str) -> str:
    """
    Analyze code file for metrics.

    Returns:
        - Python: imports, functions, classes, comments, docstrings
        - JavaScript: imports, functions, classes
        - Java: imports, classes, methods
    """
    ext = os.path.splitext(file_path)[1].lower()

    with open(file_path, 'r') as f:
        content = f.read()

    if ext == '.py':
        return json.dumps({
            "language": "python",
            "lines": len(content.splitlines()),
            "imports": len(re.findall(r'^import |^from .* import', content, re.M)),
            "functions": len(re.findall(r'^def \w+', content, re.M)),
            "classes": len(re.findall(r'^class \w+', content, re.M)),
            "comments": len(re.findall(r'#.*$', content, re.M)),
            "docstrings": len(re.findall(r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'', content)),
        })
    # Similar for other languages...
```

### Running

```bash
python examples/agents/planning_agent.py --task "Analyze all Python files in ./src"
python examples/agents/planning_agent.py --task "Create a summary report of this project"
python examples/agents/planning_agent.py --tui
```

## Document Agent

**File:** `examples/agents/document_agent.py`

Document processing using Marie API.

### Tools Included

| Tool | Description |
|------|-------------|
| `OCRTool` | Text extraction via Marie OCR |
| `DocumentClassifierTool` | Document type classification |
| `TableExtractorTool` | Table detection and extraction |
| `NERExtractorTool` | Named entity recognition |
| `KeyValueExtractorTool` | Form field extraction |
| `document_info` | File metadata |
| `check_marie_status` | API health check |

### Marie API Integration

```python
class OCRTool(AgentTool):
    """Extract text from documents using Marie OCR API."""

    def __init__(self, config: MarieAPIConfig):
        self.config = config

    def call(self, document_path: str, mode: str = "multiline", **kwargs) -> ToolOutput:
        with open(document_path, "rb") as f:
            file_data = base64.b64encode(f.read()).decode()

        response = requests.post(
            f"{self.config.api_base_url}/document/extract",
            json={
                "data": file_data,
                "queue_id": self.config.queue_id,
                "mode": mode,
                "features": ["ocr", "page_classifier", "page_splitter"]
            },
            headers={"Authorization": f"Bearer {self.config.api_key}"},
            timeout=60
        )
        response.raise_for_status()
        return ToolOutput(content=json.dumps(response.json()), tool_name=self.name)
```

### Document Message Format

```python
messages = [{
    "role": "user",
    "content": [
        {"file": document_path},  # PDF, image, etc.
        {"text": task}
    ]
}]
```

### Running

```bash
# Requires running Marie API server
python examples/agents/document_agent.py --document invoice.pdf --task "Extract all text"
python examples/agents/document_agent.py --document form.png --task "Extract name and date fields"
python examples/agents/document_agent.py --demo
python examples/agents/document_agent.py --tui
```

## Vision Document Agent

**File:** `examples/agents/vision_document_agent.py`

Advanced document understanding with task categorization.

### Tools Included

Function-based tools wrapping Marie APIs:
- `ocr(image, language, mode)` - Text extraction
- `detect_tables(image)` - Table detection
- `extract_table_structure(image, table_bbox)` - Detailed table parsing
- `classify_document(image)` - Document classification
- `extract_key_value(image, fields)` - Form field extraction
- `extract_entities(text, image)` - NER with fallback patterns
- `vqa(image, question)` - Visual question answering
- `detect_layout(image)` - Layout analysis

### Agent Types

```python
# Standard vision document agent
agent = init_vision_document_agent(agent_type="vision")

# Extraction-focused agent
agent = init_vision_document_agent(agent_type="extraction")

# Question-answering agent
agent = init_vision_document_agent(agent_type="qa")
```

### Task Categorization

```python
# Agent automatically categorizes tasks
task_info = agent.get_task_info("Extract tables from this document")

# Returns:
# {
#     "category": "table_extraction",
#     "suggested_tools": ["detect_tables", "extract_table_structure"],
#     "available_tools": ["ocr", "detect_tables", ...],
#     "pattern": "table"
# }
```

### Demos

```python
# Table extraction demo
demo_table_extraction()

# Invoice processing demo
demo_invoice_processing()

# Document Q&A demo
demo_document_qa()

# Form extraction demo
demo_form_extraction()
```

### Running

```bash
python examples/agents/vision_document_agent.py --task "Extract tables" --image doc.png
python examples/agents/vision_document_agent.py --demo table
python examples/agents/vision_document_agent.py --demo invoice
python examples/agents/vision_document_agent.py --tui
```

## Multi-Agent Router

**File:** `examples/agents/multi_agent_router.py`

Task routing to specialized sub-agents.

### Architecture

```text
User Request
     │
     ▼
┌─────────────┐
│   Router    │
│   (LLM)     │
└──────┬──────┘
       │ Analyzes request
       ▼
   ┌───┴───┐
   │       │
   ▼       ▼
┌─────┐ ┌─────┐
│Time │ │Math │ ... more agents
│Agent│ │Agent│
└─────┘ └─────┘
   │       │
   └───┬───┘
       ▼
  Response with
  Attribution
```

### Specialized Agents

```python
# Time specialist
time_agent = ReactAgent(
    llm=llm,
    function_list=["get_time"],
    name="TimeAssistant",
    description="Handles time and date queries",
)

# Math specialist
math_agent = ReactAgent(
    llm=llm,
    function_list=["calculator"],
    name="MathAssistant",
    description="Handles calculations",
)

# File specialist
file_agent = ReactAgent(
    llm=llm,
    function_list=["list_files"],
    name="FileAssistant",
    description="Handles file operations",
)

# Create router
router = Router(
    llm=llm,
    agents={
        "time": time_agent,
        "math": math_agent,
        "files": file_agent,
    },
    name="TaskRouter",
    system_message="""Analyze requests and route to appropriate agent:
    - Time/date questions → time
    - Math/calculations → math
    - File operations → files
    - General questions → respond directly
    """,
)
```

### Running

```bash
python examples/agents/multi_agent_router.py --task "What time is it?"
python examples/agents/multi_agent_router.py --task "Calculate 25 * 4"
python examples/agents/multi_agent_router.py --task "List Python files"
python examples/agents/multi_agent_router.py --tui
```

## Test Suite

**File:** `examples/agents/test_agents.py`

Comprehensive test framework for all examples.

### Usage

```bash
# List all available tests
python examples/agents/test_agents.py --list

# Show usage examples
python examples/agents/test_agents.py --usage

# Run all tests
python examples/agents/test_agents.py --all

# Quick smoke tests
python examples/agents/test_agents.py --quick

# Test specific example
python examples/agents/test_agents.py --example agent_simple

# Test with specific backend
python examples/agents/test_agents.py --example assistant_basic --backend openai
```

### Test Coverage

| Example | Tests | Description |
|---------|-------|-------------|
| agent_simple | 6 | All basic tools |
| assistant_basic | 8 | Real-world scenarios |
| planning_agent | 6 | Multi-step workflows |
| multi_agent_router | 7 | Routing scenarios |
| document_agent | 5 | Marie API (requires server) |

## Configuration

### Environment Variables

Create `examples/agents/.env`:

```bash
# OpenAI
OPENAI_API_KEY=your-openai-api-key
OPENAI_API_BASE=https://api.openai.com/v1

# Marie API (for document agents)
MARIE_API_URL=http://127.0.0.1:51000/api
MARIE_API_KEY=your-marie-api-key
MARIE_QUEUE_ID=0000-0000-0000-0000
MARIE_VQA_EXECUTOR_URL=http://127.0.0.1:51000/api/vqa
```

### Backend Selection

```python
# Environment variable
os.environ["AGENT_BACKEND"] = "openai"  # or "marie"

# Command line
python example.py --backend openai

# In code
llm = OpenAICompatibleWrapper(model="gpt-4o-mini", ...)  # OpenAI
llm = MarieEngineLLMWrapper(engine_name="qwen2_5_vl_7b", ...)  # Local
```

## Next Steps

- **[Tool Development](./tool-development.md)**: Create your own tools
- **[Built-in Agents](./built-in-agents.md)**: Understand agent types
- **[API Reference](./api-reference.md)**: Detailed API documentation