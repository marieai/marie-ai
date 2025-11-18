# Marie MCP Server

A lightweight [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server for the Marie AI document intelligence platform. This server enables AI assistants like Claude to interact with Marie AI's OCR, document classification, data extraction, and job management capabilities.

## Features

- **Document Processing**: Submit OCR extraction and template-based data extraction jobs
- **Job Management**: Monitor, control, and retrieve results from processing jobs
- **System Monitoring**: Check system health, capacity, and deployment status
- **Lightweight**: ~5MB package with no heavy dependencies (no full Marie AI package required)
- **Async-first**: Non-blocking I/O throughout
- **S3 Integration**: Automatic document upload/download to/from S3 storage
- **Retry Logic**: Built-in exponential backoff for resilient operations

## Installation

### Prerequisites

- Python >= 3.10
- Running Marie AI gateway instance
- AWS S3 credentials (for document storage)
- Marie AI API key

### Install via uv (recommended)

```bash
uv pip install marie-mcp
```

### Install from source

```bash
git clone https://github.com/marieai/marie-mcp-server.git
cd marie-mcp-server
pip install -e .
```

## Configuration

Create a `.env` file in your project directory:

```bash
# Marie Gateway Connection
MARIE_BASE_URL=http://localhost:5000
MARIE_API_KEY=your-api-key-here

# AWS S3 Configuration (required for document uploads)
AWS_ACCESS_KEY_ID=your-aws-access-key
AWS_SECRET_ACCESS_KEY=your-aws-secret-key
AWS_REGION=us-east-1
S3_BUCKET=marie

# Optional Settings
MARIE_REQUEST_TIMEOUT=300
MARIE_MAX_FILE_SIZE_MB=50
MARIE_OUTPUT_DIR=~/.marie-mcp/outputs
```

## Usage

### With Claude Desktop

Add to your Claude Desktop configuration (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "marie-ai": {
      "command": "uvx",
      "args": ["marie-mcp"],
      "env": {
        "MARIE_BASE_URL": "http://localhost:5000",
        "MARIE_API_KEY": "your-api-key-here",
        "AWS_ACCESS_KEY_ID": "your-aws-key",
        "AWS_SECRET_ACCESS_KEY": "your-aws-secret",
        "S3_BUCKET": "marie"
      }
    }
  }
}
```

Restart Claude Desktop and you'll see Marie AI tools available in the MCP menu.

### With LangChain

```python
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain.agents import AgentExecutor, create_react_agent
from langchain_anthropic import ChatAnthropic
from mcp import StdioServerParameters

# Load Marie MCP tools
marie_tools = await load_mcp_tools(
    StdioServerParameters(
        command="uvx",
        args=["marie-mcp"],
        env={
            "MARIE_BASE_URL": "http://localhost:5000",
            "MARIE_API_KEY": "your-key",
            "AWS_ACCESS_KEY_ID": "your-aws-key",
            "AWS_SECRET_ACCESS_KEY": "your-aws-secret",
        },
    )
)

# Create agent with Marie tools
llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")
agent = create_react_agent(llm, marie_tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=marie_tools)

# Run agent
result = await agent_executor.ainvoke({"input": "Extract text from invoice.pdf"})
```

### With OpenAI Agents SDK

```python
from agents import Agent
from agents.mcp import MCPServerStdio

# Create Marie MCP connection
marie_server = MCPServerStdio(
    command="uvx",
    args=["marie-mcp"],
    env={
        "MARIE_BASE_URL": "http://localhost:5000",
        "MARIE_API_KEY": "your-key",
        "AWS_ACCESS_KEY_ID": "your-aws-key",
        "AWS_SECRET_ACCESS_KEY": "your-aws-secret",
    },
)

# Create agent with Marie tools
agent = Agent(tools=marie_server.get_tools(), model="gpt-4")

# Use agent
result = agent.run("Extract data from medical_form.pdf using template 117183")
```

## Available Tools

### Document Processing

#### `extract_document_ocr`
Submit an OCR extraction job for text and layout extraction.

**Parameters:**
- `file_path` (str): Path to local document file
- `ref_id` (str): Unique document reference ID
- `ref_type` (str): Document type/category (default: "document")
- `enable_page_classifier` (bool): Enable page classification (default: False)
- `enable_page_splitter` (bool): Enable page splitting (default: False)
- `enable_template_matching` (bool): Enable template matching (default: False)
- `sla_hours` (int): Hard SLA deadline in hours (default: 4)

**Returns:** Job ID for tracking

**Example:**
```python
extract_document_ocr(
    file_path="/path/to/invoice.pdf", ref_id="invoice_12345", ref_type="invoice"
)
```

#### `extract_document_data`
Submit a template-based data extraction job (Gen5).

**Parameters:**
- `file_path` (str): Path to local document file
- `template_id` (str): Template/planner ID for extraction
- `ref_id` (str): Unique document reference ID
- `ref_type` (str): Document type/category (default: "document")
- `sla_hours` (int): Hard SLA deadline in hours (default: 4)

**Returns:** Job ID for tracking

**Example:**
```python
extract_document_data(
    file_path="/path/to/form.pdf",
    template_id="117183",
    ref_id="form_67890",
    ref_type="medical_form",
)
```

### Job Management

#### `get_job_status`
Get detailed status for a specific job.

**Parameters:**
- `job_id` (str): Job ID from submit operation

**Example:**
```python
get_job_status(job_id="job_123456")
```

#### `list_jobs`
List all jobs, optionally filtered by state.

**Parameters:**
- `state` (str, optional): Filter by state ('active', 'completed', 'failed', etc.)

**Example:**
```python
list_jobs(state="completed")
```

#### `stop_job`
Stop a running job.

**Parameters:**
- `job_id` (str): Job ID to stop

#### `delete_job`
Delete a job from the system.

**Parameters:**
- `job_id` (str): Job ID to delete

#### `get_job_results`
Download job results from S3 storage.

**Parameters:**
- `ref_id` (str): Document reference ID
- `ref_type` (str): Document type
- `output_path` (str): Local path for downloaded results

**Example:**
```python
get_job_results(
    ref_id="invoice_12345",
    ref_type="invoice",
    output_path="./results/invoice_12345.json",
)
```

### System Monitoring

#### `get_deployments`
Get information about active executors and deployments.

#### `get_capacity`
Get slot capacity and resource utilization.

#### `get_debug_info`
Get scheduler debug information and metrics.

#### `health_check`
Perform a health check on the Marie gateway.

## Architecture

### Job-Centric Workflow

All document processing goes through Marie AI's job submission system:

1. **Upload**: Document uploaded to S3
2. **Submit**: Job submitted with metadata (ref_id, ref_type, planner, SLA, etc.)
3. **Process**: Job runs asynchronously through Marie's DAG executor
4. **Results**: Results stored in S3 at `s3://{bucket}/{ref_type}/{ref_id}/`

### Components

```
marie-mcp-server/
├── src/marie_mcp/
│   ├── server.py              # MCP server entry point
│   ├── config.py              # Configuration management
│   ├── clients/
│   │   └── marie_client.py    # HTTP client for Marie gateway
│   ├── tools/
│   │   ├── document_processing.py   # OCR & data extraction tools
│   │   ├── job_management.py        # Job control tools
│   │   └── system_monitoring.py     # System info tools
│   └── utils/
│       ├── s3_utils.py        # S3 operations
│       ├── validators.py      # Input validation
│       ├── file_utils.py      # File operations
│       └── constants.py       # Shared constants
```

## Development

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/marieai/marie-mcp-server.git
cd marie-mcp-server

# Install with dev dependencies
pip install -e ".[dev]"

# Copy environment template
cp .env.example .env
# Edit .env with your credentials
```

### Running Tests

```bash
pytest tests/
```

### Code Quality

```bash
# Format code
black src/
isort src/

# Type checking
mypy src/
```

## Troubleshooting

### "MARIE_API_KEY environment variable is required"
Make sure your `.env` file contains `MARIE_API_KEY=your-key` or set it in your environment.

### "Failed to upload to S3"
Check that:
- AWS credentials are set correctly
- S3 bucket name is correct
- You have write permissions to the bucket

### "Connection refused" errors
Verify that:
- Marie gateway is running at the specified URL
- `MARIE_BASE_URL` is correct
- Network connectivity is available

### Job stuck in "pending" state
Check:
- Marie scheduler capacity: `get_capacity()`
- Scheduler debug info: `get_debug_info()`
- System health: `health_check()`

## API Reference

Full API documentation available at: https://docs.marieai.co/mcp-server

## Contributing

Contributions are welcome! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Support

- Documentation: https://docs.marieai.co
- GitHub Issues: https://github.com/marieai/marie-mcp-server/issues
- Email: support@marieai.co

## Related Projects

- [Marie AI](https://github.com/marieai/marie-ai) - Main Marie AI platform
- [Model Context Protocol](https://modelcontextprotocol.io/) - MCP specification
