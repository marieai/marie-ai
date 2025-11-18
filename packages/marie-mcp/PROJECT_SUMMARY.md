# Marie MCP Server - Project Summary

## Overview

This is a lightweight Model Context Protocol (MCP) server that enables AI assistants (like Claude) to interact with the Marie AI document intelligence platform. The server provides document processing, job management, and system monitoring capabilities through a clean MCP interface.

## Key Design Decisions

### 1. Job-Centric Architecture
- All document processing goes through Marie's job submission system (`/api/v1/invoke`)
- Documents are uploaded to S3 before job submission
- Jobs run asynchronously with SLA tracking
- Results are stored in S3 for retrieval

### 2. Lightweight Client
- **No dependency on marie package** (which is multi-GB)
- Pure `httpx` HTTP client
- Only ~5MB installed size
- Suitable for subprocess execution via MCP

### 3. Two Primary Document Processing Tools
Instead of separate endpoints for each operation:

- `extract_document_ocr` - OCR extraction (queue: `extract`, planner: `extract`)
- `extract_document_data` - Template-based extraction (queue: `gen5_extract`, planner: `{template_id}`)

Both tools:
1. Upload document to S3
2. Submit job with metadata
3. Return job_id for tracking

### 4. Complete Job Lifecycle Support
- `submit_job` - Submit processing jobs
- `get_job_status` - Check job progress
- `list_jobs` - List jobs by state
- `stop_job` - Cancel running jobs
- `delete_job` - Remove jobs
- `get_job_results` - Download results from S3

### 5. System Monitoring
- `get_deployments` - Active executors
- `get_capacity` - Resource availability
- `get_debug_info` - Scheduler diagnostics
- `health_check` - Service health

## Project Structure

```
marie-mcp-server/
├── src/marie_mcp/
│   ├── __init__.py
│   ├── server.py                    # MCP server entry point
│   ├── config.py                    # Configuration management
│   ├── clients/
│   │   ├── __init__.py
│   │   └── marie_client.py          # HTTP client (no marie dependency)
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── document_processing.py   # OCR & data extraction
│   │   ├── job_management.py        # Job control
│   │   └── system_monitoring.py     # System info
│   └── utils/
│       ├── __init__.py
│       ├── constants.py             # Shared constants
│       ├── validators.py            # Input validation
│       ├── file_utils.py            # Async file operations
│       └── s3_utils.py              # S3 upload/download
├── tests/
│   ├── __init__.py
│   └── conftest.py                  # Test fixtures
├── examples/
│   ├── claude_desktop_config.json   # Claude Desktop setup
│   ├── langchain_integration.py     # LangChain example
│   ├── openai_agents_integration.py # OpenAI Agents example
│   └── direct_usage.py              # Direct client usage
├── pyproject.toml                   # Package metadata
├── README.md                        # Documentation
├── CONTRIBUTING.md                  # Contribution guide
├── LICENSE                          # MIT License
├── .env.example                     # Environment template
└── .gitignore                       # Git ignore rules
```

## Dependencies

### Core (Runtime)
- `mcp>=1.6.0` - MCP framework
- `httpx>=0.24.0` - Async HTTP client
- `python-dotenv>=1.0.0` - Environment management
- `aiofiles>=24.1.0` - Async file I/O
- `tenacity>=9.1.2` - Retry logic
- `pydantic>=2.0.0` - Data validation
- `boto3>=1.26.0` - S3 operations

### Development
- `pytest>=7.0.0` - Testing
- `pytest-asyncio>=0.21.0` - Async test support
- `pytest-httpx>=0.22.0` - HTTP mocking
- `black>=23.0.0` - Code formatting
- `isort>=5.12.0` - Import sorting
- `mypy>=1.0.0` - Type checking

## Configuration

### Required Environment Variables
```bash
MARIE_BASE_URL=http://localhost:5000    # Marie gateway URL
MARIE_API_KEY=your-api-key              # Required for authentication
AWS_ACCESS_KEY_ID=your-aws-key          # Required for S3
AWS_SECRET_ACCESS_KEY=your-aws-secret   # Required for S3
```

### Optional Settings
```bash
AWS_REGION=us-east-1
S3_BUCKET=marie
MARIE_REQUEST_TIMEOUT=300
MARIE_MAX_FILE_SIZE_MB=50
MARIE_MAX_RETRIES=3
MARIE_OUTPUT_DIR=~/.marie-mcp/outputs
```

## Installation

```bash
# Via uv (recommended)
uv pip install marie-mcp

# From source
git clone https://github.com/marieai/marie-mcp-server.git
cd marie-mcp-server
pip install -e .
```

## Usage

### Claude Desktop
Add to `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "marie-ai": {
      "command": "uvx",
      "args": ["marie-mcp"],
      "env": {
        "MARIE_BASE_URL": "http://localhost:5000",
        "MARIE_API_KEY": "your-key",
        "AWS_ACCESS_KEY_ID": "your-aws-key",
        "AWS_SECRET_ACCESS_KEY": "your-aws-secret"
      }
    }
  }
}
```

### LangChain
```python
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import StdioServerParameters

marie_tools = await load_mcp_tools(
    StdioServerParameters(command="uvx", args=["marie-mcp"], env={...})
)
```

### OpenAI Agents
```python
from agents.mcp import MCPServerStdio

marie_server = MCPServerStdio(command="uvx", args=["marie-mcp"], env={...})
```

## API Reference

### Document Processing Tools

#### extract_document_ocr
- **Purpose**: Submit OCR extraction job
- **Parameters**: file_path, ref_id, ref_type, enable_page_classifier, enable_page_splitter, enable_template_matching, sla_hours
- **Returns**: Job ID

#### extract_document_data
- **Purpose**: Submit template-based data extraction job
- **Parameters**: file_path, template_id, ref_id, ref_type, sla_hours
- **Returns**: Job ID

### Job Management Tools

#### get_job_status
- **Purpose**: Get detailed job status
- **Parameters**: job_id
- **Returns**: Job state, progress, errors

#### list_jobs
- **Purpose**: List all jobs or filter by state
- **Parameters**: state (optional)
- **Returns**: Array of job objects

#### stop_job
- **Purpose**: Cancel running job
- **Parameters**: job_id
- **Returns**: Confirmation

#### delete_job
- **Purpose**: Delete job from system
- **Parameters**: job_id
- **Returns**: Confirmation

#### get_job_results
- **Purpose**: Download job results from S3
- **Parameters**: ref_id, ref_type, output_path
- **Returns**: Downloaded file path

### System Monitoring Tools

#### get_deployments
- **Purpose**: Get active executors info
- **Returns**: Deployment details

#### get_capacity
- **Purpose**: Get resource availability
- **Returns**: Slot capacity info

#### get_debug_info
- **Purpose**: Get scheduler diagnostics
- **Returns**: Debug information

#### health_check
- **Purpose**: Check service health
- **Returns**: Health status

## Workflow Example

1. **Submit Job**:
   ```
   extract_document_ocr(file_path="invoice.pdf", ref_id="inv_001", ref_type="invoice")
   → Returns: {"job_id": "job_123"}
   ```

2. **Check Status**:
   ```
   get_job_status(job_id="job_123")
   → Returns: {"state": "running", "progress": 50}
   ```

3. **List Jobs**:
   ```
   list_jobs(state="completed")
   → Returns: [{"job_id": "job_123", "state": "completed", ...}]
   ```

4. **Get Results**:
   ```
   get_job_results(ref_id="inv_001", ref_type="invoice", output_path="./results/inv_001.json")
   → Downloads results from S3
   ```

## Testing

```bash
# Run all tests
pytest tests/

# With coverage
pytest --cov=marie_mcp tests/
```

## Future Enhancements

Potential additions:
- SSE event subscription for real-time job updates
- Template management tools (list templates, get schema)
- Batch job submission
- Job result caching
- WebSocket support for streaming results
- Integration tests with live Marie gateway

## References

- [Marie AI Documentation](https://docs.marieai.co)
- [Model Context Protocol](https://modelcontextprotocol.io/)
- [FastMCP Documentation](https://github.com/jlowin/fastmcp)

## License

MIT License - See LICENSE file

## Support

- GitHub Issues: https://github.com/marieai/marie-mcp-server/issues
- Email: support@marieai.co
- Documentation: https://docs.marieai.co/mcp-server
