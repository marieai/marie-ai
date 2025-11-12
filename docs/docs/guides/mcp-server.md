---
sidebar_position: 5
title: MCP Server
description: Integrate Marie AI with AI assistants using Model Context Protocol
---

# Marie MCP Server

The Marie MCP Server is a lightweight bridge that enables AI assistants like Claude to access Marie AI's powerful document intelligence capabilities through the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/).

## Overview

Marie MCP Server provides AI assistants with tools to:
- **Extract text from documents** using advanced OCR
- **Extract structured data** using predefined templates
- **Manage processing jobs** with real-time status tracking
- **Monitor system capacity** and resource availability

### Key Features

- 🚀 **Lightweight**: Only ~5MB (no heavy Marie AI dependencies)
- 🔄 **Async Job Processing**: Submit jobs and track progress
- 📊 **Real-time Monitoring**: Check system health and capacity
- 🔐 **Secure**: Bearer token authentication with AWS S3 integration
- 🎯 **Multi-format Support**: PDF, TIFF, JPEG, PNG, Office documents
- 🤖 **AI Assistant Ready**: Works with Claude Desktop, LangChain, OpenAI Agents

## Prerequisites

Before you begin, ensure you have:

1. **Marie AI Gateway**: A running Marie AI gateway instance
   ```bash
   marie server --start --uses config/service/marie-dev.yml
   ```

2. **API Key**: Your Marie AI API key for authentication

3. **AWS S3 Access**: Credentials for document storage
   - AWS Access Key ID
   - AWS Secret Access Key
   - S3 Bucket name

4. **Python 3.10+**: Required for running the MCP server

5. **UV Package Manager**: For installation (recommended)
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

## Installation

### Via UV (Recommended)

```bash
uv pip install marie-mcp
```

### Via pip

```bash
pip install marie-mcp
```

### From Source

```bash
git clone https://github.com/marieai/marie-ai.git
cd marie-ai/packages/marie-mcp
pip install -e .
```

## Configuration

### Environment Variables

Create a `.env` file in your working directory:

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

### Test Configuration

Verify your configuration:

```bash
# Test Marie gateway connection
curl http://localhost:5000/check?text=health

# Test AWS S3 access
aws s3 ls s3://marie/

# Run MCP server
marie-mcp
```

You should see:
```
============================================================
Marie MCP Server
============================================================

Marie MCP Server Configuration:
  Gateway URL: http://localhost:5000
  API Key: **********xxxx
  S3 Bucket: marie
  ...

✓ All tools registered successfully
Starting MCP server...
Ready to accept connections via STDIO
```

## Integration with Claude Desktop

### Step 1: Locate Configuration File

Find your Claude Desktop configuration file:

- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
- **Linux**: `~/.config/Claude/claude_desktop_config.json`

### Step 2: Add Marie MCP Server

Edit the configuration file:

```json
{
  "mcpServers": {
    "marie-ai": {
      "command": "uvx",
      "args": ["marie-mcp"],
      "env": {
        "MARIE_BASE_URL": "http://localhost:5000",
        "MARIE_API_KEY": "your-api-key-here",
        "AWS_ACCESS_KEY_ID": "your-aws-access-key",
        "AWS_SECRET_ACCESS_KEY": "your-aws-secret-key",
        "AWS_REGION": "us-east-1",
        "S3_BUCKET": "marie"
      }
    }
  }
}
```

### Step 3: Restart Claude Desktop

Completely quit and restart Claude Desktop for changes to take effect.

### Step 4: Verify Integration

In Claude Desktop, you should see Marie AI tools available in the MCP tools menu (hammer icon).

## Available Tools

### Document Processing

#### extract_document_ocr

Extracts text and layout information from documents using OCR.

**Parameters:**
- `file_path` (string, required): Path to the document file
- `ref_id` (string, required): Unique document reference ID
- `ref_type` (string, optional): Document type/category (default: "document")
- `enable_page_classifier` (boolean, optional): Enable page classification (default: false)
- `enable_page_splitter` (boolean, optional): Enable page splitting (default: false)
- `enable_template_matching` (boolean, optional): Enable template matching (default: false)
- `sla_hours` (integer, optional): Hard SLA deadline in hours (default: 4)

**Returns:** Job ID for tracking

**Example in Claude:**
```
Extract text from the invoice at /path/to/invoice.pdf
with ref_id "invoice_001" and ref_type "invoice"
```

#### extract_document_data

Extracts structured data from documents using predefined templates.

**Parameters:**
- `file_path` (string, required): Path to the document file
- `template_id` (string, required): Template/planner ID for extraction
- `ref_id` (string, required): Unique document reference ID
- `ref_type` (string, optional): Document type/category (default: "document")
- `sla_hours` (integer, optional): Hard SLA deadline in hours (default: 4)

**Returns:** Job ID for tracking

**Example in Claude:**
```
Extract data from medical_form.pdf using template 117183
with ref_id "form_001" and ref_type "medical_form"
```

### Job Management

#### get_job_status

Retrieves detailed status information for a specific job.

**Parameters:**
- `job_id` (string, required): Job ID from submission

**Returns:** Job state, progress, and error messages

**Example in Claude:**
```
Check the status of job "job_abc123"
```

#### list_jobs

Lists all jobs, optionally filtered by state.

**Parameters:**
- `state` (string, optional): Filter by state ('created', 'pending', 'active', 'running', 'completed', 'failed', 'cancelled')

**Returns:** Array of job objects

**Example in Claude:**
```
Show me all completed jobs
```

#### stop_job

Stops a running job.

**Parameters:**
- `job_id` (string, required): Job ID to stop

**Example in Claude:**
```
Stop job "job_abc123"
```

#### delete_job

Deletes a job from the system.

**Parameters:**
- `job_id` (string, required): Job ID to delete

**Example in Claude:**
```
Delete job "job_abc123"
```

#### get_job_results

Downloads job results from S3 storage.

**Parameters:**
- `ref_id` (string, required): Document reference ID used during submission
- `ref_type` (string, required): Document type used during submission
- `output_path` (string, required): Local path for downloaded results

**Example in Claude:**
```
Download results for ref_id "invoice_001" and ref_type "invoice"
to ./results/invoice_001.json
```

### System Monitoring

#### get_deployments

Retrieves information about active executors and deployments.

**Example in Claude:**
```
Show me all active deployments
```

#### get_capacity

Retrieves slot capacity and resource utilization.

**Example in Claude:**
```
What's the current system capacity?
```

#### get_debug_info

Retrieves scheduler debug information and metrics.

**Example in Claude:**
```
Get scheduler debug information
```

#### health_check

Performs a health check on the Marie gateway.

**Example in Claude:**
```
Check Marie system health
```

## Usage Examples

### Example 1: OCR Document Extraction

**User Prompt:**
```
Extract text from the invoice at /Users/me/Documents/invoice.pdf
with ref_id "invoice_12345" and ref_type "invoice"
```

**Claude Response:**
```json
{
  "status": "success",
  "job_id": "job_abc123",
  "s3_path": "s3://marie/invoice/invoice_12345/invoice.pdf",
  "ref_id": "invoice_12345",
  "ref_type": "invoice",
  "message": "OCR extraction job submitted. Use get_job_status() to track progress."
}
```

**Check Status:**
```
Check the status of job "job_abc123"
```

**Download Results:**
```
Get results for ref_id "invoice_12345" and ref_type "invoice"
saved to ./results/invoice_12345.json
```

### Example 2: Template-Based Data Extraction

**User Prompt:**
```
Extract data from medical_form.pdf using template 117183
```

**Claude Response:**
```json
{
  "status": "success",
  "job_id": "job_xyz789",
  "template_id": "117183",
  "message": "Data extraction job submitted."
}
```

### Example 3: System Monitoring

**User Prompt:**
```
What's the current system status and capacity?
```

**Claude performs multiple tool calls:**
1. `health_check()` - Checks gateway health
2. `get_capacity()` - Gets available slots
3. `get_deployments()` - Lists active executors

**Claude summarizes:**
```
The Marie AI system is healthy and operational:
- Gateway Status: OK
- Available Slots: 8/10 (80% capacity)
- Active Executors: extractor, classifier, ner_extractor
- Processing: 2 jobs currently running
```

## Output Directories

Processing results are saved to your local machine:

```
~/.marie-mcp/outputs/
├── document_extraction/
│   └── invoice_12345_20250112_143052.json
└── information_extraction/
    └── form_001_20250112_144523.json
```

Each result file contains:
- Extracted text/data
- Metadata (timestamps, job info)
- Processing details

## Workflow

### Standard Processing Flow

```mermaid
graph LR
    A[User Request] --> B[Upload to S3]
    B --> C[Submit Job]
    C --> D[Job Runs Async]
    D --> E[Check Status]
    E --> F[Download Results]
```

1. **Upload**: Document uploaded to S3
2. **Submit**: Job submitted with metadata (ref_id, ref_type, template, SLA)
3. **Process**: Job runs asynchronously through Marie's DAG executor
4. **Monitor**: Track progress via job_id
5. **Retrieve**: Download results from S3 when complete

### Job States

- `created` - Job created but not yet queued
- `pending` - Job queued for execution
- `active` - Job picked up by scheduler
- `running` - Job currently processing
- `completed` - Job finished successfully
- `failed` - Job encountered an error
- `cancelled` - Job was stopped by user

## Troubleshooting

### Issue: "MARIE_API_KEY environment variable is required"

**Solution:**
Ensure your `.env` file exists and contains:
```bash
MARIE_API_KEY=your-api-key-here
```

Or set the environment variable:
```bash
export MARIE_API_KEY=your-api-key-here
```

### Issue: "Failed to upload to S3"

**Possible Causes:**
- Incorrect AWS credentials
- S3 bucket doesn't exist
- No write permissions

**Solution:**
1. Verify AWS credentials:
   ```bash
   aws s3 ls s3://marie/
   ```

2. Check bucket permissions:
   ```bash
   aws s3api get-bucket-acl --bucket marie
   ```

3. Test upload manually:
   ```bash
   aws s3 cp test.txt s3://marie/test/
   ```

### Issue: Claude Desktop Doesn't Show Tools

**Solution:**
1. Verify config file is valid JSON:
   ```bash
   python -m json.tool ~/Library/Application\ Support/Claude/claude_desktop_config.json
   ```

2. Check Claude Desktop logs:
   - macOS: `~/Library/Logs/Claude/`
   - Windows: `%APPDATA%\Claude\logs\`

3. Restart Claude Desktop completely (quit and reopen)

4. Test MCP server directly:
   ```bash
   marie-mcp
   ```

### Issue: "Connection refused" to Marie Gateway

**Solution:**
1. Verify Marie gateway is running:
   ```bash
   curl http://localhost:5000/check?text=health
   ```

2. Check MARIE_BASE_URL in config:
   ```bash
   echo $MARIE_BASE_URL
   ```

3. Verify no firewall blocking port 5000

4. Check Marie gateway logs for errors

### Issue: Job Stuck in "pending" State

**Solution:**
1. Check system capacity:
   ```
   In Claude: "What's the current capacity?"
   ```

2. Get scheduler debug info:
   ```
   In Claude: "Get scheduler debug information"
   ```

3. Verify executors are running:
   ```
   In Claude: "Show me all deployments"
   ```

4. Check Marie gateway logs for executor issues

### Issue: Results Not Found in S3

**Solution:**
1. Wait for job to complete:
   ```
   In Claude: "Check status of job xyz"
   ```

2. Verify job completed successfully (state: "completed")

3. Check S3 path:
   ```bash
   aws s3 ls s3://marie/your-ref-type/your-ref-id/
   ```

4. Look for `.meta.json` file containing results

## Advanced Usage

### Integration with LangChain

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
            "MARIE_API_KEY": "your-api-key",
            "AWS_ACCESS_KEY_ID": "your-aws-key",
            "AWS_SECRET_ACCESS_KEY": "your-aws-secret"
        }
    )
)

# Create agent
llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")
agent = create_react_agent(llm, marie_tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=marie_tools)

# Use agent
result = await agent_executor.ainvoke({
    "input": "Extract text from invoice.pdf"
})
```

### Integration with OpenAI Agents

```python
from agents import Agent
from agents.mcp import MCPServerStdio

# Create Marie MCP connection
marie_server = MCPServerStdio(
    command="uvx",
    args=["marie-mcp"],
    env={
        "MARIE_BASE_URL": "http://localhost:5000",
        "MARIE_API_KEY": "your-api-key"
    }
)

# Create agent
agent = Agent(
    name="DocumentProcessor",
    instructions="You are a document processing assistant...",
    tools=marie_server.get_tools(),
    model="gpt-4"
)

# Use agent
result = agent.run("Extract data from form.pdf using template 117183")
```

## Best Practices

### 1. Use Meaningful Reference IDs

```python
# Good
ref_id = "invoice_2024_Q1_company_ABC_12345"

# Avoid
ref_id = "doc1"
```

### 2. Set Appropriate SLAs

```python
# High priority documents
sla_hours = 1

# Standard processing
sla_hours = 4

# Batch processing
sla_hours = 24
```

### 3. Monitor Job Progress

Don't assume instant processing. Always:
1. Submit job and store job_id
2. Check status periodically
3. Handle failed/cancelled states
4. Download results when completed

### 4. Handle Errors Gracefully

```python
# In your application
try:
    result = submit_ocr_job(file_path)
    job_id = result['job_id']

    # Poll for completion
    while True:
        status = check_job_status(job_id)
        if status['state'] == 'completed':
            break
        elif status['state'] == 'failed':
            handle_failure(status['error'])
            break
        time.sleep(5)

    results = download_results(ref_id, ref_type)
except Exception as e:
    logger.error(f"Processing failed: {e}")
```

### 5. Clean Up Resources

```python
# Delete completed jobs periodically
completed_jobs = list_jobs(state='completed')
for job in completed_jobs:
    if job['completed_at'] < one_week_ago:
        delete_job(job['job_id'])
```

## API Reference

For complete API documentation, see:
- [Marie MCP Package Documentation](https://github.com/marieai/marie-ai/tree/main/packages/marie-mcp)
- [MCP Protocol Specification](https://modelcontextprotocol.io/)
- [Marie AI API Documentation](/docs/api)

## Support

Need help?

- 📖 **Documentation**: [docs.marieai.co](https://docs.marieai.co)
- 💬 **GitHub Discussions**: [github.com/marieai/marie-ai/discussions](https://github.com/marieai/marie-ai/discussions)
- 🐛 **Issues**: [github.com/marieai/marie-ai/issues](https://github.com/marieai/marie-ai/issues)
- 📧 **Email**: support@marieai.co

## Next Steps

- Learn about [Document Extraction](/docs/extract)
- Explore [Template Matching](/docs/guides/template_matching)
- Understand [Service Architecture](/docs/guides/architecture-overview)
- Check [API Reference](/docs/api)
