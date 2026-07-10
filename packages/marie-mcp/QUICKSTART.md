# Marie MCP Server - Quick Start Guide

Get up and running with Marie MCP Server in 5 minutes!

## Prerequisites

- Python 3.12
- Running Marie AI gateway
- AWS S3 credentials
- Marie AI API key

## Installation

```bash
# Install via uv
uv add marie-mcp

# Or install from source
git clone https://github.com/marieai/marie-mcp-server.git
cd marie-mcp-server
uv sync
```

## Configuration

Create a `.env` file:

```bash
# Copy template
cp .env.example .env

# Edit with your credentials
nano .env
```

Required settings:
```bash
MARIE_BASE_URL=http://localhost:5000
MARIE_API_KEY=your-api-key-here
AWS_ACCESS_KEY_ID=your-aws-access-key
AWS_SECRET_ACCESS_KEY=your-aws-secret-key
```

## Test the Server

Run the server directly to verify configuration:

```bash
# This will start the MCP server via STDIO
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

Registering MCP tools...
✓ All tools registered successfully

Starting MCP server...
Ready to accept connections via STDIO
============================================================
```

Press Ctrl+C to stop.

## Use with Claude Desktop

1. **Locate Claude Desktop config**:
   - macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - Windows: `%APPDATA%\Claude\claude_desktop_config.json`
   - Linux: `~/.config/Claude/claude_desktop_config.json`

2. **Add Marie MCP server**:

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

3. **Restart Claude Desktop**

4. **Verify tools are available**:
   - Click the MCP icon (tools icon) in Claude
   - You should see Marie AI tools listed

## Your First Document Processing

In Claude Desktop, try:

```
Extract text from the invoice at /path/to/invoice.pdf
with ref_id "invoice_001" and ref_type "invoice"
```

Claude will:
1. Call `extract_document_ocr` tool
2. Upload document to S3
3. Submit OCR extraction job
4. Return job_id

Then ask:
```
Check the status of job_id "job_xxxxx"
```

When complete, retrieve results:
```
Get the results for ref_id "invoice_001" and ref_type "invoice"
and save to ./results/invoice_001.json
```

## Example Workflows

### OCR Extraction
```
User: Extract text from receipt.pdf (ref_id: "receipt_123", ref_type: "receipt")
Claude: → extract_document_ocr(...)
        ← {"job_id": "job_abc123"}

User: Check job status
Claude: → get_job_status("job_abc123")
        ← {"state": "completed"}

User: Get the results
Claude: → get_job_results(ref_id="receipt_123", ref_type="receipt", ...)
        ← Results downloaded to ./results/receipt_123.json
```

### Template-Based Extraction
```
User: Extract data from medical_form.pdf using template 117183
Claude: → extract_document_data(
          file_path="medical_form.pdf",
          template_id="117183",
          ref_id="form_001",
          ref_type="medical_form"
        )
        ← {"job_id": "job_xyz789"}
```

### System Monitoring
```
User: What's the system status?
Claude: → health_check()
        → get_capacity()
        → get_deployments()
        ← Shows health, capacity, and active executors
```

## Troubleshooting

### "MARIE_API_KEY environment variable is required"
- Make sure `.env` file exists in current directory
- Or set environment variable: `export MARIE_API_KEY=your-key`

### "Failed to upload to S3"
- Check AWS credentials are correct
- Verify S3 bucket exists and you have write access
- Test with: `aws s3 ls s3://marie/`

### "Connection refused"
- Verify Marie gateway is running: `curl http://localhost:5000/check?text=health`
- Check MARIE_BASE_URL is correct
- Ensure no firewall blocking connection

### Claude Desktop doesn't show tools
- Check Claude Desktop logs for errors
- Verify config JSON is valid: `python -m json.tool claude_desktop_config.json`
- Restart Claude Desktop completely
- Try running `marie-mcp` directly to test

## Next Steps

- Read full [README.md](README.md) for complete documentation
- Check [examples/](examples/) for integration examples
- See [CONTRIBUTING.md](CONTRIBUTING.md) to contribute
- Read [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) for architecture details

## Support

- GitHub Issues: https://github.com/marieai/marie-mcp-server/issues
- Documentation: https://docs.marieai.co/mcp-server
- Email: support@marieai.co

## Common Commands Reference

```bash
# Install
uv add marie-mcp

# Run server
marie-mcp

# Run with custom env file
ENV_FILE=/path/to/.env marie-mcp

# Check version
python -c "import marie_mcp; print(marie_mcp.__version__)"

# Run tests
pytest tests/

# Format code
black src/
isort src/
```

Happy processing! 🚀
