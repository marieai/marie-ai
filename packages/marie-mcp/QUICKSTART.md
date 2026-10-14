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
MARIE_BASE_URL=http://localhost:51000
MARIE_API_KEY=your-api-key-here
AWS_ACCESS_KEY_ID=your-aws-access-key
AWS_SECRET_ACCESS_KEY=your-aws-secret-key
MARIE_FABRIC_GROUP_ID=your-fabric-group  # Required for runtime diagnostics
```

## Test the Server

### Run the automated checks

```bash
cd ~/dev/marieai/marie-ai
.venv/bin/python -m pytest packages/marie-mcp/tests -q
```

Use the repository's `.venv` for local development. The suite includes real STDIO
initialization, tool discovery, MCP error responses, and client cleanup. Gateway
responses and S3 transfers are tested locally without production credentials or
cloud access. A passing suite does not verify a live deployment.

To run only the STDIO smoke test:

```bash
.venv/bin/python -m pytest \
  packages/marie-mcp/tests/test_server.py::test_stdio_initialize_list_and_error_call -q
```

### Connect with MCP Inspector

[MCP Inspector](https://github.com/modelcontextprotocol/inspector) lets you discover
and call tools interactively. Its current version requires Node.js 22.19 or newer.
The command below starts the source checkout using the repository's `.venv`.

```bash
cd ~/dev/marieai/marie-ai/packages/marie-mcp

read -rsp "Marie API key: " MARIE_API_KEY
printf '\n'

npx @modelcontextprotocol/inspector \
  -e "PYTHONPATH=$PWD/src" \
  -e "MARIE_API_KEY=$MARIE_API_KEY" \
  -e "MARIE_BASE_URL=http://localhost:51000" \
  ~/dev/marieai/marie-ai/.venv/bin/python \
  -m marie.mcp.server.server
```

Enter a valid gateway API key and change `MARIE_BASE_URL` if your gateway runs
elsewhere. Pass server configuration explicitly with Inspector's
[`-e KEY=VALUE` option](https://github.com/modelcontextprotocol/inspector/blob/main/docs/mcp-server-configuration.md).
Do not assume variables exported in the launch shell reach the server subprocess.

Open the URL printed by Inspector, connect, and list tools. Expect nine tools:

- `extract_document_ocr`
- `extract_document_data`
- `get_job_status`
- `list_jobs`
- `get_job_results`
- `get_deployments`
- `get_capacity`
- `get_debug_info`
- `health_check`

`stop_job` and `delete_job` are unavailable because the gateway does not perform
those operations. For startup and discovery alone, you can enter
`smoke-test-key`; gateway calls require a valid API key.

Call `health_check`, `get_capacity`, and `get_deployments` to verify gateway
connectivity. For `get_debug_info`, supply an authorized `fabric_group_id` in the
tool form or add `-e "MARIE_FABRIC_GROUP_ID=$MARIE_FABRIC_GROUP_ID"` to the launch
command. The key must have runtime observability permission for that fabric.

Configuration and startup messages appear on stderr. STDOUT is reserved for MCP
protocol messages. Starting the server alone does not exercise tool calls.

### Verify extraction against a live gateway and S3

Configure AWS credentials and the same S3 bucket used by the gateway in your shell.
After entering the API key as shown above, restart Inspector with those values:

```bash
cd ~/dev/marieai/marie-ai/packages/marie-mcp

npx @modelcontextprotocol/inspector \
  -e "PYTHONPATH=$PWD/src" \
  -e "MARIE_API_KEY=$MARIE_API_KEY" \
  -e "MARIE_BASE_URL=${MARIE_BASE_URL:-http://localhost:51000}" \
  -e "AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID" \
  -e "AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY" \
  -e "AWS_REGION=${AWS_REGION:-us-east-1}" \
  -e "S3_BUCKET=${S3_BUCKET:-${MARIE_S3_BUCKET:-marie}}" \
  -e "MARIE_OUTPUT_DIR=$HOME/.marie-mcp/outputs" \
  ~/dev/marieai/marie-ai/.venv/bin/python \
  -m marie.mcp.server.server
```

This check uploads a document and creates a real processing job. Use a small test
PDF and a unique `ref_id` for each run.

1. Call `extract_document_ocr` with an absolute `file_path`, the unique `ref_id`,
   and `ref_type="invoice"`. Confirm the response contains a `job_id`.
2. Call `get_job_status` with that returned UUID until processing completes.
3. Call `get_job_results` with the same `ref_id` and `ref_type`, and
   `output_path="smoke-test/result.json"`.
4. Open `~/.marie-mcp/outputs/smoke-test/result.json` and check the extracted
   content against the test PDF. If you changed `MARIE_OUTPUT_DIR`, look there.

Relative result paths start inside `MARIE_OUTPUT_DIR`. Paths outside that directory
are rejected. A completed download replaces an existing file at its destination.

This verifies MCP → gateway → scheduler → executor → S3 → local result. If a
submission times out, reconcile existing jobs before submitting again; acceptance
may be unknown and submissions are not automatically retried.

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
        "MARIE_BASE_URL": "http://localhost:51000",
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

The server loaded successfully, but its process has no API key. An Inspector
traceback may wrap this `ValueError` inside an `ExceptionGroup`; the missing key
is the underlying startup failure.

- Restart Inspector with `-e "MARIE_API_KEY=$MARIE_API_KEY"`, as shown above, or
  set `MARIE_API_KEY` in Inspector's server environment settings before reconnecting.
- For direct execution, export `MARIE_API_KEY` or configure a `.env` file loaded
  by the server. Inspector's working directory can differ from your shell, so use
  explicit `-e` options for reproducible testing.
- A synthetic key only enables local startup and tool discovery. Use a valid key
  for gateway operations.

### "Failed to upload to S3"
- Check AWS credentials are correct
- Verify S3 bucket exists and you have write access
- Test with: `aws s3 ls s3://marie/`

### "Connection refused"

The local Marie service configuration exposes HTTP on port `51000` and gRPC on
`52000`. Set `MARIE_BASE_URL=http://localhost:51000` for this setup. The MCP
package's fallback is `http://localhost:5000`, so pass the HTTP URL explicitly.
A `ConnectError` occurs before an HTTP response and can leave gateway logs empty.

After changing the URL, restart the MCP subprocess through Inspector. Check the
`Gateway URL` line in the server's startup output, then retry `health_check` and
`list_jobs`.

- Verify Marie gateway is running: `curl http://localhost:51000/check?text=health`
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

# Check version
python -c "import marie.mcp.server; print(marie.mcp.server.__version__)"

# Run tests from the Marie AI repository root
.venv/bin/python -m pytest packages/marie-mcp/tests -q

# Format code
black src/
isort src/
```

Happy processing! 🚀
