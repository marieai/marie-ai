---
sidebar_position: 6
title: MCP Quick Start
description: Get started with Marie MCP Server in 5 minutes
---

# Marie MCP Server Quick Start

Get your Marie AI-powered document processing assistant running in Claude Desktop in just 5 minutes!

## What You'll Build

By the end of this guide, you'll have:
- ✅ Marie MCP Server installed and configured
- ✅ Claude Desktop with Marie AI tools
- ✅ Ability to extract text from documents
- ✅ Ability to extract structured data with templates

## Prerequisites

### 1. Running Marie AI Gateway

You need a Marie AI gateway running. If you don't have one:

```bash
# Start Marie AI gateway
marie server --start --uses config/service/marie-dev.yml
```

Verify it's running:
```bash
curl http://localhost:5000/check?text=health
# Should return: {"result": "ok"}
```

### 2. Get Your API Key

Contact your Marie AI administrator or check your Marie AI console for your API key.

### 3. AWS S3 Access

Marie MCP uploads documents to S3. You'll need:
- AWS Access Key ID
- AWS Secret Access Key
- S3 Bucket name (usually "marie")

## Step 1: Install Marie MCP

```bash
# Install using UV (recommended)
uv pip install marie-mcp

# Or using pip
pip install marie-mcp
```

Verify installation:
```bash
marie-mcp --help
```

## Step 2: Configure Environment

Create a `.env` file:

```bash
# Create .env file
cat > ~/.marie-mcp.env << EOF
MARIE_BASE_URL=http://localhost:5000
MARIE_API_KEY=your-api-key-here
AWS_ACCESS_KEY_ID=your-aws-access-key
AWS_SECRET_ACCESS_KEY=your-aws-secret-key
AWS_REGION=us-east-1
S3_BUCKET=marie
EOF
```

Test configuration:
```bash
# Source the env file
source ~/.marie-mcp.env

# Test connection
curl $MARIE_BASE_URL/check?text=health

# Test S3 access
aws s3 ls s3://$S3_BUCKET/
```

## Step 3: Configure Claude Desktop

### Find Your Config File

**macOS:**
```bash
open ~/Library/Application\ Support/Claude/
```

**Windows:**
```powershell
explorer %APPDATA%\Claude\
```

**Linux:**
```bash
nautilus ~/.config/Claude/
```

### Edit claude_desktop_config.json

Open `claude_desktop_config.json` and add:

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

:::tip
Replace `your-api-key-here` with your actual API key and AWS credentials!
:::

### Restart Claude Desktop

Completely quit Claude Desktop (Cmd+Q on macOS) and restart it.

## Step 4: Verify Installation

### Check Tools Menu

In Claude Desktop:
1. Click the **tools icon** (hammer) in the bottom toolbar
2. You should see **"marie-ai"** in the list
3. Expand it to see 11 available tools

### Test with a Simple Query

In Claude, type:

```
Check the Marie AI system health and capacity
```

Claude should:
1. Call `health_check()` tool
2. Call `get_capacity()` tool
3. Report back the system status

Expected response:
```
The Marie AI system is healthy and operational:
- Gateway Status: OK
- Available Capacity: X/Y slots (Z% available)
- Active Deployments: extractor, classifier, ner_extractor
```

## Step 5: Your First Document Processing

### OCR Text Extraction

1. **Prepare a document** (PDF, TIFF, JPEG, or PNG)
   ```bash
   # Example: Download a sample invoice
   curl -o ~/invoice.pdf https://example.com/sample-invoice.pdf
   ```

2. **In Claude, type:**
   ```
   Extract text from the invoice at /Users/yourname/invoice.pdf
   with ref_id "invoice_001" and ref_type "invoice"
   ```

3. **Claude will:**
   - Upload the document to S3
   - Submit an OCR extraction job
   - Return a job ID

4. **Check the status:**
   ```
   What's the status of job "job_abc123"?
   ```

5. **Download results when complete:**
   ```
   Download the results for ref_id "invoice_001"
   and ref_type "invoice" to ~/results/invoice_001.json
   ```

### Template-Based Data Extraction

If you have a template ID (e.g., for medical forms):

```
Extract data from /Users/yourname/medical_form.pdf
using template 117183 with ref_id "form_001"
and ref_type "medical_form"
```

## Common First-Time Issues

### Issue: "No tools showing in Claude"

**Fix:**
1. Verify `claude_desktop_config.json` is valid JSON:
   ```bash
   python -m json.tool ~/Library/Application\ Support/Claude/claude_desktop_config.json
   ```
2. Restart Claude completely (quit and reopen)
3. Check Claude logs for errors

### Issue: "Connection refused"

**Fix:**
1. Verify Marie gateway is running:
   ```bash
   curl http://localhost:5000/check?text=health
   ```
2. Check `MARIE_BASE_URL` in config
3. Ensure no firewall blocking port 5000

### Issue: "Upload failed"

**Fix:**
1. Test AWS credentials:
   ```bash
   aws s3 ls s3://marie/
   ```
2. Verify S3 bucket exists
3. Check IAM permissions for the bucket

## Next Steps

### Learn More Tools

Explore all 11 available tools:

**Document Processing:**
- `extract_document_ocr` - OCR text extraction
- `extract_document_data` - Template-based extraction

**Job Management:**
- `get_job_status` - Check job progress
- `list_jobs` - List all jobs
- `stop_job` - Cancel a job
- `delete_job` - Remove a job
- `get_job_results` - Download results

**System Monitoring:**
- `get_deployments` - View active executors
- `get_capacity` - Check system capacity
- `get_debug_info` - Get scheduler info
- `health_check` - System health

### Example Workflows

**Batch Processing:**
```
I have 10 invoices in ~/invoices/ folder.
Extract text from all of them using OCR.
Use ref_ids like "batch_001", "batch_002", etc.
and ref_type "invoice_batch".
```

**Data Extraction with Template:**
```
I have 5 medical forms. Extract structured data
using template 117183. Save results to ~/medical_data/
```

**Monitor System:**
```
Show me all active jobs, completed jobs in the last hour,
and current system capacity
```

### Read Full Documentation

- [MCP Server Guide](/docs/guides/mcp-server) - Complete documentation
- [API Reference](/docs/api) - Marie AI API details
- [Template Matching](/docs/guides/template_matching) - Create extraction templates

## Troubleshooting

Still having issues? Check:

1. **Marie Gateway Logs**: Look for errors in Marie gateway console
2. **Claude Desktop Logs**:
   - macOS: `~/Library/Logs/Claude/`
   - Windows: `%APPDATA%\Claude\logs\`
3. **MCP Server Logs**: Run `marie-mcp` directly to see output

## Get Help

- 💬 [GitHub Discussions](https://github.com/marieai/marie-ai/discussions)
- 🐛 [Report Issues](https://github.com/marieai/marie-ai/issues)
- 📧 [Email Support](mailto:support@marieai.co)
- 📖 [Full Documentation](https://docs.marieai.co)

## Summary

You've successfully:
- ✅ Installed Marie MCP Server
- ✅ Configured Claude Desktop
- ✅ Tested system connectivity
- ✅ Processed your first document

**Happy document processing!** 🚀

---

**Next:** [Read the complete MCP Server Guide →](/docs/guides/mcp-server)
