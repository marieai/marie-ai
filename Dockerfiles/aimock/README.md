# AIMock - Comprehensive AI Stack Mocking

A complete mock server for the entire AI stack, built on [AIMock by CopilotKit](https://github.com/CopilotKit/aimock).

## What It Mocks

| Component | Providers/Protocols |
|-----------|---------------------|
| **LLMs** | OpenAI, Claude, Gemini, Bedrock, Azure, Vertex AI, Ollama, Cohere (13 total) |
| **MCP** | Full JSON-RPC 2.0 protocol (tools, resources, prompts) |
| **A2A** | Agent-to-agent protocol with SSE streaming |
| **Vector DBs** | Pinecone, Qdrant, ChromaDB |
| **Services** | Web search (Tavily), reranking (Cohere), moderation |

## Quick Start

### Option 1: Fixture-Based Mocking (Simple, No Code)

```bash
# Start AIMock server with JSON fixtures
docker compose -f docker-compose.mock-llm.yml up -d

# Verify it's running
curl http://localhost:4010/health
```

### Option 2: Programmatic Mocking (Custom Logic)

```bash
# Start AIMock server with custom handlers
docker compose -f docker-compose.mock-llm-programmatic.yml up -d

# Or run directly
cd aimock/programmatic
npm install
npm start
```

## Endpoints

| Endpoint | Description |
|----------|-------------|
| `http://localhost:4010/v1` | OpenAI-compatible API |
| `http://localhost:4010/anthropic` | Anthropic/Claude API |
| `http://localhost:4010/gemini` | Google Gemini API |
| `http://localhost:4010/mcp` | MCP protocol |
| `http://localhost:4010/vector` | Vector DB APIs |
| `http://localhost:4010/metrics` | Prometheus metrics |

## Usage Examples

### OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:4010/v1", api_key="mock-key")

response = client.chat.completions.create(
    model="gpt-4o", messages=[{"role": "user", "content": "extract invoice data"}]
)
print(response.choices[0].message.content)
```

### With LiteLLM

```python
import litellm

# Point to AIMock
litellm.api_base = "http://localhost:4010/v1"
litellm.api_key = "mock-key"

response = litellm.completion(
    model="openai/gpt-4o",
    messages=[{"role": "user", "content": "classify this document"}],
)
```

### Anthropic SDK

```python
from anthropic import Anthropic

client = Anthropic(base_url="http://localhost:4010/anthropic", api_key="mock-key")

message = client.messages.create(
    model="claude-3-opus-20240229",
    messages=[{"role": "user", "content": "summarize this document"}],
)
```

### Streaming

```python
response = client.chat.completions.create(
    model="gpt-4o", messages=[{"role": "user", "content": "stream test"}], stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

## Configuration

### Main Config: `aimock/config/aimock.json`

```json
{
  "llm": {
    "fixtures": "/fixtures/llm",
    "providers": ["openai", "claude", "gemini"],
    "streaming": {
      "ttft": 50,        // Time to first token (ms)
      "tps": 50,         // Tokens per second
      "jitter": 10       // Randomness (ms)
    }
  },
  "mcp": {
    "tools": "/fixtures/mcp/tools.json",
    "resources": "/fixtures/mcp/resources.json"
  },
  "chaos": {
    "enabled": false,
    "errorRate": 0.0,    // Probability of 500 errors
    "malformedRate": 0.0 // Probability of malformed JSON
  }
}
```

## Fixture Format

### LLM Fixtures (`fixtures/llm/*.json`)

```json
[
  {
    "match": {
      "userMessage": "keyword to match"
    },
    "response": {
      "content": "Response text here",
      "model": "gpt-4o",
      "finishReason": "stop"
    },
    "opts": {
      "chunkSize": 10,    // For streaming
      "latency": 100      // Response delay (ms)
    }
  }
]
```

### Tool Call Response

```json
{
  "match": {
    "userMessage": "search documents"
  },
  "response": {
    "toolCalls": [{
      "id": "call_001",
      "type": "function",
      "function": {
        "name": "search_documents",
        "arguments": "{\"query\": \"invoice\", \"limit\": 10}"
      }
    }],
    "finishReason": "tool_calls"
  }
}
```

## Included Fixtures

### Document Processing (`document-processing.json`)
- `extract` - Invoice/document extraction
- `classify` - Document classification
- `summarize` - Document summarization
- `analyze` - Entity analysis
- `ocr` - OCR text extraction

### RAG Queries (`rag-queries.json`)
- `query` - Knowledge base Q&A
- `search` - Document search
- `find` - Document lookup
- `compare` - Document comparison

### General (`general.json`)
- `hello` - Greeting response
- `help` - Help documentation
- `test` - Test confirmation
- `json` - JSON response
- `error` - Error message
- `stream` - Streaming test

## Programmatic Custom Handlers

For dynamic responses based on request content, use the programmatic approach:

### LLM Message Handlers

```typescript
import { LLMock } from "@copilotkit/aimock";

const mock = new LLMock({ port: 4010 });

// Pattern-based handler with dynamic response
mock.onMessage(/extract.*invoice/i, async (message: string) => {
  const hasLineItems = message.toLowerCase().includes("line item");
  return {
    content: JSON.stringify({
      document_type: "invoice",
      extracted_fields: {
        invoice_number: "INV-" + Math.floor(Math.random() * 10000),
        ...(hasLineItems && { line_items: [...] }),
      },
      confidence: 0.95,
    }),
  };
});

// Simple static response
mock.onMessage(/classify/i, {
  content: JSON.stringify({ classification: "invoice", confidence: 0.97 }),
});

// Tool call response
mock.onMessage(/search.*documents/i, {
  content: null,
  tool_calls: [{
    id: "call_" + Date.now(),
    type: "function",
    function: {
      name: "search_documents",
      arguments: JSON.stringify({ query: "...", limit: 10 }),
    },
  }],
});

await mock.start();
```

### MCP Tool Handlers

```typescript
import { MCPMock } from "@copilotkit/aimock";

const mcp = new MCPMock({ port: 4011 });

// Dynamic tool execution
mcp.onToolCall("extract_document", async (args) => {
  const documentId = args.document_id || "unknown";
  return {
    content: [{
      type: "text",
      text: JSON.stringify({
        document_id: documentId,
        extracted_fields: { ... },
        confidence: 0.94,
      }),
    }],
  };
});

// Resource handler
mcp.onResourceRead("documents://recent", async () => {
  return {
    contents: [{
      uri: "documents://recent",
      mimeType: "application/json",
      text: JSON.stringify([...recentDocs]),
    }],
  };
});

await mcp.start();
```

### Vector Database Handlers

```typescript
import { VectorMock } from "@copilotkit/aimock";

const vector = new VectorMock({ port: 4012 });

// Custom semantic search
vector.onQuery("documents", async (query) => {
  const results = searchDocuments(query.vector, query.topK);
  return {
    matches: results.map(doc => ({
      id: doc.id,
      score: doc.relevance,
      metadata: { title: doc.title },
    })),
  };
});

// Stateful upsert
const store = new Map();
vector.onUpsert("documents", async (vectors) => {
  vectors.forEach(v => store.set(v.id, v));
  return { upsertedCount: vectors.length };
});

await vector.start();
```

### Running Programmatic Server

```bash
# Build and run with Docker
docker compose -f docker-compose.mock-llm-programmatic.yml up -d

# Or run directly for development
cd aimock/programmatic
npm install
npm run dev  # Watch mode
```

## Advanced Features

### Record & Replay

Capture real API responses and save as fixtures:

```bash
# Start in record mode
docker compose -f docker-compose.mock-llm.yml up -d
docker exec marie-aimock aimock --record --provider-openai https://api.openai.com

# Captured responses saved to /fixtures
```

### Chaos Testing

Enable failure injection:

```json
{
  "chaos": {
    "enabled": true,
    "errorRate": 0.1,      // 10% of requests return 500
    "malformedRate": 0.05, // 5% return malformed JSON
    "disconnectRate": 0.02 // 2% disconnect mid-stream
  }
}
```

### Prometheus Metrics

```bash
# Get metrics
curl http://localhost:4010/metrics

# Example metrics:
# aimock_requests_total{provider="openai"} 150
# aimock_latency_seconds{quantile="0.99"} 0.045
# aimock_fixture_matches_total{fixture="extract"} 42
```

## Integration with Marie Batch Queue

For testing the batch queue system:

```python
from marie.engine.batch_processor import BatchProcessor

# Configure to use AIMock
processor = BatchProcessor(api_base="http://aimock:4010/v1", api_key="mock")

# Deterministic responses for testing
results = await processor.process_batch(
    [
        {"content": "extract invoice data"},
        {"content": "classify this document"},
    ]
)
```

## CI/CD Integration

### GitHub Actions

```yaml
services:
  aimock:
    image: ghcr.io/copilotkit/aimock:latest
    ports:
      - 4010:4010
    volumes:
      - ./fixtures:/fixtures

steps:
  - name: Run tests
    env:
      OPENAI_BASE_URL: http://localhost:4010/v1
      OPENAI_API_KEY: mock
    run: pytest tests/
```

### GitLab CI

```yaml
services:
  - name: ghcr.io/copilotkit/aimock:latest
    alias: aimock

test:
  variables:
    OPENAI_BASE_URL: http://aimock:4010/v1
  script:
    - pytest tests/
```

## Directory Structure

```
aimock/
├── config/
│   └── aimock.json          # Main configuration
├── fixtures/                # Fixture-based mocking
│   ├── llm/
│   │   ├── document-processing.json
│   │   ├── rag-queries.json
│   │   └── general.json
│   ├── mcp/
│   │   ├── tools.json
│   │   ├── resources.json
│   │   └── prompts.json
│   ├── vector/
│   │   └── collections.json
│   └── a2a/
│       └── agents.json
└── programmatic/            # Programmatic mocking
    ├── package.json
    ├── server.ts            # Main server with LLM handlers
    ├── mcp-handlers.ts      # MCP tool/resource handlers
    ├── vector-handlers.ts   # Vector DB handlers
    └── Dockerfile
```

## Choosing Between Approaches

| Feature | Fixture-Based | Programmatic |
|---------|--------------|--------------|
| **Setup** | Just JSON files | Requires Node.js |
| **Responses** | Static, pattern-matched | Dynamic, computed |
| **State** | Stateless | Can be stateful |
| **Use Case** | CI/CD, simple tests | Complex workflows |
| **Maintenance** | Edit JSON | Edit TypeScript |

## Troubleshooting

**Server not starting:**
```bash
docker compose -f docker-compose.mock-llm.yml logs aimock
```

**Fixture not matching:**
- Check that input contains the match keyword (case-insensitive)
- First matching fixture wins
- Restart container after changing fixtures

**Streaming not working:**
- Ensure `stream=True` in request
- Check `chunkSize` in fixture opts

## Resources

- [AIMock Documentation](https://aimock.copilotkit.dev/)
- [GitHub Repository](https://github.com/CopilotKit/aimock)
- [CopilotKit](https://www.copilotkit.ai/)
