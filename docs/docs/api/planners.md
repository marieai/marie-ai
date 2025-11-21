---
sidebar_position: 5
---

# Query Planner API

REST API for managing query planners on Marie-AI gateways.

## Base URL

```
http://{gateway-host}:{port}/api/planners
```

All endpoints are under the `/api/planners` path and return JSON responses.

## Authentication

If your gateway has authentication enabled, include the authorization header:

```http
Authorization: Bearer {your-token}
```

## Endpoints

### List All Planners

Get a list of all registered query planners with their metadata.

```http
GET /api/planners
```

#### Response (200 OK)

```json
{
  "planners": [
    {
      "name": "invoice_processor",
      "description": "Process invoice documents with extraction and validation",
      "version": "1.0.0",
      "tags": ["invoice", "extraction", "validation"],
      "category": "document_processing",
      "source_type": "json",
      "source_module": null,
      "plan_definition": {
        "nodes": [
          {
            "task_id": "01930d8c-0000-7000-8000-000000000000",
            "query_str": "START: Initialize processing",
            "dependencies": [],
            "node_type": "COMPUTE",
            "definition": {
              "method": "NOOP",
              "endpoint": "noop",
              "params": {}
            }
          }
        ]
      },
      "created_at": "2025-11-20T10:00:00.000Z",
      "updated_at": "2025-11-20T10:00:00.000Z"
    },
    {
      "name": "receipt_processor",
      "description": null,
      "version": "1.0.0",
      "tags": [],
      "category": null,
      "source_type": "code",
      "source_module": "grapnel_g5.query.receipt",
      "plan_definition": null,
      "created_at": null,
      "updated_at": null
    }
  ],
  "total": 2
}
```

#### Source Types

- `json`: Registered via API or JSON file
- `code`: Registered via Python decorator
- `wheel`: Loaded from wheel package

#### Example

```bash
curl http://localhost:8080/api/planners
```

```python
import requests

response = requests.get("http://localhost:8080/api/planners")
data = response.json()

for planner in data["planners"]:
    print(f"{planner['name']} ({planner['source_type']}): {planner.get('description', 'N/A')}")
```

---

### Get Specific Planner

Retrieve metadata and plan definition for a specific query planner.

```http
GET /api/planners/{name}
```

#### Path Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | string | **Required.** Unique name of the planner |

#### Response (200 OK)

```json
{
  "name": "invoice_processor",
  "description": "Process invoice documents with extraction and validation",
  "version": "1.0.0",
  "tags": ["invoice", "extraction", "validation"],
  "category": "document_processing",
  "source_type": "json",
  "source_module": null,
  "plan_definition": {
    "nodes": [
      {
        "task_id": "01930d8c-0000-7000-8000-000000000000",
        "query_str": "START: Initialize processing",
        "dependencies": [],
        "node_type": "COMPUTE",
        "definition": {
          "method": "NOOP",
          "endpoint": "noop",
          "params": {}
        }
      },
      {
        "task_id": "01930d8c-0001-7000-8000-000000000000",
        "query_str": "Extract invoice data",
        "dependencies": ["01930d8c-0000-7000-8000-000000000000"],
        "node_type": "COMPUTE",
        "definition": {
          "method": "EXECUTOR_ENDPOINT",
          "endpoint": "/extract",
          "params": {"layout": "invoice"}
        }
      }
    ]
  },
  "created_at": "2025-11-20T10:00:00.000Z",
  "updated_at": "2025-11-20T10:00:00.000Z"
}
```

#### Response (404 Not Found)

```json
{
  "error": "Planner 'unknown_planner' not found"
}
```

#### Example

```bash
curl http://localhost:8080/api/planners/invoice_processor
```

```python
import requests

response = requests.get("http://localhost:8080/api/planners/invoice_processor")

if response.status_code == 200:
    planner = response.json()
    print(f"Planner: {planner['name']}")
    print(f"Nodes: {len(planner['plan_definition']['nodes'])}")
else:
    print(f"Error: {response.json()['error']}")
```

---

### Register New Planner

Register a new query planner from a JSON plan definition.

```http
POST /api/planners
```

#### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | **Yes** | Unique name for the planner |
| `plan` | object | **Yes** | QueryPlan JSON structure |
| `description` | string | No | Description of the planner |
| `version` | string | No | Version string (default: "1.0.0") |
| `tags` | array | No | List of tags for categorization |
| `category` | string | No | Category for organization |

#### Plan Structure

The `plan` object must contain a `nodes` array with QueryPlan structure:

```json
{
  "nodes": [
    {
      "task_id": "uuid-here",
      "query_str": "Task description",
      "dependencies": ["parent-task-id"],
      "node_type": "COMPUTE",
      "definition": {
        "method": "EXECUTOR_ENDPOINT",
        "endpoint": "/extract",
        "params": {}
      }
    }
  ]
}
```

#### Response (201 Created)

```json
{
  "success": true,
  "message": "Planner 'invoice_processor' registered successfully",
  "planner": {
    "name": "invoice_processor",
    "description": "Process invoice documents",
    "version": "1.0.0",
    "tags": ["invoice", "extraction"],
    "category": "document_processing",
    "source_type": "json",
    "plan_definition": { ... },
    "created_at": "2025-11-20T10:00:00.000Z",
    "updated_at": "2025-11-20T10:00:00.000Z"
  }
}
```

#### Response (400 Bad Request)

```json
{
  "success": false,
  "error": "Failed to register planner 'invoice_processor'"
}
```

Possible causes:
- Planner name already exists
- Invalid plan definition structure
- Validation errors in node definitions

#### Example

```bash
curl -X POST http://localhost:8080/api/planners \
  -H "Content-Type: application/json" \
  -d '{
    "name": "invoice_processor",
    "plan": {
      "nodes": [
        {
          "task_id": "01930d8c-0000-7000-8000-000000000000",
          "query_str": "START",
          "dependencies": [],
          "node_type": "COMPUTE",
          "definition": {
            "method": "NOOP",
            "endpoint": "noop",
            "params": {}
          }
        }
      ]
    },
    "description": "Process invoice documents",
    "version": "1.0.0",
    "tags": ["invoice", "extraction"],
    "category": "document_processing"
  }'
```

```python
import requests

plan_data = {
    "name": "invoice_processor",
    "plan": {
        "nodes": [
            {
                "task_id": "01930d8c-0000-7000-8000-000000000000",
                "query_str": "START: Initialize processing",
                "dependencies": [],
                "node_type": "COMPUTE",
                "definition": {
                    "method": "NOOP",
                    "endpoint": "noop",
                    "params": {}
                }
            },
            {
                "task_id": "01930d8c-0001-7000-8000-000000000000",
                "query_str": "Extract invoice data",
                "dependencies": ["01930d8c-0000-7000-8000-000000000000"],
                "node_type": "COMPUTE",
                "definition": {
                    "method": "EXECUTOR_ENDPOINT",
                    "endpoint": "/extract",
                    "params": {"layout": "invoice"}
                }
            }
        ]
    },
    "description": "Process invoice documents",
    "version": "1.0.0",
    "tags": ["invoice", "extraction"],
    "category": "document_processing"
}

response = requests.post(
    "http://localhost:8080/api/planners",
    json=plan_data
)

if response.status_code == 201:
    result = response.json()
    print(f"✓ {result['message']}")
else:
    print(f"✗ {response.json()['error']}")
```

---

### Unregister Planner

Remove a query planner from the registry.

```http
DELETE /api/planners/{name}
```

#### Path Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | string | **Required.** Name of the planner to unregister |

#### Response (200 OK)

```json
{
  "success": true,
  "message": "Planner 'invoice_processor' unregistered successfully"
}
```

#### Response (404 Not Found)

```json
{
  "success": false,
  "error": "Planner 'invoice_processor' not found"
}
```

#### Notes

- Only JSON-based planners can be unregistered via API
- Code-based planners require server restart to remove
- Unregistering a planner deletes its JSON file from storage
- This operation cannot be undone

#### Example

```bash
curl -X DELETE http://localhost:8080/api/planners/invoice_processor
```

```python
import requests

response = requests.delete("http://localhost:8080/api/planners/invoice_processor")

if response.status_code == 200:
    result = response.json()
    print(f"✓ {result['message']}")
else:
    print(f"✗ {response.json()['error']}")
```

---

## Node Definition Methods

### NOOP

No-operation node for workflow structure:

```json
{
  "method": "NOOP",
  "endpoint": "noop",
  "params": {}
}
```

### EXECUTOR_ENDPOINT

Execute an executor endpoint:

```json
{
  "method": "EXECUTOR_ENDPOINT",
  "endpoint": "/extract",
  "params": {
    "layout": "invoice",
    "confidence_threshold": 0.8
  }
}
```

### LLM

Call a language model:

```json
{
  "method": "LLM",
  "endpoint": "extract",
  "model_name": "gpt-4",
  "params": {
    "layout": "invoice",
    "temperature": 0.1
  }
}
```

### BRANCH

Conditional branching:

```json
{
  "method": "BRANCH",
  "endpoint": "branch",
  "paths": [
    {
      "name": "invoice_path",
      "conditions": [
        {
          "path": "$.classification.category",
          "operator": "equals",
          "value": "invoice"
        }
      ]
    }
  ]
}
```

### MERGER_ENHANCED

Merge parallel branches:

```json
{
  "method": "MERGER_ENHANCED",
  "endpoint": "merger",
  "strategy": "wait_all_active",
  "merge_function": "default"
}
```

### HITL_APPROVAL

Human approval step:

```json
{
  "method": "HITL_APPROVAL",
  "endpoint": "approval",
  "title": "Review Invoice Data",
  "description": "Please verify extracted information",
  "approval_type": "binary",
  "timeout_minutes": 60
}
```

### HITL_CORRECTION

Human correction step:

```json
{
  "method": "HITL_CORRECTION",
  "endpoint": "correction",
  "title": "Correct Invoice Fields",
  "fields": [
    {"name": "invoice_number", "type": "text", "required": true},
    {"name": "total_amount", "type": "number", "required": true}
  ],
  "timeout_minutes": 120
}
```

---

## Error Codes

| Status Code | Description |
|-------------|-------------|
| 200 | Success |
| 201 | Created (planner registered) |
| 400 | Bad Request (invalid plan structure) |
| 404 | Not Found (planner doesn't exist) |
| 500 | Internal Server Error |

---

## Rate Limiting

No rate limiting is currently enforced on planner endpoints.

---

## OpenAPI Specification

The planner endpoints are documented in the gateway's OpenAPI specification:

```bash
curl http://localhost:8080/openapi.json
```

Or view the interactive Swagger UI:

```
http://localhost:8080/docs
```

---

## Examples

### Complete Workflow

1. **List existing planners**
   ```bash
   curl http://localhost:8080/api/planners
   ```

2. **Register a new planner**
   ```bash
   curl -X POST http://localhost:8080/api/planners \
     -H "Content-Type: application/json" \
     -d @my_planner.json
   ```

3. **Verify registration**
   ```bash
   curl http://localhost:8080/api/planners/my_planner
   ```

4. **Execute the planner** (via job submission)
   ```bash
   curl -X POST http://localhost:8080/api/execute \
     -H "Content-Type: application/json" \
     -d '{
       "planner_name": "my_planner",
       "data": {...}
     }'
   ```

5. **Unregister when done**
   ```bash
   curl -X DELETE http://localhost:8080/api/planners/my_planner
   ```

### TypeScript Integration

```typescript
interface QueryPlanner {
  name: string;
  description?: string;
  version: string;
  tags: string[];
  category?: string;
  source_type: 'json' | 'code' | 'wheel';
  plan_definition?: {
    nodes: QueryNode[];
  };
}

interface QueryNode {
  task_id: string;
  query_str: string;
  dependencies: string[];
  node_type: 'COMPUTE' | 'BRANCH' | 'MERGER';
  definition: any;
}

class PlannerClient {
  constructor(private baseUrl: string, private token?: string) {}

  private getHeaders() {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };
    if (this.token) {
      headers['Authorization'] = `Bearer ${this.token}`;
    }
    return headers;
  }

  async listPlanners(): Promise<QueryPlanner[]> {
    const response = await fetch(`${this.baseUrl}/api/planners`, {
      headers: this.getHeaders(),
    });
    const data = await response.json();
    return data.planners;
  }

  async getPlanner(name: string): Promise<QueryPlanner> {
    const response = await fetch(`${this.baseUrl}/api/planners/${name}`, {
      headers: this.getHeaders(),
    });
    if (!response.ok) {
      throw new Error(`Planner not found: ${name}`);
    }
    return response.json();
  }

  async registerPlanner(planner: {
    name: string;
    plan: { nodes: QueryNode[] };
    description?: string;
    version?: string;
    tags?: string[];
    category?: string;
  }): Promise<QueryPlanner> {
    const response = await fetch(`${this.baseUrl}/api/planners`, {
      method: 'POST',
      headers: this.getHeaders(),
      body: JSON.stringify(planner),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Failed to register planner');
    }

    const result = await response.json();
    return result.planner;
  }

  async unregisterPlanner(name: string): Promise<void> {
    const response = await fetch(`${this.baseUrl}/api/planners/${name}`, {
      method: 'DELETE',
      headers: this.getHeaders(),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Failed to unregister planner');
    }
  }
}

// Usage
const client = new PlannerClient('http://localhost:8080', 'your-token');

// List planners
const planners = await client.listPlanners();
console.log('Available planners:', planners.map(p => p.name));

// Register planner
await client.registerPlanner({
  name: 'invoice_processor',
  plan: { nodes: [...] },
  description: 'Process invoice documents',
  version: '1.0.0',
  tags: ['invoice'],
});

// Get planner
const planner = await client.getPlanner('invoice_processor');
console.log('Planner nodes:', planner.plan_definition?.nodes.length);

// Unregister
await client.unregisterPlanner('invoice_processor');
```

---

## See Also

- [Query Planners Guide](../guides/query-planners.md) - Comprehensive guide
- [Document Pipelines](../guides/pipelines.md) - Pipeline configuration
- [CLI Reference](./cli.md) - Command-line interface
