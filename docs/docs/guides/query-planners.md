---
sidebar_position: 5
---

# Query Planners

Query Planners are the foundation of Marie-AI's DAG-based execution system. They define workflows as directed acyclic graphs (DAGs) of tasks that can include conditional branching, parallel execution, and human-in-the-loop (HITL) interactions.

## Overview

Query planners transform high-level processing requirements into executable DAGs that the scheduler can orchestrate. They support:

- **Linear workflows**: Sequential task execution
- **Parallel execution**: Concurrent task processing
- **Conditional branching**: Dynamic routing based on runtime conditions
- **HITL workflows**: Human approval and correction steps
- **Subgraph composition**: Nested workflow patterns

## Planner Types

Marie-AI supports three types of query planners:

### 1. Code-Based Planners (Python)

Traditional Python functions decorated with `@register_query_plan`:

```python
from marie.query_planner import (
    Query,
    QueryPlan,
    QueryType,
    PlannerInfo,
    register_query_plan,
    NoopQueryDefinition,
    ExecutorEndpointQueryDefinition
)

@register_query_plan("invoice_processor")
def invoice_processing_planner(planner_info: PlannerInfo, **kwargs) -> QueryPlan:
    """Process invoice documents with extraction and validation"""
    base_id = planner_info.base_id

    nodes = [
        Query(
            task_id=f"{base_id}-0000",
            query_str="START: Initialize invoice processing",
            dependencies=[],
            node_type=QueryType.COMPUTE,
            definition=NoopQueryDefinition()
        ),
        Query(
            task_id=f"{base_id}-0001",
            query_str="Extract invoice data",
            dependencies=[f"{base_id}-0000"],
            node_type=QueryType.COMPUTE,
            definition=ExecutorEndpointQueryDefinition(
                method="EXECUTOR_ENDPOINT",
                endpoint="/extract",
                params={"layout": "invoice"}
            )
        ),
        Query(
            task_id=f"{base_id}-0002",
            query_str="END: Complete processing",
            dependencies=[f"{base_id}-0001"],
            node_type=QueryType.COMPUTE,
            definition=NoopQueryDefinition()
        )
    ]

    return QueryPlan(nodes=nodes)
```

### 2. Wheel-Based Planners

Planners distributed as Python wheel files for modular deployment:

```yaml
# config/service/marie.yml
query_planners:
  watch_wheels: true
  wheel_directories:
    - /mnt/data/marie-ai/config/wheels
  planners:
    - name: custom_planner
      py_module: custom_package.query.planner
```

### 3. JSON-Based Planners (New!)

**Planners defined as JSON structures** that can be registered at runtime without Python code. This enables visual workflow builders like Marie Studio to publish templates directly to the gateway.

```json
{
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
      "query_str": "Extract document data",
      "dependencies": ["01930d8c-0000-7000-8000-000000000000"],
      "node_type": "COMPUTE",
      "definition": {
        "method": "EXECUTOR_ENDPOINT",
        "endpoint": "/extract",
        "params": {"layout": "invoice"}
      }
    }
  ]
}
```

## JSON Planner Registration API

### Python API

```python
from marie.query_planner import QueryPlanRegistry

# Set storage path for persistence (optional)
QueryPlanRegistry.set_storage_path("/path/to/planners")

# Register a planner from JSON
success = QueryPlanRegistry.register_from_json(
    name="invoice_processor",
    plan_definition=plan_json,
    description="Process invoice documents",
    version="1.0.0",
    tags=["invoice", "extraction"],
    category="document_processing"
)

# List all planners with metadata
planners = QueryPlanRegistry.list_planners_with_metadata()

# Get specific planner metadata
metadata = QueryPlanRegistry.get_metadata("invoice_processor")

# Unregister a planner
QueryPlanRegistry.unregister("invoice_processor")

# Load planners from storage on startup
QueryPlanRegistry.load_json_planners_from_storage()
```

### HTTP API

The gateway exposes RESTful endpoints for planner management:

#### List All Planners

```bash
GET /api/planners
```

Response:
```json
{
  "planners": [
    {
      "name": "invoice_processor",
      "description": "Process invoice documents",
      "version": "1.0.0",
      "tags": ["invoice", "extraction"],
      "category": "document_processing",
      "source_type": "json",
      "plan_definition": { ... },
      "created_at": "2025-11-20T10:00:00",
      "updated_at": "2025-11-20T10:00:00"
    }
  ],
  "total": 1
}
```

#### Get Specific Planner

```bash
GET /api/planners/{name}
```

Response (200 OK):
```json
{
  "name": "invoice_processor",
  "description": "Process invoice documents",
  "version": "1.0.0",
  "tags": ["invoice", "extraction"],
  "source_type": "json",
  "plan_definition": {
    "nodes": [ ... ]
  }
}
```

#### Register New Planner

```bash
POST /api/planners
Content-Type: application/json

{
  "name": "invoice_processor",
  "plan": {
    "nodes": [ ... ]
  },
  "description": "Process invoice documents",
  "version": "1.0.0",
  "tags": ["invoice", "extraction"],
  "category": "document_processing"
}
```

Response (201 Created):
```json
{
  "success": true,
  "message": "Planner 'invoice_processor' registered successfully",
  "planner": { ... }
}
```

#### Unregister Planner

```bash
DELETE /api/planners/{name}
```

Response (200 OK):
```json
{
  "success": true,
  "message": "Planner 'invoice_processor' unregistered successfully"
}
```

## Node Types and Definitions

### Query Node Structure

Each node in a query plan has:

- `task_id`: Unique UUID7 identifier
- `query_str`: Human-readable description
- `dependencies`: List of task IDs this node depends on
- `node_type`: Type of node (COMPUTE, BRANCH, MERGER, etc.)
- `definition`: Execution definition specific to the method

### Node Types

#### COMPUTE

Standard computation nodes that execute tasks:

```json
{
  "node_type": "COMPUTE",
  "definition": {
    "method": "EXECUTOR_ENDPOINT",
    "endpoint": "/extract",
    "params": {"layout": "invoice"}
  }
}
```

#### BRANCH

Conditional branching based on JSONPath or Python conditions:

```json
{
  "node_type": "BRANCH",
  "definition": {
    "method": "BRANCH",
    "conditions": [
      {
        "path": "$.classification.category",
        "operator": "equals",
        "value": "invoice",
        "target_branch": "invoice_path"
      }
    ]
  }
}
```

#### MERGER

Merges results from parallel branches:

```json
{
  "node_type": "MERGER",
  "definition": {
    "method": "MERGER_ENHANCED",
    "strategy": "wait_all_active",
    "merge_function": "default"
  }
}
```

### Definition Methods

#### NOOP

No-operation node for workflow structure:

```json
{
  "method": "NOOP",
  "endpoint": "noop",
  "params": {}
}
```

#### EXECUTOR_ENDPOINT

Call an executor endpoint:

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

#### LLM

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

#### HITL_APPROVAL

Human approval step:

```json
{
  "method": "HITL_APPROVAL",
  "endpoint": "approval",
  "params": {
    "title": "Review Invoice Data",
    "description": "Please verify extracted invoice information",
    "approval_type": "binary",
    "timeout_minutes": 60,
    "notification": {
      "enabled": true,
      "recipients": ["reviewer@example.com"]
    }
  }
}
```

#### HITL_CORRECTION

Human correction step:

```json
{
  "method": "HITL_CORRECTION",
  "endpoint": "correction",
  "params": {
    "title": "Correct Invoice Fields",
    "fields": [
      {"name": "invoice_number", "type": "text", "required": true},
      {"name": "total_amount", "type": "number", "required": true}
    ],
    "timeout_minutes": 120
  }
}
```

## Configuration

### Gateway Configuration

Enable JSON planner storage in your gateway configuration:

```yaml
# config/service/marie.yml
jtype: Flow
version: '1'
protocol: http

query_planners:
  # Watch for wheel installations
  watch_wheels: true
  wheel_directories:
    - /mnt/data/marie-ai/config/wheels

  # Storage path for JSON-based planners
  storage_path: /mnt/data/marie-ai/config/planners

  # Code-based planners
  planners:
    - name: existing_planner
      py_module: grapnel_g5.query.planner
```

### Initialize Storage on Startup

```python
from marie.query_planner import QueryPlanRegistry

# Set storage path
QueryPlanRegistry.set_storage_path("/mnt/data/marie-ai/config/planners")

# Load existing JSON planners
result = QueryPlanRegistry.load_json_planners_from_storage()
print(f"Loaded {result['loaded']} planners, {result['failed']} failed")
```

## Integration with Marie Studio

Marie Studio can use the HTTP API to manage query plan templates:

```typescript
// Publish a template from Marie Studio
const publishTemplate = async (template: QueryPlanTemplate) => {
  const response = await fetch(`${gatewayUrl}/api/planners`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${token}`
    },
    body: JSON.stringify({
      name: template.name,
      plan: template.planDefinition,
      description: template.description,
      version: template.version,
      tags: template.tags,
      category: template.category
    })
  });

  return response.json();
};

// List registered planners
const listPlanners = async () => {
  const response = await fetch(`${gatewayUrl}/api/planners`);
  const data = await response.json();
  return data.planners;
};

// Import a registered plan as a template
const importPlan = async (planName: string) => {
  const response = await fetch(`${gatewayUrl}/api/planners/${planName}`);
  const planner = await response.json();

  // Create local template from registered plan
  return {
    name: `${planner.name} (imported)`,
    description: planner.description,
    planDefinition: planner.plan_definition,
    sourceType: 'imported',
    sourceGatewayId: gatewayId
  };
};
```

## Execution Flow

1. **Planner Selection**: Client specifies planner name in execution request
2. **Plan Generation**: Registry retrieves planner function and executes it
3. **DAG Creation**: QueryPlan is converted to executable DAG
4. **Job Scheduling**: Scheduler assigns jobs based on dependencies and SLAs
5. **Task Execution**: Executors process tasks according to definitions
6. **Result Aggregation**: Results are collected and returned to client

## Advanced Features

### Conditional Branching

Create dynamic workflows that adapt based on intermediate results:

```python
from marie.query_planner.branching import (
    BranchQueryDefinition,
    BranchPath,
    BranchCondition
)

branch_node = Query(
    task_id="branch-001",
    query_str="Route based on document type",
    node_type=QueryType.BRANCH,
    definition=BranchQueryDefinition(
        method="BRANCH",
        endpoint="branch",
        paths=[
            BranchPath(
                name="invoice_path",
                conditions=[
                    BranchCondition(
                        path="$.classification.category",
                        operator="equals",
                        value="invoice"
                    )
                ]
            ),
            BranchPath(
                name="receipt_path",
                conditions=[
                    BranchCondition(
                        path="$.classification.category",
                        operator="equals",
                        value="receipt"
                    )
                ]
            )
        ]
    )
)
```

### Parallel Subgraphs

Execute multiple processing paths concurrently:

```python
# Create parallel extraction branches
extract_branch_1 = Query(...)  # Extract tables
extract_branch_2 = Query(...)  # Extract text
extract_branch_3 = Query(...)  # Extract images

# Merge results
merger = Query(
    task_id="merger-001",
    query_str="Merge all extraction results",
    dependencies=[
        extract_branch_1.task_id,
        extract_branch_2.task_id,
        extract_branch_3.task_id
    ],
    node_type=QueryType.MERGER,
    definition=EnhancedMergerQueryDefinition(
        method="MERGER_ENHANCED",
        strategy=MergerStrategy.WAIT_ALL_ACTIVE
    )
)
```

### HITL Workflows

Integrate human decision points for approval, correction, and routing:

```python
from marie.query_planner.hitl import (
    HitlApprovalQueryDefinition,
    HitlCorrectionQueryDefinition,
    HitlRouterQueryDefinition
)

# Human approval node
approval = Query(
    task_id="approval-001",
    query_str="Human review of extraction results",
    node_type=QueryType.COMPUTE,
    definition=HitlApprovalQueryDefinition(
        method="HITL_APPROVAL",
        endpoint="hitl/approval",
        title="Review Extracted Data",
        description="Please verify the extracted invoice information",
        approval_type="binary",
        priority="high",
        auto_approve={
            "enabled": True,
            "confidence_threshold": 0.95
        },
        timeout={
            "enabled": True,
            "duration_seconds": 86400,
            "strategy": "use_default"
        }
    )
)

# Human correction node
correction = Query(
    task_id="correction-001",
    query_str="Correct extraction errors",
    node_type=QueryType.COMPUTE,
    definition=HitlCorrectionQueryDefinition(
        method="HITL_CORRECTION",
        endpoint="hitl/correction",
        title="Correct Invoice Fields",
        correction_type="structured",
        fields=[
            {
                "key": "invoice_number",
                "label": "Invoice Number",
                "type": "text",
                "required": True
            },
            {
                "key": "total_amount",
                "label": "Total Amount",
                "type": "number",
                "required": True
            }
        ],
        auto_validate={
            "enabled": True,
            "confidence_threshold": 0.9
        }
    )
)

# Confidence-based router (non-blocking)
router = Query(
    task_id="router-001",
    query_str="Route based on AI confidence",
    node_type=QueryType.COMPUTE,
    definition=HitlRouterQueryDefinition(
        method="HITL_ROUTER",
        endpoint="hitl/router",
        auto_approve_threshold=0.95,
        human_review_threshold=0.7,
        below_threshold_action="review"
    )
)
```

For detailed information on HITL capabilities, human-driven branching, and comparisons with other systems, see the [HITL Guide](./hitl.md).

## Best Practices

### 1. Plan Naming

Use descriptive, unique names:
- ✅ `invoice_extraction_with_validation`
- ✅ `medical_records_hipaa_compliant`
- ❌ `plan1`
- ❌ `test`

### 2. Version Management

Follow semantic versioning:
- `1.0.0`: Initial release
- `1.1.0`: New features (backward compatible)
- `2.0.0`: Breaking changes

### 3. Metadata

Always provide rich metadata:
```python
QueryPlanRegistry.register_from_json(
    name="invoice_processor",
    plan_definition=plan,
    description="Extract and validate invoice data with HITL review",
    version="1.2.0",
    tags=["invoice", "extraction", "validation", "hitl"],
    category="financial_documents"
)
```

### 4. Error Handling

Include error handling paths in complex workflows:
```json
{
  "branches": [
    {"name": "success_path", "conditions": [...]},
    {"name": "error_path", "conditions": [...]}
  ]
}
```

### 5. Testing

Test planners thoroughly before deployment:
```python
from marie.job.job_manager import generate_job_id

# Test plan generation
planner_info = PlannerInfo(name="test", base_id=generate_job_id())
plan = planner_func(planner_info)

# Validate structure
assert len(plan.nodes) > 0
assert all(node.task_id for node in plan.nodes)
```

## Monitoring and Debugging

### View Registered Planners

```bash
# List all planners
curl http://gateway:8080/api/planners

# Get specific planner details
curl http://gateway:8080/api/planners/invoice_processor
```

### Plan Visualization

Export plan to YAML for inspection:

```python
from marie.query_planner.planner import plan_to_yaml

plan = planner_func(planner_info)
yaml_str = plan_to_yaml(plan, "debug_plan.yaml")
```

### Execution Tracking

Link executions back to planners using metadata:

```python
metadata = QueryPlanRegistry.get_metadata("invoice_processor")
if metadata:
    print(f"Plan: {metadata.name}")
    print(f"Version: {metadata.version}")
    print(f"Created: {metadata.created_at}")
```

## Troubleshooting

### Planner Not Found

**Problem**: `KeyError: "Query planner 'xyz' is not registered!"`

**Solution**: Verify planner is registered:
```python
planners = QueryPlanRegistry.list_planners()
print("Available planners:", planners)
```

### Invalid Plan Definition

**Problem**: `ValidationError` when registering JSON planner

**Solution**: Validate plan structure:
```python
from marie.query_planner import QueryPlan

try:
    plan = QueryPlan(**plan_json)
except ValidationError as e:
    print("Validation errors:", e.errors())
```

### Planner Not Persisting

**Problem**: JSON planners not saved to disk

**Solution**: Set storage path before registration:
```python
QueryPlanRegistry.set_storage_path("/path/to/storage")
```

### Duplicate Planner Error

**Problem**: `ValueError: Planner 'xyz' is already registered`

**Solution**: Unregister existing planner first:
```python
QueryPlanRegistry.unregister("xyz")
QueryPlanRegistry.register_from_json(...)
```

## See Also

- [Human-in-the-Loop (HITL)](./hitl.md) - Complete HITL guide with approval, correction, routing, and human-driven branching
- [Document Pipelines](./pipelines.md) - Pipeline configuration
- [Service Discovery](./service-discovery.md) - Multi-gateway setup
- [Architecture Overview](./architecture-overview.md) - System design
- [API Reference](../api/cli.md) - CLI commands
