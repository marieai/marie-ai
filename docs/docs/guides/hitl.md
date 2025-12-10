---
sidebar_position: 6
---

# Human-in-the-Loop (HITL)

Marie-AI's Human-in-the-Loop (HITL) system enables workflows to pause execution and request human input for approval, data correction, and decision-making. This guide covers HITL capabilities, implementation patterns, and comparisons with other systems.

## Overview

HITL executors integrate with Marie Studio's frontend to provide a complete human-in-the-loop workflow experience. The system supports:

- **Quality Assurance**: Human verification of AI predictions
- **Data Correction**: Human correction of extracted data with field-level validation
- **Confidence-Based Routing**: Automatic routing based on AI confidence scores
- **Approval Workflows**: Multi-user approval with role-based assignment
- **Branching Decisions**: Human-driven workflow path selection

## Architecture

```
┌─────────────────┐
│  Marie Studio   │
│   (Frontend)    │◄──────┐
└─────────────────┘       │
         │                │
         │ tRPC API       │ Polling
         ▼                │
┌─────────────────┐       │
│  Marie Studio   │       │
│    (Backend)    │       │
└─────────────────┘       │
         │                │
         │ PostgreSQL     │
         ▼                │
┌─────────────────┐       │
│   Database      │       │
│  (HITL Tables)  │◄──────┤
└─────────────────┘       │
         ▲                │
         │                │
         │ Database Ops   │
         │                │
┌─────────────────┐       │
│  Marie-AI       │       │
│   (Backend)     │───────┘
│                 │
│  HITL Executors │
└─────────────────┘
```

## HITL Executors

### 1. HitlApprovalExecutor

**Endpoint**: `/hitl/approval`

Pauses workflow and requests human approval or decision-making.

**Features**:
- Binary approval (approve/reject)
- Single/multi/ranked choice selection
- Auto-approval based on confidence threshold
- Timeout handling with configurable strategies
- Multi-user approval requirements

**Example**:

```python
from marie.query_planner import Query, QueryType
from marie.query_planner.hitl import HitlApprovalQueryDefinition

approval_node = Query(
    task_id="hitl_approval_1",
    query_str="HITL Approval - Verify Classification",
    dependencies=["classify_1"],
    node_type=QueryType.COMPUTE,
    definition=HitlApprovalQueryDefinition(
        endpoint="hitl/approval",
        title="Verify Invoice Classification",
        description="Review the AI classification result",
        approval_type="binary",  # or "single_choice", "multi_choice", "ranked_choice"
        priority="high",
        auto_approve={
            "enabled": True,
            "confidence_threshold": 0.95
        },
        timeout={
            "enabled": True,
            "duration_seconds": 86400,  # 24 hours
            "strategy": "use_default",
        },
        assigned_to={
            "user_ids": [],
            "roles": ["finance_manager", "accountant"]
        },
        params={
            "context_data": {
                "classification": "invoice",
                "confidence": 0.87
            }
        }
    ),
)
```

**Auto-Approval Logic**:
- If `confidence >= auto_approve.confidence_threshold`, automatically approves
- Otherwise, creates database request and waits for human response

**Timeout Strategies**:
- `use_default`: Use the default option specified in config
- `fail`: Raise TimeoutError and fail the workflow
- `escalate`: Escalate to different users/roles (planned)
- `skip_downstream`: Skip downstream tasks and continue

### 2. HitlCorrectionExecutor

**Endpoint**: `/hitl/correction`

Pauses workflow and requests human correction of extracted data.

**Features**:
- Structured data correction with field-level validation
- Auto-validation for high-confidence fields
- Side-by-side comparison UI
- Field exemptions (always require review)
- Multiple field types: text, number, date, enum, boolean, json

**Example**:

```python
from marie.query_planner.hitl import HitlCorrectionQueryDefinition

correction_node = Query(
    task_id="hitl_correction_1",
    query_str="HITL Correction - Review Invoice Data",
    dependencies=["extract_1"],
    node_type=QueryType.COMPUTE,
    definition=HitlCorrectionQueryDefinition(
        endpoint="hitl/correction",
        title="Review Extracted Invoice Data",
        description="Verify and correct extracted fields",
        correction_type="structured",
        priority="high",
        fields=[
            {
                "key": "invoice_number",
                "label": "Invoice Number",
                "type": "text",
                "required": True,
                "confidence_threshold": 0.9,
            },
            {
                "key": "total_amount",
                "label": "Total Amount",
                "type": "number",
                "required": True,
                "confidence_threshold": 0.95,
            },
        ],
        auto_validate={
            "enabled": True,
            "confidence_threshold": 0.9,
            "exempt_fields": ["total_amount"],  # Always require human review
        },
        params={
            "original_data": {
                "invoice_number": "INV-123456",
                "total_amount": 1234.56
            },
            "field_confidences": {
                "invoice_number": 0.92,
                "total_amount": 0.96
            }
        }
    ),
)
```

**Auto-Validation Logic**:
- Fields with `confidence >= auto_validate.confidence_threshold` are auto-validated
- Fields in `exempt_fields` always require human review
- If all non-exempt fields auto-validate, skips human review entirely

### 3. HitlRouterExecutor

**Endpoint**: `/hitl/router`

Routes workflow execution based on confidence without pausing.

**Features**:
- Three-way routing: auto_approve, human_review, reject
- Conditional routing rules (e.g., always review if amount > $10k)
- Non-blocking (doesn't create database requests)

**Example**:

```python
from marie.query_planner.hitl import HitlRouterQueryDefinition

router_node = Query(
    task_id="hitl_router_1",
    query_str="HITL Router - Confidence-Based Routing",
    dependencies=["classify_1"],
    node_type=QueryType.COMPUTE,
    definition=HitlRouterQueryDefinition(
        endpoint="hitl/router",
        auto_approve_threshold=0.95,
        human_review_threshold=0.7,
        below_threshold_action="review",  # or "reject"
        always_review_if=[
            {"field": "total_amount", "condition": "gt", "value": 10000},
            {"field": "vendor_type", "condition": "equals", "value": "new"},
        ],
        params={
            "confidence": 0.88,
            "data": {
                "total_amount": 15000,
                "vendor_type": "existing"
            }
        }
    ),
)
```

**Routing Logic**:
1. Check conditional rules first (`always_review_if`)
2. If any rule matches, route to `human_review`
3. Otherwise, route based on confidence:
   - `confidence >= auto_approve_threshold` → `auto_approve`
   - `confidence >= human_review_threshold` → `human_review`
   - `confidence < human_review_threshold` → `review` or `reject`

## Human-Driven Branching

Marie-AI supports human-driven workflow branching similar to Apache Airflow's `HITLBranchOperator`, with enhanced capabilities.

### Option 1: HITL Approval + JSONPath Branch (Recommended)

Combine `HitlApprovalExecutor` with `BranchQueryDefinition` for declarative human-driven branching:

```python
from marie.query_planner.branching import (
    BranchQueryDefinition,
    BranchPath,
    BranchCondition,
    BranchEvaluationMode
)

# Step 1: Human makes a decision via HITL Approval
hitl_approval = Query(
    task_id="hitl_branch_decision",
    query_str="Human Decision: Select Processing Path",
    dependencies=["extract_1"],
    node_type=QueryType.COMPUTE,
    definition=HitlApprovalQueryDefinition(
        endpoint="hitl/approval",
        title="Select Processing Path",
        description="Choose how to process this document",
        approval_type="single_choice",
        priority="high",
        options=[
            {"value": "standard", "label": "Standard Processing"},
            {"value": "expedited", "label": "Expedited Processing"},
            {"value": "manual", "label": "Manual Review"},
        ],
    ),
)

# Step 2: Branch based on human decision
branch = Query(
    task_id="branch_on_decision",
    query_str="BRANCH: Route based on human decision",
    dependencies=["hitl_branch_decision"],
    node_type=QueryType.COMPUTE,
    definition=BranchQueryDefinition(
        endpoint="branch",
        paths=[
            BranchPath(
                path_id="standard_path",
                condition=BranchCondition(
                    jsonpath="$.tags.hitl_decision",
                    operator="==",
                    value="standard"
                ),
                target_node_ids=["standard_process_1"],
            ),
            BranchPath(
                path_id="expedited_path",
                condition=BranchCondition(
                    jsonpath="$.tags.hitl_decision",
                    operator="==",
                    value="expedited"
                ),
                target_node_ids=["expedited_process_1"],
            ),
            BranchPath(
                path_id="manual_path",
                condition=BranchCondition(
                    jsonpath="$.tags.hitl_decision",
                    operator="==",
                    value="manual"
                ),
                target_node_ids=["manual_review_1"],
            ),
        ],
        evaluation_mode=BranchEvaluationMode.FIRST_MATCH,
    ),
)

# Step 3: Define downstream nodes for each path
standard_process = Query(
    task_id="standard_process_1",
    query_str="Standard Processing",
    dependencies=["branch_on_decision"],
    node_type=QueryType.COMPUTE,
    definition=ExecutorEndpointQueryDefinition(endpoint="process/standard")
)

expedited_process = Query(
    task_id="expedited_process_1",
    query_str="Expedited Processing",
    dependencies=["branch_on_decision"],
    node_type=QueryType.COMPUTE,
    definition=ExecutorEndpointQueryDefinition(endpoint="process/expedited")
)
```

### Option 2: SWITCH Statement (Simpler Syntax)

For value-based routing, use `SwitchQueryDefinition`:

```python
from marie.query_planner.branching import SwitchQueryDefinition

# HITL Approval for decision
hitl_approval = Query(...)  # Same as above

# SWITCH on human decision
switch = Query(
    task_id="switch_on_decision",
    query_str="SWITCH: Route based on human choice",
    dependencies=["hitl_branch_decision"],
    node_type=QueryType.COMPUTE,
    definition=SwitchQueryDefinition(
        endpoint="switch",
        switch_field="$.tags.hitl_decision",
        cases={
            "standard": ["standard_process_1"],
            "expedited": ["expedited_process_1"],
            "manual": ["manual_review_1"],
        },
        default_case=["default_process_1"],
    ),
)
```

### Option 3: Multi-Choice Selection (Advanced)

Allow humans to activate **multiple branches** simultaneously:

```python
# Multi-select approval
hitl_multi_select = Query(
    task_id="hitl_multi_branch",
    query_str="Select Processing Steps to Execute",
    dependencies=["classify_1"],
    node_type=QueryType.COMPUTE,
    definition=HitlApprovalQueryDefinition(
        endpoint="hitl/approval",
        title="Select Processing Steps",
        description="Choose which processing steps to execute (select multiple)",
        approval_type="multi_choice",  # Multiple selections allowed
        priority="high",
        options=[
            {"value": "ocr", "label": "Run OCR"},
            {"value": "ner", "label": "Extract Entities"},
            {"value": "classify", "label": "Classify Document"},
            {"value": "translate", "label": "Translate Text"},
        ],
    ),
)

# Branch with ALL_MATCH mode (activates all matching paths)
branch_multi = Query(
    task_id="branch_multi",
    query_str="BRANCH: Activate selected paths",
    dependencies=["hitl_multi_branch"],
    node_type=QueryType.COMPUTE,
    definition=BranchQueryDefinition(
        endpoint="branch",
        evaluation_mode=BranchEvaluationMode.ALL_MATCH,
        paths=[
            BranchPath(
                path_id="ocr_path",
                condition=BranchCondition(
                    jsonpath="$.tags.hitl_decisions[?(@=='ocr')]",
                    operator="exists"
                ),
                target_node_ids=["ocr_1"],
            ),
            BranchPath(
                path_id="ner_path",
                condition=BranchCondition(
                    jsonpath="$.tags.hitl_decisions[?(@=='ner')]",
                    operator="exists"
                ),
                target_node_ids=["ner_1"],
            ),
        ],
    ),
)
```

## Complete HITL Workflow Example

This example demonstrates a complete invoice processing workflow combining routing, correction, and approval:

```python
from marie.query_planner import (
    ConditionalQueryPlanBuilder,
    Query,
    QueryType,
    PlannerInfo,
    register_query_plan,
)
from marie.query_planner.hitl import (
    HitlApprovalQueryDefinition,
    HitlCorrectionQueryDefinition,
    HitlRouterQueryDefinition,
)

@register_query_plan("invoice_processing_hitl")
def invoice_processing_workflow(planner_info: PlannerInfo, **kwargs):
    """
    Complete HITL workflow: Extract -> Route -> Correct -> Approve -> Process

    High confidence: Auto-process
    Medium confidence: Human correction + approval
    Low confidence: Reject
    """
    nodes = []

    # 1. Extract invoice data
    extract = Query(
        task_id=f"{planner_info.base_id}-0001",
        query_str="Extract invoice data",
        dependencies=[],
        node_type=QueryType.COMPUTE,
        definition=ExecutorEndpointQueryDefinition(
            endpoint="extract/invoice",
            params={"model": "invoice_extractor_v2"}
        ),
    )
    nodes.append(extract)

    # 2. Route based on confidence
    router = Query(
        task_id=f"{planner_info.base_id}-0002",
        query_str="Route based on confidence",
        dependencies=[extract.task_id],
        node_type=QueryType.COMPUTE,
        definition=HitlRouterQueryDefinition(
            endpoint="hitl/router",
            auto_approve_threshold=0.95,
            human_review_threshold=0.7,
            always_review_if=[
                {"field": "total_amount", "condition": "gt", "value": 10000}
            ],
        ),
    )
    nodes.append(router)

    # 3a. High confidence path: Auto-process
    auto_process = Query(
        task_id=f"{planner_info.base_id}-0003",
        query_str="Auto-process high confidence invoices",
        dependencies=[router.task_id],
        node_type=QueryType.COMPUTE,
        definition=ExecutorEndpointQueryDefinition(endpoint="process/invoice"),
    )
    nodes.append(auto_process)

    # 3b. Medium confidence path: Human correction
    correction = Query(
        task_id=f"{planner_info.base_id}-0004",
        query_str="Human correction of extracted data",
        dependencies=[router.task_id],
        node_type=QueryType.COMPUTE,
        definition=HitlCorrectionQueryDefinition(
            endpoint="hitl/correction",
            title="Review Invoice Data",
            correction_type="structured",
            fields=[
                {
                    "key": "invoice_number",
                    "label": "Invoice Number",
                    "type": "text",
                    "required": True,
                },
                {
                    "key": "total_amount",
                    "label": "Total Amount",
                    "type": "number",
                    "required": True,
                },
            ],
            auto_validate={
                "enabled": True,
                "confidence_threshold": 0.9
            },
        ),
    )
    nodes.append(correction)

    # 3c. Approval after correction
    approval = Query(
        task_id=f"{planner_info.base_id}-0005",
        query_str="Human approval after correction",
        dependencies=[correction.task_id],
        node_type=QueryType.COMPUTE,
        definition=HitlApprovalQueryDefinition(
            endpoint="hitl/approval",
            title="Approve Corrected Invoice",
            approval_type="binary",
            priority="high",
        ),
    )
    nodes.append(approval)

    # 3d. Process reviewed invoices
    process_reviewed = Query(
        task_id=f"{planner_info.base_id}-0006",
        query_str="Process reviewed invoices",
        dependencies=[approval.task_id],
        node_type=QueryType.COMPUTE,
        definition=ExecutorEndpointQueryDefinition(endpoint="process/invoice"),
    )
    nodes.append(process_reviewed)

    # 3e. Low confidence path: Reject
    reject = Query(
        task_id=f"{planner_info.base_id}-0007",
        query_str="Reject low confidence invoices",
        dependencies=[router.task_id],
        node_type=QueryType.COMPUTE,
        definition=NoopQueryDefinition(),
    )
    nodes.append(reject)

    return QueryPlan(nodes=nodes)
```

## Configuration

### Database Configuration

HITL executors connect to the Marie Studio PostgreSQL database:

```bash
# Option 1: Use DATABASE_URL (preferred)
export DATABASE_URL="postgresql://user:password@host:port/database"

# Option 2: Individual environment variables
export MARIE_STUDIO_DB_HOST="localhost"
export MARIE_STUDIO_DB_PORT="5432"
export MARIE_STUDIO_DB_NAME="marie_studio"
export MARIE_STUDIO_DB_USER="postgres"
export MARIE_STUDIO_DB_PASSWORD="your_password"
```

### Executor Configuration

```python
from marie.executor.hitl import HitlApprovalExecutor

executor = HitlApprovalExecutor(
    db_config={
        "host": "localhost",
        "port": 5432,
        "database": "marie_studio",
        "user": "postgres",
        "password": "your_password",
    },
    poll_interval=5,  # Seconds between polling for responses
    max_poll_attempts=None,  # Max polls before timeout (None = unlimited)
)
```

## Comparison with Apache Airflow HITL

Marie-AI's HITL system is more advanced than Apache Airflow 3.1's HITL operators:

| Feature | Airflow 3.1 HITL | Marie-AI HITL |
|---------|------------------|---------------|
| **Human approval** | Yes (ApprovalOperator) | Yes (HitlApprovalExecutor) |
| **Binary approval** | Yes | Yes |
| **Single choice** | Yes (HITLOperator) | Yes (`approval_type="single_choice"`) |
| **Multiple choice** | Yes (HITLOperator) | Yes (`approval_type="multi_choice"`) |
| **Ranked choice** | No | Yes (`approval_type="ranked_choice"`) |
| **Auto-approval by confidence** | No | Yes (confidence thresholds) |
| **Structured data correction** | No | Yes (field-level validation) |
| **Field-level confidence** | No | Yes |
| **Auto-validation** | No | Yes (skip human review for high confidence) |
| **Confidence-based routing** | No | Yes (HitlRouterExecutor) |
| **Conditional routing rules** | No | Yes (always_review_if) |
| **Timeout strategies** | Basic | 5 strategies (use_default, fail, escalate, skip_downstream, use_original) |
| **Priority levels** | Not specified | 4 levels (low, medium, high, critical) |
| **Role-based assignment** | Limited | Yes (user_ids + roles) |
| **Multi-user approval** | Yes | Yes (require_all_approval) |
| **Human-driven branching** | Yes (HITLBranchOperator) | Yes (HitlApproval + BranchQueryDefinition) |
| **Multi-path activation** | No | Yes (ALL_MATCH evaluation mode) |
| **JSONPath branching** | No | Yes |
| **Response time tracking** | Not specified | Yes |
| **Side-by-side comparison UI** | No | Yes |
| **Field exemptions** | No | Yes (always require review for specific fields) |

### Key Advantages

**Marie-AI's HITL excels at**:
- AI/ML workflows with confidence-based decision-making
- Document intelligence with structured data correction
- High-throughput workflows with auto-approval/validation
- Complex branching with JSONPath conditions
- Field-level data validation and correction

**Airflow's HITL excels at**:
- General workflow approvals
- Organizations already using Airflow
- Direct UI link generation
- Mature ecosystem

## Database Schema

HITL executors use these tables in the `marie_scheduler` schema:

### hitl_requests

```sql
CREATE TABLE marie_scheduler.hitl_requests (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dag_id TEXT NOT NULL,
    job_id TEXT NOT NULL,
    request_type TEXT NOT NULL,  -- 'approval', 'correction', 'router'
    title TEXT NOT NULL,
    description TEXT,
    priority TEXT NOT NULL,
    context_data JSONB,
    config JSONB,
    status TEXT NOT NULL,
    timeout_at TIMESTAMP,
    assigned_user_ids TEXT[],
    assigned_roles TEXT[],
    require_all_approval BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

### hitl_responses

```sql
CREATE TABLE marie_scheduler.hitl_responses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    request_id UUID NOT NULL REFERENCES marie_scheduler.hitl_requests(id),
    responded_by TEXT NOT NULL,
    decision TEXT,
    corrected_data JSONB,
    feedback TEXT,
    response_time_seconds INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);
```

## Testing

### Mock Workflows

See `tests/integration/scheduler/mock_query_plans.py` for example HITL workflows:

- `mock_hitl_approval`: Simple approval workflow
- `mock_hitl_correction`: Data correction workflow
- `mock_hitl_router`: Confidence-based routing workflow
- `mock_hitl_complete_workflow`: Complex workflow combining all HITL nodes

### Running Tests

```bash
# Run HITL executor tests
pytest tests/integration/scheduler/ -k hitl -v
```

## Troubleshooting

### Database Connection Errors

**Issue**: `psycopg2.OperationalError: could not connect to server`

**Solution**:
1. Verify database is running: `psql -U postgres -d marie_studio`
2. Check environment variables are set correctly
3. Verify PostgreSQL is listening on the correct host/port

### Auto-Approval Not Working

**Issue**: Requests not auto-approving despite high confidence

**Solution**:
1. Verify `auto_approve.enabled` is `True`
2. Check confidence value: `context_data.confidence >= auto_approve.confidence_threshold`
3. Ensure confidence is passed in `params.context_data.confidence`

### Missing Responses

**Issue**: Executor polling forever, no response received

**Solution**:
1. Check frontend is running and accessible
2. Verify user can see request in `/hitl` inbox
3. Check database for responses:
   ```sql
   SELECT * FROM marie_scheduler.hitl_responses
   WHERE request_id = '<request_id>';
   ```
4. Verify `poll_interval` is reasonable (default: 5s)

## Future Enhancements

- WebSocket/SSE support for real-time notifications (eliminate polling)
- Email/Slack/Teams notification integration
- Multi-user approval with quorum logic
- Escalation workflows (auto-escalate after reminders)
- Analytics dashboard (approval rates, response times)
- Active learning from corrections (improve models based on feedback)
- Bulk approval operations
- SLA tracking and alerts

## See Also

- [Query Planners](./query-planners.md) - Overview of Marie-AI's query planning system
- [Branching and Conditional Execution](./query-planners.md#branching) - Detailed branching documentation
- [Executor README](../../marie/executor/hitl/README.md) - Technical implementation details
