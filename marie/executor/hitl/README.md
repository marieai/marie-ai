# HITL (Human-in-the-Loop) Executors

This directory contains Python executors for the Marie-AI HITL (Human-in-the-Loop) system that integrate with the Marie Studio frontend.

## Overview

HITL executors enable workflows to pause execution and wait for human input, allowing for:
- **Quality Assurance**: Human verification of AI predictions
- **Data Correction**: Human correction of extracted data
- **Routing**: Confidence-based routing with human review fallback

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

## Executors

### 1. HitlApprovalExecutor

**Endpoint**: `/hitl/approval`

**Purpose**: Pauses workflow and requests human approval/decision.

**Features**:
- Binary approval (approve/reject)
- Single/multi/ranked choice selection
- Auto-approval based on confidence threshold
- Timeout handling with configurable strategies

**Example Usage**:
```python
from marie.query_planner import HitlApprovalQueryDefinition, Query

approval_node = Query(
    task_id="hitl_approval_1",
    query_str="HITL Approval - Verify Classification",
    dependencies=["classify_1"],
    node_type=QueryType.COMPUTE,
    definition=HitlApprovalQueryDefinition(
        endpoint="hitl/approval",
        title="Verify Invoice Classification",
        description="Review the AI classification result",
        approval_type="binary",
        priority="high",
        auto_approve={"enabled": True, "confidence_threshold": 0.95},
        timeout={
            "enabled": True,
            "duration_seconds": 86400,  # 24 hours
            "strategy": "use_default",
        },
        assigned_to={"user_ids": [], "roles": ["finance_manager", "accountant"]},
        params={
            "dag_id": "dag_123",
            "job_id": "job_456",
            "context_data": {"classification": "invoice", "confidence": 0.87},
        },
    ),
)
```

**Auto-Approval Logic**:
- If `confidence >= auto_approve.confidence_threshold`, automatically approves
- Otherwise, creates database request and waits for human response

**Timeout Strategies**:
- `use_default`: Use the default option specified in config
- `fail`: Raise TimeoutError and fail the workflow
- `escalate`: Escalate to different users/roles (future)
- `skip_downstream`: Skip downstream tasks and continue

### 2. HitlCorrectionExecutor

**Endpoint**: `/hitl/correction`

**Purpose**: Pauses workflow and requests human correction of extracted data.

**Features**:
- Structured data correction with field-level validation
- Auto-validation for high-confidence fields
- Side-by-side comparison (frontend)
- Field exemptions (always require review)

**Example Usage**:
```python
from marie.query_planner import HitlCorrectionQueryDefinition, Query

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
            "dag_id": "dag_123",
            "job_id": "job_456",
            "original_data": {"invoice_number": "INV-123456", "total_amount": 1234.56},
            "field_confidences": {"invoice_number": 0.92, "total_amount": 0.96},
        },
    ),
)
```

**Auto-Validation Logic**:
- Fields with `confidence >= auto_validate.confidence_threshold` are auto-validated
- Fields in `exempt_fields` always require human review
- If all non-exempt fields auto-validate, skips human review entirely

**Timeout Strategies**:
- `use_original`: Use original extracted data
- `fail`: Raise TimeoutError and fail the workflow
- `skip_downstream`: Skip downstream tasks

### 3. HitlRouterExecutor

**Endpoint**: `/hitl/router`

**Purpose**: Routes workflow execution based on confidence without pausing.

**Features**:
- Three-way routing: auto_approve, human_review, reject
- Conditional routing rules (e.g., always review if amount > $10k)
- Non-blocking (doesn't create database requests)

**Example Usage**:
```python
from marie.query_planner import HitlRouterQueryDefinition, Query

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
            "data": {"total_amount": 15000, "vendor_type": "existing"},
        },
    ),
)
```

**Routing Logic**:
1. Check conditional rules first (`always_review_if`)
2. If any rule matches, route to `human_review`
3. Otherwise, route based on confidence:
   - `confidence >= auto_approve_threshold` → `auto_approve`
   - `confidence >= human_review_threshold` → `human_review`
   - `confidence < human_review_threshold` → `review` or `reject` (based on `below_threshold_action`)

**Conditional Rule Types**:
- `exists`: Field exists in data
- `empty`: Field is null/empty
- `equals`: Field equals value
- `gt`: Field greater than value
- `lt`: Field less than value
- `gte`: Field greater than or equal to value
- `lte`: Field less than or equal to value

## Configuration

### Database Configuration

HITL executors connect to the Marie Studio PostgreSQL database. Configure via environment variables:

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

All HITL executors support these configuration parameters:

```python
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

## Database Schema

HITL executors interact with these tables in the `marie_scheduler` schema:

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

## Workflow Integration

### Example: Complete HITL Workflow

```python
from marie.query_planner import (
    ConditionalQueryPlanBuilder,
    HitlApprovalQueryDefinition,
    HitlCorrectionQueryDefinition,
    HitlRouterQueryDefinition,
)


def create_invoice_processing_workflow():
    builder = ConditionalQueryPlanBuilder()

    # 1. Extract data from invoice
    extract = builder.add_node(
        task_id="extract_invoice_data",
        query_str="Extract invoice data",
        node_type=QueryType.COMPUTE,
        definition=ExecutorEndpointQueryDefinition(
            endpoint="extract/invoice", params={"model": "invoice_extractor_v2"}
        ),
    )

    # 2. Route based on confidence
    router = builder.add_node(
        task_id="hitl_router",
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

    # 3a. Auto-approve path (high confidence)
    auto_process = builder.add_node(
        task_id="auto_process",
        query_str="Auto-process high confidence invoices",
        dependencies=[router.task_id],
        node_type=QueryType.COMPUTE,
        definition=ExecutorEndpointQueryDefinition(endpoint="process/invoice"),
    )

    # 3b. Human review path (medium confidence)
    correction = builder.add_node(
        task_id="hitl_correction",
        query_str="Human correction of extracted data",
        dependencies=[router.task_id],
        node_type=QueryType.COMPUTE,
        definition=HitlCorrectionQueryDefinition(
            endpoint="hitl/correction",
            title="Review Invoice Data",
            correction_type="structured",
            fields=[...],
            auto_validate={"enabled": True, "confidence_threshold": 0.9},
        ),
    )

    approval = builder.add_node(
        task_id="hitl_approval",
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

    process_reviewed = builder.add_node(
        task_id="process_reviewed",
        query_str="Process reviewed invoices",
        dependencies=[approval.task_id],
        node_type=QueryType.COMPUTE,
        definition=ExecutorEndpointQueryDefinition(endpoint="process/invoice"),
    )

    # 3c. Reject path (low confidence)
    reject = builder.add_node(
        task_id="reject",
        query_str="Reject low confidence invoices",
        dependencies=[router.task_id],
        node_type=QueryType.COMPUTE,
        definition=NoopQueryDefinition(),
    )

    return builder.build()
```

## Testing

### Mock Workflows

See `/home/greg/dev/marieai/marie-ai/tests/integration/scheduler/mock_query_plans.py` for example HITL workflows:

- `mock_hitl_approval`: Simple approval workflow
- `mock_hitl_correction`: Data correction workflow
- `mock_hitl_router`: Confidence-based routing workflow
- `mock_hitl_complete_workflow`: Complex workflow combining all HITL nodes

### Running Tests

```bash
# Run HITL executor tests
cd /home/greg/dev/marieai/marie-ai
pytest tests/integration/scheduler/ -k hitl

# View mock workflows in Marie Studio
# 1. Start Marie Studio frontend/backend
# 2. Create a DAG run using one of the mock workflows
# 3. HITL requests will appear in the /hitl inbox
```

## TODO

### Notification Service

The notification service is not yet implemented. Future implementation will include:

**Features**:
- Email notifications when HITL requests are created
- In-app notifications in Marie Studio
- Reminder notifications for pending requests
- Escalation notifications

**Implementation Plan**:

1. **Create Notification Service** (`marie/services/notification_service.py`):
   ```python
   class NotificationService:
       def send_email(self, user_id, subject, body):
           # Email notification logic
           pass

       def send_in_app(self, user_id, title, message):
           # In-app notification logic
           pass

       def send_reminder(self, request_id):
           # Reminder logic
           pass
   ```

2. **Integrate with HITL Executors**:
   - Call notification service after creating HITL request
   - Schedule reminders based on notification config
   - Send completion notifications

3. **Email Templates**:
   - Create HTML email templates for different notification types
   - Support personalization (user name, request details)

4. **Configuration**:
   ```bash
   export MARIE_SMTP_HOST="smtp.gmail.com"
   export MARIE_SMTP_PORT="587"
   export MARIE_SMTP_USER="your-email@gmail.com"
   export MARIE_SMTP_PASSWORD="your-app-password"
   export MARIE_NOTIFICATION_FROM="noreply@marie-ai.com"
   ```

### Database Migration

Before using HITL executors, run the database migration:

```bash
cd /home/greg/dev/marieai/marie-studio
psql -U postgres -d marie_studio -f packages/@marie/db/migrations/005_add_hitl_tables.sql
```

Or use Prisma:

```bash
cd packages/@marie/db
pnpm prisma migrate dev
pnpm prisma:generate
```

## Troubleshooting

### Database Connection Errors

**Issue**: `psycopg2.OperationalError: could not connect to server`

**Solution**:
1. Verify database is running: `psql -U postgres -d marie_studio`
2. Check environment variables are set correctly
3. Verify PostgreSQL is listening on the correct host/port

### Timeout Issues

**Issue**: HITL requests timing out immediately

**Solution**:
1. Check timeout configuration in QueryDefinition
2. Verify `timeout.enabled` is set to `True`
3. Increase `timeout.duration_seconds` if needed
4. Check database for stuck requests: `SELECT * FROM marie_scheduler.hitl_requests WHERE status = 'pending'`

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
3. Check database for responses: `SELECT * FROM marie_scheduler.hitl_responses WHERE request_id = '<request_id>'`
4. Verify `poll_interval` is reasonable (default: 5s)

## Future Enhancements

- [ ] WebSocket/SSE support for real-time notifications (eliminate polling)
- [ ] Notification service with email/Slack/Teams integration
- [ ] Multi-user approval workflows with quorum logic
- [ ] Escalation workflows (auto-escalate after X reminders)
- [ ] Analytics dashboard (approval rates, response times, etc.)
- [ ] Active learning from corrections (improve models based on feedback)
- [ ] Bulk approval operations (approve multiple requests at once)
- [ ] SLA tracking and alerts (warn if requests near timeout)

## License

Same as Marie-AI project license.
