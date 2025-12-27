---
sidebar_position: 6
---

# Job API reference

Complete reference for job management APIs, including REST endpoints and Python interfaces.

## REST API

The MarieGateway exposes job management endpoints over HTTP.

### Submit a job

Submit a new job for processing.

**Endpoint:** `POST /api/v1/invoke`

**Headers:**
- `Authorization: Bearer <token>` (required)
- `Content-Type: application/json`

**Request body:**

```json
{
  "header": {},
  "parameters": {
    "command": "extract",
    "asset_key": "s3://bucket/document.pdf",
    "pages": [1, 2, 3],
    "api_key": "optional-override"
  }
}
```

**Response:**

```json
{
  "header": {},
  "parameters": {
    "job_id": "01234567-89ab-cdef-0123-456789abcdef",
    "status": "submitted"
  },
  "data": null
}
```

**Example:**

```bash
curl -X POST http://localhost:54322/api/v1/invoke \
  -H "Authorization: Bearer your-token" \
  -H "Content-Type: application/json" \
  -d '{
    "parameters": {
      "command": "extract",
      "asset_key": "s3://bucket/document.pdf"
    }
  }'
```

### List jobs

Retrieve a list of jobs, optionally filtered by state.

**Endpoint:** `GET /api/jobs` or `GET /api/jobs/{state}`

**Parameters:**
- `state` (optional): Filter by job state (`CREATED`, `ACTIVE`, `COMPLETED`, `FAILED`, etc.)

**Response:**

```json
{
  "status": "OK",
  "result": [
    {
      "id": "01234567-89ab-cdef-0123-456789abcdef",
      "name": "extract",
      "state": "COMPLETED",
      "priority": 10,
      "created_at": "2024-01-15T10:30:00Z",
      "completed_at": "2024-01-15T10:31:00Z"
    }
  ]
}
```

**Examples:**

```bash
# List all jobs
curl http://localhost:54322/api/jobs

# List active jobs
curl http://localhost:54322/api/jobs/ACTIVE

# List failed jobs
curl http://localhost:54322/api/jobs/FAILED
```

### Stop a job

Stop a running or pending job.

**Endpoint:** `GET /api/jobs/{job_id}/stop`

**Response:**

```json
{
  "status": "OK",
  "result": "Job stopped"
}
```

**Example:**

```bash
curl http://localhost:54322/api/jobs/01234567-89ab-cdef-0123-456789abcdef/stop
```

### Delete a job

Delete a job record.

**Endpoint:** `DELETE /api/jobs/{job_id}`

**Response:**

```json
{
  "status": "OK",
  "result": "Job deleted"
}
```

**Example:**

```bash
curl -X DELETE http://localhost:54322/api/jobs/01234567-89ab-cdef-0123-456789abcdef
```

### Get capacity

Retrieve current executor capacity information.

**Endpoint:** `GET /api/capacity`

**Response:**

```json
{
  "status": "OK",
  "result": {
    "slots": [
      {
        "name": "extract",
        "capacity": 4,
        "target": 4,
        "used": 2,
        "available": 2,
        "holders": ["worker-1", "worker-2"],
        "notes": ""
      }
    ],
    "totals": {
      "capacity": 12,
      "used": 5,
      "available": 7
    }
  }
}
```

**Example:**

```bash
curl http://localhost:54322/api/capacity
```

### Get debug information

Retrieve scheduler debug information.

**Endpoint:** `GET /api/debug`

**Response:**

```json
{
  "status": "OK",
  "result": {
    "active_dags": 5,
    "pending_jobs": 23,
    "active_jobs": 8,
    "scheduler_status": "running"
  }
}
```

**Example:**

```bash
curl http://localhost:54322/api/debug
```

### Reset active DAGs

Reset all active DAGs (admin operation).

**Endpoint:** `POST /api/debug/reset-dags`

**Response:**

```json
{
  "status": "OK",
  "result": {
    "success": true,
    "reset_count": 5
  }
}
```

**Example:**

```bash
curl -X POST http://localhost:54322/api/debug/reset-dags
```

## Python API

### PostgreSQLJobScheduler

The main scheduler class for job management.

#### submit_job

Submit a job for processing.

```python
from marie.scheduler import PostgreSQLJobScheduler
from marie.scheduler.models import WorkInfo
from marie.scheduler.state import WorkState
from datetime import datetime, timedelta, timezone

scheduler = PostgreSQLJobScheduler(config=config)

now = datetime.now(timezone.utc)
job = WorkInfo(
    name="extract",
    priority=10,
    data={"asset_key": "s3://bucket/document.pdf"},
    state=WorkState.CREATED,
    retry_limit=2,
    retry_delay=5,
    retry_backoff=True,
    start_after=now,
    expire_in_seconds=3600,
    keep_until=now + timedelta(days=7),
)

job_id = await scheduler.submit_job(job, overwrite=True)
```

**Parameters:**
- `work_info` (WorkInfo): Job definition
- `overwrite` (bool): Replace existing job with same ID (default: `True`)

**Returns:** Job ID string

#### list_jobs

List jobs with optional filtering.

```python
from marie.scheduler.state import WorkState

# List all jobs
jobs = await scheduler.list_jobs()

# Filter by state
active_jobs = await scheduler.list_jobs(state=WorkState.ACTIVE)

# Filter by queue
queue_jobs = await scheduler.list_jobs(queue="high-priority")

# Limit results
recent_jobs = await scheduler.list_jobs(limit=100)
```

**Parameters:**
- `state` (WorkState, optional): Filter by job state
- `queue` (str, optional): Filter by queue name
- `limit` (int, optional): Maximum results to return

**Returns:** List of job records

#### get_job

Retrieve a single job by ID.

```python
job = await scheduler.get_job(job_id)
if job:
    print(f"State: {job.state}")
    print(f"Priority: {job.priority}")
```

**Parameters:**
- `job_id` (str): Job identifier

**Returns:** Job record or `None`

#### cancel_job

Cancel a pending or active job.

```python
await scheduler.cancel_job(job_id, work_item)
```

**Parameters:**
- `job_id` (str): Job identifier
- `work_item` (WorkInfo): Current job state

#### resume_job

Resume a cancelled job.

```python
await scheduler.resume_job(job_id)
```

**Parameters:**
- `job_id` (str): Job identifier

#### complete

Mark a job as completed (internal use).

```python
await scheduler.complete(
    job_id=job_id,
    work_item=job,
    output_metadata={"pages_processed": 10}
)
```

**Parameters:**
- `job_id` (str): Job identifier
- `work_item` (WorkInfo): Job state
- `output_metadata` (dict): Completion metadata
- `force` (bool): Force completion even if not active

#### fail

Mark a job as failed (internal use).

```python
await scheduler.fail(
    job_id=job_id,
    work_item=job,
    output_metadata={"error": "Connection timeout"}
)
```

**Parameters:**
- `job_id` (str): Job identifier
- `work_item` (WorkInfo): Job state
- `output_metadata` (dict): Failure metadata

### WorkInfo model

Job definition model.

```python
from marie.scheduler.models import WorkInfo
from marie.scheduler.state import WorkState
from datetime import datetime, timedelta, timezone

now = datetime.now(timezone.utc)

job = WorkInfo(
    # Required fields
    name="extract",                    # Executor endpoint
    priority=0,                        # Scheduling priority
    data={},                           # Job payload
    state=WorkState.CREATED,           # Initial state
    retry_limit=2,                     # Max retries
    retry_delay=5,                     # Retry delay (seconds)
    retry_backoff=True,                # Exponential backoff
    start_after=now,                   # Earliest start
    expire_in_seconds=3600,            # TTL
    keep_until=now + timedelta(days=7),# Retention

    # Optional fields
    id="custom-id",                    # Custom ID (auto-generated if omitted)
    dag_id="batch-123",                # DAG association
    dependencies=["job-1", "job-2"],   # Job dependencies
    job_level=0,                       # DAG depth level
    soft_sla=now + timedelta(hours=1), # Target completion
    hard_sla=now + timedelta(hours=4), # Deadline
    policy="default",                  # Queue policy
)
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | str | No | Unique identifier (auto-generated) |
| `dag_id` | str | No | DAG association |
| `name` | str | Yes | Executor endpoint |
| `priority` | int | Yes | Scheduling priority |
| `data` | dict | Yes | Job payload |
| `state` | WorkState | Yes | Current state |
| `retry_limit` | int | Yes | Maximum retries |
| `retry_delay` | int | Yes | Retry delay (seconds) |
| `retry_backoff` | bool | Yes | Enable exponential backoff |
| `start_after` | datetime | Yes | Earliest execution time |
| `expire_in_seconds` | int | Yes | Time-to-live |
| `keep_until` | datetime | Yes | Record retention |
| `dependencies` | list[str] | No | Dependent job IDs |
| `job_level` | int | No | DAG depth (default: 0) |
| `soft_sla` | datetime | No | Target completion |
| `hard_sla` | datetime | No | Hard deadline |
| `policy` | str | No | Queue policy name |

### WorkState enum

Job state enumeration.

```python
from marie.scheduler.state import WorkState

# Check state properties
state = WorkState.COMPLETED

state.is_terminal()              # True
state.is_successful_terminal()   # True (COMPLETED or SKIPPED)
state.was_executed()             # True (False for SKIPPED, CANCELLED)

# Get all terminal states
WorkState.terminal_states()      # [COMPLETED, SKIPPED, EXPIRED, CANCELLED, FAILED]
```

### ExistingWorkPolicy enum

Control behavior when submitting jobs with existing IDs.

```python
from marie.scheduler.models import ExistingWorkPolicy

# Available policies
ExistingWorkPolicy.ALLOW_ALL        # Accept all submissions
ExistingWorkPolicy.REJECT_ALL       # Reject all submissions
ExistingWorkPolicy.REPLACE          # Replace non-terminal jobs
ExistingWorkPolicy.ALLOW_DUPLICATE  # Allow duplicate submissions
ExistingWorkPolicy.REJECT_DUPLICATE # Reject duplicates (default)
```

## Error handling

### REST API errors

Errors return JSON with status and message:

```json
{
  "status": "error",
  "message": "Job not found",
  "code": "NOT_FOUND"
}
```

Common HTTP status codes:

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad request (invalid parameters) |
| 401 | Unauthorized (missing/invalid token) |
| 404 | Not found (job doesn't exist) |
| 500 | Internal server error |

### Python exceptions

```python
try:
    job = await scheduler.get_job(job_id)
except Exception as e:
    print(f"Error: {e}")
```

## Next steps

- [Job lifecycle](./job-lifecycle.md) - Understanding job states
- [Scheduling](./scheduling.md) - Priority and retry configuration
- [Maintenance](./maintenance.md) - Database operations
