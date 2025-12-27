---
sidebar_position: 3
---

# DAG workflows

A DAG (Directed Acyclic Graph) groups multiple jobs that should execute together as a workflow. DAGs enable batch processing with dependencies, progress tracking, and SLA management across related jobs.

## What is a DAG

A DAG represents a workflow where:

- Multiple jobs are grouped under a single identifier
- Jobs can have dependencies on other jobs
- The workflow completes when all jobs reach terminal states
- SLAs apply to the entire workflow, not individual jobs

```text
       ┌─────────┐
       │  Start  │
       └────┬────┘
            │
            ▼
       ┌─────────┐
       │   OCR   │
       └────┬────┘
            │
     ┌──────┴──────┐
     │             │
     ▼             ▼
┌─────────┐   ┌─────────┐
│Classify │   │ Extract │
└────┬────┘   └────┬────┘
     │             │
     └──────┬──────┘
            │
            ▼
       ┌─────────┐
       │  Store  │
       └─────────┘
```

## When to use DAGs

Use DAGs when you need to:

- **Process multi-page documents**: Each page can be a separate job with a shared document ID
- **Chain processing steps**: OCR → Classification → Extraction → Storage
- **Track batch progress**: Monitor completion percentage across many jobs
- **Apply workflow SLAs**: Set deadlines for entire document processing, not individual steps
- **Handle dependencies**: Ensure jobs execute in the correct order

For simple, independent jobs, submit them directly without a DAG.

## Basic DAG structure

A DAG consists of:

| Component | Description |
|-----------|-------------|
| `dag_id` | Unique identifier for the workflow |
| Jobs | Individual work items with `dag_id` reference |
| Dependencies | Job-to-job execution ordering |
| State | Overall workflow state (mirrors job states) |
| SLAs | Soft and hard deadlines for the workflow |

### Creating a DAG

Jobs are associated with a DAG through their `dag_id` field:

```python
from datetime import datetime, timedelta, timezone
from marie.scheduler.models import WorkInfo
from marie.scheduler.state import WorkState

now = datetime.now(timezone.utc)
dag_id = "doc-processing-12345"

# First job: OCR
ocr_job = WorkInfo(
    id="job-ocr-001",
    dag_id=dag_id,
    name="ocr",
    priority=10,
    data={"asset_key": "s3://bucket/document.pdf", "page": 1},
    state=WorkState.CREATED,
    job_level=0,  # First level in DAG
    retry_limit=2,
    retry_delay=5,
    retry_backoff=True,
    start_after=now,
    expire_in_seconds=3600,
    keep_until=now + timedelta(days=7),
)

# Second job: Extract (depends on OCR)
extract_job = WorkInfo(
    id="job-extract-001",
    dag_id=dag_id,
    name="extract",
    priority=10,
    data={"asset_key": "s3://bucket/document.pdf", "page": 1},
    state=WorkState.CREATED,
    job_level=1,  # Second level
    dependencies=["job-ocr-001"],  # Must complete before this runs
    retry_limit=2,
    retry_delay=5,
    retry_backoff=True,
    start_after=now,
    expire_in_seconds=3600,
    keep_until=now + timedelta(days=7),
)
```

## Dependencies

Jobs can depend on other jobs within the same DAG. A job only becomes ready for execution when all its dependencies have completed successfully.

### Defining dependencies

Use the `dependencies` field with a list of job IDs:

```python
job = WorkInfo(
    id="job-003",
    dag_id="my-dag",
    dependencies=["job-001", "job-002"],  # Both must complete first
    # ... other fields
)
```

### Dependency resolution

The scheduler automatically:

1. Tracks which jobs have pending dependencies
2. Marks jobs as ready when dependencies complete
3. Handles cascading failures (if a dependency fails, dependent jobs may be skipped)

```text
Dependencies:           Execution Order:

job-001 ─────┐
             ├──▶ job-003    job-001 → job-002 → job-003
job-002 ─────┘                  (parallel)
```

### Job levels

The `job_level` field indicates depth in the DAG. Jobs at lower levels are prioritized:

| Level | Description |
|-------|-------------|
| 0 | Entry points (no dependencies) |
| 1 | Depends on level 0 jobs |
| 2 | Depends on level 1 jobs |
| ... | ... |

The scheduler uses job levels to:
- Prioritize critical path jobs
- Estimate workflow progress
- Optimize execution order

## DAG states and completion

### DAG states

DAGs have states that mirror job states:

| State | Description |
|-------|-------------|
| `CREATED` | DAG submitted, jobs waiting |
| `ACTIVE` | At least one job is executing |
| `COMPLETED` | All jobs completed successfully |
| `FAILED` | One or more jobs failed |

### Completion criteria

A DAG is considered complete when all its jobs are in terminal states:

- **Success**: All jobs are `COMPLETED` or `SKIPPED`
- **Failure**: At least one job is `FAILED`, `EXPIRED`, or `CANCELLED`

### Notifications

When a DAG completes, the scheduler:

1. Updates the DAG state in the database
2. Sends a completion notification
3. Triggers any configured webhooks or callbacks

## Monitoring DAGs

### REST API

```bash
# Get DAG status (via deployment-status endpoint)
curl http://localhost:54322/api/deployment-status

# List jobs in a DAG
curl "http://localhost:54322/api/jobs?dag_id=doc-processing-12345"
```

### Progress tracking

Calculate DAG progress by counting jobs in terminal states:

```python
jobs = await scheduler.list_jobs(dag_id=dag_id)
total = len(jobs)
completed = sum(1 for j in jobs if j.state.is_terminal())
progress = (completed / total) * 100 if total > 0 else 0
```

## DAG SLAs

SLAs can be set at the DAG level to track overall workflow deadlines:

```python
from datetime import datetime, timedelta, timezone

now = datetime.now(timezone.utc)

# When creating jobs for the DAG
job = WorkInfo(
    dag_id="my-dag",
    soft_sla=now + timedelta(hours=1),  # Target: complete within 1 hour
    hard_sla=now + timedelta(hours=4),  # Deadline: must complete within 4 hours
    # ... other fields
)
```

| SLA Type | Description |
|----------|-------------|
| `soft_sla` | Target completion time; missing triggers alerts |
| `hard_sla` | Hard deadline; missing may fail the workflow |

## Best practices

1. **Use meaningful DAG IDs**: Include document ID or batch identifier for traceability

2. **Set job levels correctly**: Ensure levels reflect actual dependency depth

3. **Handle partial failures**: Design workflows to handle some jobs failing while others succeed

4. **Monitor DAG progress**: Track completion percentage for long-running workflows

5. **Set appropriate SLAs**: Use `soft_sla` for monitoring and `hard_sla` for critical deadlines

## Next steps

- [Scheduling](./scheduling.md) - Priority and capacity management
- [Configuration](./configuration.md) - DAG manager settings
- [API reference](./api.md) - Complete API documentation
