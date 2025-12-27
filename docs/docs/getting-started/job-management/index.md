---
sidebar_position: 1
---

# Job management

Marie-AI includes a production-grade job scheduler for managing document processing workloads. The scheduler handles job queuing, prioritization, retries, and distributed execution across multiple workers.

## What is job management

Job management in Marie-AI provides:

- **Job queuing**: Submit jobs to named queues for ordered processing
- **Priority scheduling**: Higher-priority jobs execute before lower-priority ones
- **Retry handling**: Automatic retries with configurable backoff policies
- **SLA tracking**: Soft and hard deadlines for job completion
- **DAG workflows**: Group related jobs with dependencies
- **Distributed execution**: Scale across multiple workers with slot-based capacity management

## Key concepts

### Jobs

A job represents a unit of work to be processed by an Executor. Each job has:

| Property | Description |
|----------|-------------|
| `id` | Unique identifier (UUID) |
| `name` | Executor endpoint (e.g., `extract`, `classify`) |
| `priority` | Scheduling priority (higher = more important) |
| `data` | Job payload and parameters |
| `state` | Current lifecycle state |

### States

Jobs transition through states during their lifecycle:

| State | Description | Terminal |
|-------|-------------|----------|
| `CREATED` | Job submitted, waiting to be scheduled | No |
| `RETRY` | Scheduled for retry after failure | No |
| `ACTIVE` | Currently executing | No |
| `COMPLETED` | Finished successfully | Yes |
| `SKIPPED` | Intentionally skipped (branch not taken) | Yes |
| `EXPIRED` | Exceeded time-to-live | Yes |
| `CANCELLED` | User cancelled | Yes |
| `FAILED` | Execution failed | Yes |

### Queues

Queues organize jobs by purpose or priority level. Each queue can have its own retry policy and expiration settings.

### DAGs

A DAG (Directed Acyclic Graph) groups multiple jobs that should execute together. DAGs support:

- Job dependencies (execute in order)
- Batch tracking (monitor progress across all jobs)
- SLA management (deadlines apply to entire workflow)

## Quick start

### Submit a job via REST API

```bash
curl -X POST http://localhost:54322/api/v1/invoke \
  -H "Authorization: Bearer your-token" \
  -H "Content-Type: application/json" \
  -d '{
    "parameters": {
      "command": "extract",
      "asset_key": "s3://bucket/document.pdf",
      "pages": [1, 2, 3]
    }
  }'
```

### List jobs

```bash
# List all jobs
curl http://localhost:54322/api/jobs

# Filter by state
curl http://localhost:54322/api/jobs/ACTIVE
```

### Check job status

```bash
curl http://localhost:54322/api/jobs/{job_id}
```

## Job submission model

When submitting jobs programmatically, use the `WorkInfo` model:

```python
from datetime import datetime, timedelta, timezone
from marie.scheduler.models import WorkInfo
from marie.scheduler.state import WorkState

now = datetime.now(timezone.utc)

job = WorkInfo(
    name="extract",                    # Executor endpoint
    priority=10,                       # Higher = more important
    data={                             # Job payload
        "asset_key": "s3://bucket/doc.pdf",
        "pages": [1, 2, 3],
    },
    state=WorkState.CREATED,
    retry_limit=2,                     # Max retry attempts
    retry_delay=5,                     # Seconds between retries
    retry_backoff=True,                # Exponential backoff
    start_after=now,                   # Earliest start time
    expire_in_seconds=3600,            # TTL (1 hour)
    keep_until=now + timedelta(days=7),
    soft_sla=now + timedelta(hours=1), # Target completion
    hard_sla=now + timedelta(hours=4), # Deadline
)
```

## Architecture overview

```text
┌─────────────┐     ┌─────────────┐     ┌─────────────────────┐
│   Client    │────▶│   Gateway   │────▶│      Scheduler      │
│  (REST/gRPC)│     │  (MarieGW)  │     │   (PostgreSQL)      │
└─────────────┘     └─────────────┘     └─────────────────────┘
                                                  │
                    ┌─────────────────────────────┼─────────────────────────────┐
                    │                             │                             │
                    ▼                             ▼                             ▼
            ┌─────────────┐              ┌─────────────┐              ┌─────────────┐
            │  Executor 1 │              │  Executor 2 │              │  Executor N │
            │   (OCR)     │              │  (Extract)  │              │  (Classify) │
            └─────────────┘              └─────────────┘              └─────────────┘
```

The scheduler:

1. Receives job submissions from the Gateway
2. Stores jobs in PostgreSQL
3. Schedules jobs based on priority and capacity
4. Dispatches jobs to available Executors
5. Tracks completion and handles retries

## Next steps

- [Job lifecycle](./job-lifecycle.md) - States, transitions, and error handling
- [DAG workflows](./dags.md) - Batch processing with dependencies
- [Scheduling](./scheduling.md) - Priority, SLAs, and retry policies
- [Configuration](./configuration.md) - Scheduler setup and tuning
- [API reference](./api.md) - REST and Python APIs
- [Maintenance](./maintenance.md) - Database operations and cleanup
