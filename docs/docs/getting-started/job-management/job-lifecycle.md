---
sidebar_position: 2
---

# Job lifecycle

Understanding how jobs move through the system is essential for building reliable document processing workflows.

## Job states

Every job has a state that indicates where it is in its lifecycle.

### Non-terminal states

Jobs in these states are still being processed:

| State | Description |
|-------|-------------|
| `CREATED` | Job has been submitted and is waiting to be scheduled |
| `RETRY` | Job failed and is scheduled for another attempt |
| `ACTIVE` | Job is currently being executed by an Executor |

### Terminal states

Jobs in these states have finished processing:

| State | Description | Success |
|-------|-------------|---------|
| `COMPLETED` | Job executed successfully | Yes |
| `SKIPPED` | Job intentionally not executed (branch condition not met) | Yes |
| `EXPIRED` | Job exceeded its time-to-live before completion | No |
| `CANCELLED` | Job was cancelled by user request | No |
| `FAILED` | Job execution failed after all retry attempts | No |

:::note
Both `COMPLETED` and `SKIPPED` are considered successful outcomes. A skipped job means the work was intentionally not executed, typically because a branch condition was not met in a DAG workflow.
:::

## State transitions

```text
                                    ┌──────────────────────────────────────────┐
                                    │                                          │
                                    ▼                                          │
┌─────────┐     ┌─────────┐     ┌─────────┐     ┌───────────┐                  │
│ CREATED │────▶│ ACTIVE  │────▶│COMPLETED│     │  SKIPPED  │                  │
└─────────┘     └─────────┘     └─────────┘     └───────────┘                  │
    │               │                               ▲                          │
    │               │                               │ (branch not taken)       │
    │               ▼                               │                          │
    │           ┌─────────┐                    ┌─────────┐                     │
    │           │ FAILED  │◀───────────────────│  RETRY  │─────────────────────┘
    │           └─────────┘  (retry exhausted) └─────────┘
    │               ▲                               ▲
    │               │                               │ (failure + retries left)
    │               │                               │
    ▼               │                               │
┌─────────┐         │           ┌─────────┐         │
│ EXPIRED │         └───────────│ ACTIVE  │─────────┘
└─────────┘         (no retries)└─────────┘
    ▲
    │ (TTL exceeded)
    │
┌─────────┐
│CANCELLED│◀─────── (user request from any non-terminal state)
└─────────┘
```

### Valid transitions

| From | To | Trigger |
|------|----|---------|
| `CREATED` | `ACTIVE` | Job scheduled and dispatched |
| `CREATED` | `EXPIRED` | TTL exceeded before scheduling |
| `CREATED` | `CANCELLED` | User cancellation |
| `ACTIVE` | `COMPLETED` | Successful execution |
| `ACTIVE` | `SKIPPED` | Branch condition not met |
| `ACTIVE` | `RETRY` | Execution failed, retries remaining |
| `ACTIVE` | `FAILED` | Execution failed, no retries left |
| `RETRY` | `ACTIVE` | Retry scheduled and dispatched |
| `RETRY` | `EXPIRED` | TTL exceeded during retry wait |
| `RETRY` | `CANCELLED` | User cancellation |

## Job submission

When you submit a job, it enters the `CREATED` state:

```python
from datetime import datetime, timedelta, timezone
from marie.scheduler.models import WorkInfo
from marie.scheduler.state import WorkState

now = datetime.now(timezone.utc)

job = WorkInfo(
    name="extract",
    priority=0,
    data={"asset_key": "s3://bucket/document.pdf"},
    state=WorkState.CREATED,
    retry_limit=2,
    retry_delay=5,
    retry_backoff=True,
    start_after=now,
    expire_in_seconds=3600,
    keep_until=now + timedelta(days=7),
)
```

### Delayed execution

Use `start_after` to schedule jobs for future execution:

```python
# Execute 1 hour from now
job = WorkInfo(
    name="extract",
    start_after=datetime.now(timezone.utc) + timedelta(hours=1),
    # ... other fields
)
```

### Expiration

Jobs that are not completed within `expire_in_seconds` transition to `EXPIRED`:

```python
job = WorkInfo(
    name="extract",
    expire_in_seconds=3600,  # 1 hour TTL
    # ... other fields
)
```

## Job execution

Once scheduled, a job transitions to `ACTIVE` and is dispatched to an Executor:

1. **Scheduling**: The scheduler selects the job based on priority and capacity
2. **Leasing**: The job is leased to prevent duplicate execution
3. **Dispatch**: The job is sent to an available Executor
4. **Execution**: The Executor processes the job
5. **Result**: The job transitions to a terminal state

### Execution timeout

Active jobs have a run timeout (`run_ttl_seconds`, default 60s). If execution exceeds this time, the lease expires and the job may be rescheduled.

## Job completion

### Success

When an Executor completes a job successfully, it transitions to `COMPLETED`:

```python
# Internal scheduler operation
await scheduler.complete(
    job_id=job.id,
    work_item=job,
    output_metadata={"result": "success", "pages_processed": 10}
)
```

### Skipped

Jobs can be marked as `SKIPPED` when a branch condition is not met in a DAG:

```python
# Internal scheduler operation
await scheduler.complete(
    job_id=job.id,
    work_item=job,
    output_metadata={"reason": "branch_not_taken"}
)
```

## Error handling

### Automatic retries

When a job fails, the scheduler checks if retries are available:

```python
job = WorkInfo(
    name="extract",
    retry_limit=3,      # Maximum 3 retry attempts
    retry_delay=5,      # 5 seconds initial delay
    retry_backoff=True, # Enable exponential backoff
    # ... other fields
)
```

With exponential backoff enabled, retry delays increase:
- Attempt 1: 5 seconds
- Attempt 2: 10 seconds
- Attempt 3: 20 seconds

### Failure

If all retries are exhausted, the job transitions to `FAILED`:

```python
# Internal scheduler operation
await scheduler.fail(
    job_id=job.id,
    work_item=job,
    output_metadata={"error": "Connection timeout", "attempts": 3}
)
```

### Cancellation

Users can cancel jobs that are not yet in a terminal state:

```bash
# Via REST API
curl http://localhost:54322/api/jobs/{job_id}/stop
```

```python
# Via Python API
await scheduler.cancel_job(job_id, work_item)
```

Cancelled jobs can be resumed:

```python
await scheduler.resume_job(job_id)
```

## Checking job state

### REST API

```bash
# Get job details
curl http://localhost:54322/api/jobs/{job_id}

# List jobs by state
curl http://localhost:54322/api/jobs/ACTIVE
curl http://localhost:54322/api/jobs/FAILED
```

### Python API

```python
from marie.scheduler.state import WorkState

# Get single job
job = await scheduler.get_job(job_id)
print(f"State: {job.state}")

# List jobs by state
failed_jobs = await scheduler.list_jobs(state=WorkState.FAILED)
```

## State helper methods

The `WorkState` enum provides helper methods:

```python
from marie.scheduler.state import WorkState

state = WorkState.COMPLETED

# Check if terminal (no further transitions)
state.is_terminal()  # True

# Check if successful terminal
state.is_successful_terminal()  # True (COMPLETED or SKIPPED)

# Check if work was executed
state.was_executed()  # True (False for SKIPPED, CANCELLED)

# Get all terminal states
WorkState.terminal_states()  # [COMPLETED, SKIPPED, EXPIRED, CANCELLED, FAILED]
```

## Best practices

1. **Set appropriate TTLs**: Use `expire_in_seconds` to prevent jobs from staying in the queue indefinitely

2. **Configure retries thoughtfully**: Balance between giving jobs enough chances and not wasting resources on consistently failing jobs

3. **Monitor failed jobs**: Set up alerts for jobs in `FAILED` state to investigate issues

4. **Use `keep_until`**: Specify how long job records should be retained for auditing

5. **Handle cancellation gracefully**: Design Executors to check for cancellation signals during long-running operations

## Next steps

- [DAG workflows](./dags.md) - Group jobs with dependencies
- [Scheduling](./scheduling.md) - Priority and SLA configuration
- [API reference](./api.md) - Complete API documentation
