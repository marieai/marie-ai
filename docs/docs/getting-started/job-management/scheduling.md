---
sidebar_position: 4
---

# Scheduling

The Marie-AI scheduler uses a priority-based system to determine which jobs execute first, combined with SLA tracking and intelligent retry policies.

## Priority system

Jobs are scheduled based on multiple factors, ranked in order of importance:

| Factor | Description |
|--------|-------------|
| 1. Runnable | Jobs with available executor slots rank higher |
| 2. Existing DAGs | Continuing existing DAGs before starting new ones |
| 3. Job level | Deeper jobs (critical path) execute first |
| 4. Priority | User-defined priority value |
| 5. Available slots | Executors with more free slots preferred |
| 6. Estimated runtime | Shorter jobs preferred (tie-breaker) |
| 7. FIFO | Original submission order |

### Setting priority

Higher priority values mean more important jobs:

```python
from marie.scheduler.models import WorkInfo
from marie.scheduler.state import WorkState

# High priority job
urgent_job = WorkInfo(
    name="extract",
    priority=100,  # High priority
    # ... other fields
)

# Normal priority job
normal_job = WorkInfo(
    name="extract",
    priority=0,  # Default priority
    # ... other fields
)
```

### Priority guidelines

| Priority | Use case |
|----------|----------|
| 100+ | Critical, time-sensitive processing |
| 50-99 | Important, expedited processing |
| 1-49 | Normal priority |
| 0 | Default, lowest priority |

## SLA handling

SLAs (Service Level Agreements) define target and deadline times for job completion.

### SLA types

| Type | Description |
|------|-------------|
| `soft_sla` | Target completion time; missing triggers alerts but doesn't fail the job |
| `hard_sla` | Hard deadline; missing typically fails the job or triggers escalation |

### Setting SLAs

```python
from datetime import datetime, timedelta, timezone

now = datetime.now(timezone.utc)

job = WorkInfo(
    name="extract",
    soft_sla=now + timedelta(hours=1),   # Target: 1 hour
    hard_sla=now + timedelta(hours=4),   # Deadline: 4 hours
    # ... other fields
)
```

### SLA monitoring

The scheduler tracks SLA compliance through heartbeat metrics:

- Jobs approaching `soft_sla` may be prioritized
- Jobs exceeding `hard_sla` may transition to `EXPIRED`
- SLA metrics are available through the monitoring endpoints

## Retry policies

When jobs fail, the scheduler can automatically retry them based on configured policies.

### Retry configuration

```python
from marie.scheduler.models import WorkInfo, RetryPolicy

job = WorkInfo(
    name="extract",
    retry_limit=3,       # Maximum retry attempts
    retry_delay=5,       # Initial delay in seconds
    retry_backoff=True,  # Enable exponential backoff
    # ... other fields
)
```

### Default values

| Setting | Default | Description |
|---------|---------|-------------|
| `retry_limit` | 2 | Maximum retry attempts |
| `retry_delay` | 2 | Initial delay between retries (seconds) |
| `retry_backoff` | true | Enable exponential backoff |
| `timeout_retry_limit` | 3 | Additional retries for timeout failures |

### Backoff calculation

With exponential backoff enabled, retry delays increase:

```text
Attempt 1: retry_delay seconds
Attempt 2: retry_delay * 2 seconds
Attempt 3: retry_delay * 4 seconds
...
```

Example with `retry_delay=5`:
- Retry 1: 5 seconds
- Retry 2: 10 seconds
- Retry 3: 20 seconds

### Backoff types

| Type | Description |
|------|-------------|
| `EXPONENTIAL_BACKOFF` | Delay doubles each attempt |
| `FIXED_BACKOFF` | Constant delay between attempts |

## Slot and capacity management

The scheduler tracks available capacity across executors to prevent overloading.

### How slots work

Each executor has a limited number of slots (concurrent job capacity):

```text
┌─────────────────────────────────────────────────┐
│                   Scheduler                      │
│                                                  │
│   Executor: extract    Slots: [■][■][□][□]      │
│   Executor: classify   Slots: [■][□][□]          │
│   Executor: ocr        Slots: [■][■][■][□][□]   │
│                                                  │
│   ■ = occupied    □ = available                 │
└─────────────────────────────────────────────────┘
```

### Capacity-aware scheduling

The scheduler:

1. Queries available slots from the capacity manager
2. Only schedules jobs for executors with free slots
3. Prefers executors with more available capacity (load balancing)
4. Releases slots when jobs complete or fail

### Checking capacity

```bash
# Via REST API
curl http://localhost:54322/api/capacity
```

Response:

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

## Distributed scheduling

For high-availability deployments, Marie-AI supports distributed scheduling with lease-based coordination.

### Lease management

When `distributed_scheduler=True`, the scheduler uses database-level leases:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lease_ttl_seconds` | 5 | How long a scheduler holds a job lease |
| `run_ttl_seconds` | 60 | Maximum execution time before lease expires |

### How distributed scheduling works

1. **Lease acquisition**: Scheduler reserves jobs in the database
2. **Lease renewal**: Active jobs renew leases periodically
3. **Lease expiration**: Expired leases allow job rescheduling
4. **Conflict resolution**: Database ensures only one scheduler processes each job

### Enabling distributed mode

```yaml
scheduler:
  distributed_scheduler: true
  lease_ttl_seconds: 5
  run_ttl_seconds: 60
```

## Poll intervals

The scheduler dynamically adjusts its polling frequency based on activity:

| Interval | Duration | Condition |
|----------|----------|-----------|
| Minimum | 0.25s | Active work available |
| Short | 1.0s | No ready work but slots available |
| Initial | 2.25s | Starting point |
| Maximum | 8.0s | Idle or repeated failures |

This adaptive polling reduces database load during idle periods while maintaining responsiveness when jobs are available.

## Execution planner

The `GlobalPriorityExecutionPlanner` ranks jobs for execution using a composite sort key:

```python
sort_key = (
    is_blocked,           # False < True (runnable first)
    is_new_dag,           # False < True (existing DAGs first)
    -job_level,           # Descending (deeper jobs first)
    -priority,            # Descending (higher priority first)
    -free_slots,          # Descending (more slots preferred)
    estimated_runtime,    # Ascending (shorter jobs first)
    fifo_index            # Ascending (older jobs first)
)
```

## Best practices

1. **Use priority sparingly**: Reserve high priorities for genuinely urgent work

2. **Set realistic SLAs**: Base SLAs on actual processing times plus buffer

3. **Configure retries appropriately**:
   - Use higher retry limits for transient failures (network issues)
   - Use lower limits for deterministic failures (invalid input)

4. **Monitor capacity**: Ensure enough executor slots for expected workload

5. **Balance DAG sizes**: Very large DAGs can dominate scheduling; consider splitting

6. **Review failed jobs**: High failure rates may indicate configuration or capacity issues

## Next steps

- [Configuration](./configuration.md) - Scheduler configuration options
- [API reference](./api.md) - REST and Python APIs
- [Maintenance](./maintenance.md) - Database operations
