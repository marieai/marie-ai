
# Marie Scheduler: Execution and Scheduling Behavior

## Overview

This document describes the scheduling and execution flow used by the Marie Scheduler. It explains how the system selects and dispatches jobs across distributed executors using dynamic, SLA-aware strategies and resource constraints.

---

## Scheduling Strategy

Marie Scheduler uses a global planner that evaluates job execution order based on multiple criteria:

1. **Critical-path preference** — Jobs deeper in the DAG are prioritized to unblock dependent jobs.
2. **SLA-based priority** — Jobs closer to (or past) their SLA deadlines are ranked higher.
3. **Executor capacity** — Jobs targeting executors with more free slots are favored.
4. **DAG reuse** — Existing DAGs are preferred to avoid startup overhead.
5. **Estimated runtime** — Shorter jobs are scheduled first when other factors are equal.
6. **Burst boost** — Recently activated DAGs are temporarily boosted to minimize cold start latency.

---

## SLA-Based Priority System

Marie Scheduler calculates a numeric priority score for every job based on SLA deadlines. These values are used directly by the global planner to determine execution order.

### Priority Semantics

- **Higher values = more urgent**
- Jobs are scored using 15-minute time buckets
- Late jobs accumulate higher scores as time passes
- Upcoming jobs receive lower scores inversely weighted by time
- Jobs with no SLA are assigned the lowest score (`0`)

---

### Priority Tiers

| Tier                    | Condition                          | Priority Range  |
|-------------------------|-------------------------------------|-----------------|
| **Overdue Hard SLA**    | `NOW() > hard_sla`                  | 1000–1999       |
| **Overdue Soft SLA**    | `NOW() > soft_sla` (no hard SLA)    | 500–999         |
| **Upcoming Soft SLA**   | `soft_sla > NOW()`                  | 1–499           |
| **No SLA**              | SLA not defined                     | 0               |

- Each bucket = 15 minutes (900 seconds)
- `LEAST(...)` and `GREATEST(...)` clamp scores to prevent runaway values or negatives

---

### Priority Update Logic

The following SQL logic is used by the Marie Scheduler to update job priorities:

```sql
CREATE OR REPLACE FUNCTION marie_scheduler.refresh_job_priority()
RETURNS void AS $$
DECLARE
    now_time TIMESTAMP := NOW();
BEGIN
    UPDATE marie_scheduler.job
    SET priority = (
          CASE
            WHEN hard_sla IS NOT NULL AND NOW() > hard_sla THEN
              1000 + LEAST(
                999,
                FLOOR(EXTRACT(EPOCH FROM (NOW() - hard_sla)) / 900)
              )  -- overdue hard SLA
            WHEN soft_sla IS NOT NULL AND NOW() > soft_sla THEN
              500 + LEAST(
                499,
                FLOOR(EXTRACT(EPOCH FROM (NOW() - soft_sla)) / 900)
              )   -- overdue soft SLA
            WHEN soft_sla IS NOT NULL THEN
              GREATEST(
                1,
                500 - CEIL(EXTRACT(EPOCH FROM (soft_sla - NOW())) / 900)
              )   -- upcoming soft SLA
            ELSE
              0   -- no SLA
        END
    )
    WHERE state IN ('created', 'retry');
END;

```

This function is called:

- After N (default 10) job creations
- Periodically via a background scheduler(e.g., every 1 minutes via pg_cron)

---

## GlobalPriorityExecutionPlanner

This is the core algorithm used to compute job sort order globally across all queues:

### Sort Order:
- `job_level`: deeper DAG jobs scheduled first
- `priority`: higher number = more urgent
- `free_slots`: more available slots preferred
- `is_new_dag`: existing DAGs preferred
- `estimated_runtime`: shorter is better
- `burst_boost`: recently activated DAGs prioritized

---

## ASCII Diagram: Global Priority Execution Planner

```text
+---------------------------+
|  Marie Scheduler (_poll) |
+---------------------------+
             |
             v
  +-----------------------------+
  | Fetch ready jobs by queue   |
  +-----------------------------+
             |
             v
  +-----------------------------+
  | Flatten & filter jobs       |
  +-----------------------------+
             |
             v
  +---------------------------------------------+
  | GlobalPriorityExecutionPlanner.plan()       |
  +---------------------------------------------+
             |
             v
  +--------------------------------------------------------------+
  | Sort by:                                                     |
  |   - job_level (desc)                                         |
  |   - priority (desc: higher number = more urgent)             |
  |   - free_slots (desc)                                        |
  |   - DAG activity (existing DAGs preferred)                   |
  |   - estimated_runtime (asc)                                  |
  |   - burst_boost (prioritize recently activated DAGs)         |
  +--------------------------------------------------------------+
             |
             v
  +-----------------------------+
  | Schedule jobs on executors  |
  +-----------------------------+
             |
             v
  +-----------------------------+
  | mark_as_active(job)         |
  |  → Marks job as active in DB|
  |  → Activates DAG if needed  |
  +-----------------------------+
             |
             v
  +-----------------------------+
  | enqueue(job)                |
  |  → Sends job to executor    |
  |  → Decrements executor slot |
  +-----------------------------+
             |
             v
  +-----------------------------+
  | resolve_dag_status(job)     |
  |  → Checks if DAG node is    |
  |    terminal or unblocks next|
  |  → Updates DAG state        |
  +-----------------------------+
             |
             v
  +-----------------------------+
  | Await confirmation via ETCD |
  |  → Waits for executor ack   |
  |  → Resets scheduling state  |
  +-----------------------------+
```

---

## DAG-Level Completion

When jobs complete, DAG state is resolved to determine if execution is finished.

### ASCII Diagram: DAG-Level Resolution Flow

```text
+-------------------------------+
|  Job marked as completed      |
+-------------------------------+
             |
             v
  +-------------------------------+
  | resolve_dag_status(job)       |
  +-------------------------------+
             |
             v
  +-------------------------------------------+
  | Fetch DAG by dag_id                        |
  |  → Is this the last node in the DAG?       |
  |  → Are all downstream jobs completed?      |
  +-------------------------------------------+
             |
             v
  +-------------------------------+
  | YES: Final job completed      |
  | → DAG marked as completed     |
  | → Update completed_on field   |
  +-------------------------------+
             |
             v
  | Notify and archive DAG state  |
             |
             v
        [END – DAG Completed]

             |
             |
             |
             v
  +-------------------------------+
  | NO: DAG still active          |
  | → Update internal DAG graph   |
  | → Unblock dependent jobs      |
  +-------------------------------+
             |
             v
  | Enqueue next ready jobs       |
  | → Re-enter scheduling loop    |
             |
             v
        [CONTINUE DAG Execution]
```

---

## Conclusion

Marie Scheduler's planning system balances SLA pressure, system throughput, and execution fairness. It is designed to support bursty DAG workloads, enforce soft and hard deadlines, and provide predictable behavior under constrained resources.

---

## Pluggable Execution Planner Interface

The Marie Scheduler supports **pluggable scheduling strategies** via the execution planner abstraction. This enables different scheduling policies to be used depending on system goals, workload types, or SLA conditions.

### Default Planner: `GlobalPriorityExecutionPlanner`

This is the standard planner described earlier, which considers SLA priority, DAG structure, available slots, and recent DAG activity.

---

### Alternative Planners

Marie also ships with alternate planners which can be swapped dynamically:

#### `HRRNExecutionPlanner` — Highest Response Ratio Next

This strategy favors jobs that have waited longer relative to their estimated runtime. The formula used is:

```text
Response Ratio = (Wait Time + Estimated Runtime) / Estimated Runtime
```

- Promotes fairness for long-waiting jobs
- Deprioritizes new or short-duration tasks
- Balances aging jobs against resource availability

#### `SJFSExecutionPlanner` — Shortest Job First

This strategy prioritizes jobs with the shortest estimated runtime first:

- Optimizes throughput by clearing quick jobs
- Reduces average response time
- May penalize long-running jobs

---

## Swapping Planners

To change the planner, the scheduler can be configured with a different class that implements the `plan()` method signature:

```python
def plan(
    self,
    jobs: Sequence[FlatJob],
    slots: dict[str, int],
    active_dags: set[str],
    recently_activated_dags: set[str] = set(),
) -> Sequence[FlatJob]:
    ...
```

