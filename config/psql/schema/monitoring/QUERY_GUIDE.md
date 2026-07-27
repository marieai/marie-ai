# Marie scheduler monitoring queries

The files in this directory provide read-only operational diagnostics,
throughput reports, cohort analytics, and optional monitoring views for the
PostgreSQL scheduler.

Monitoring files are not loaded automatically with the scheduler schema. Core
tables, generated time-slot columns, and helper functions are created by the
numbered migrations in `config/psql/schema`.

## Query inventory

| File | Purpose | Execution model |
| --- | --- | --- |
| `throughput_analysis.sql` | Hourly and overall query-plan and task throughput | Installs read-only SQL functions |
| `submission_lifecycle_analysis.sql` | Locate a gateway submission across DAG, task, attempt, and worker records | One read-only SQL statement |
| `outstanding_work_analysis.sql` | Queue totals, recent arrivals, leases, blockers, and outstanding jobs | Read-only `psql` script |
| `job_failure_analysis.sql` | One job's scheduler, worker, error, retry, and attempt timeline | Read-only SQL script |
| `example_queries.sql` | Creation-time cohorts, SLA analytics, and load distribution | Individual read-only queries |
| `slots_basis.sql` | Outstanding jobs distributed across generated 15-minute slot bases | Read-only `psql` query |
| `scheduler_hot_path_index_analysis.sql` | Scheduler maintenance indexes and query plans | Read-only diagnostics with `EXPLAIN ANALYZE` |
| `dag_chain_debug_view.sql` | Dependency state for every DAG task | Installs a view |
| `dag_gantt_dependency_status_view.sql` | Dependency execution timing and state | Installs a view |
| `dag_bucketed_sla_view.sql` | Current SLA buckets by DAG and queue | Installs a view |

## Safety

Use a read-only transaction and bounded execution time for ad hoc production
queries:

```sql
BEGIN TRANSACTION READ ONLY;
SET LOCAL statement_timeout = '60s';
SET LOCAL lock_timeout = '3s';

-- Run selected queries.

ROLLBACK;
```

Do not execute the complete monitoring directory as one migration. The
throughput file installs functions and the view files install views; the other
files are reports or diagnostic templates.

Scheduler payloads can contain document references and request metadata. Keep
exports in an access-controlled incident location.

## Submitted job lookup

Use `submission_lifecycle_analysis.sql` when the gateway logged `Job submitted
with id ...` but a lookup in `kv_store_worker_history` returned no rows.

Replace the UUID in the `target` CTE, then execute the statement in any SQL
client. It does not use `psql` variables or metacommands. The result combines:

- the current DAG and DAG history;
- every current task and scheduler state transition in that DAG;
- durable activation, dispatch, terminal, and recovery attempts; and
- current and historical worker status for every generated task ID.

The UUID returned by the gateway is the submission/DAG ID. A query planner may
also reuse it for one task, but worker KV keys are created from task job IDs,
not from the DAG ID in general. An empty worker-history query therefore does
not mean the submission is missing.

Interpret the first matching layer:

| Result | Meaning |
| --- | --- |
| `dag.current` or `job.current` | The submission was durably persisted. Inspect task state and attempts next. |
| `job.current` with `created` and no `attempt` | The task has not been admitted or activated. Check dependencies, `start_after`, scheduler leadership, and capacity. |
| `attempt` without `dispatch_started_at` | The task was activated but dispatch did not begin. |
| `attempt` with dispatch start but no confirmation | Inspect `dispatch_error`, gateway routing, and executor readiness. |
| Confirmed dispatch without worker records | Inspect the executor request path and worker status-store writes. |
| `worker.current` or `worker.history` | The executor observed the task; use its task `job_id` for worker-specific investigation. |
| `not_found` | No durable scheduler or worker layer contains the UUID in this database. |

`Job submitted with id ...` currently means the gateway placed the request in
the scheduler's in-memory submission queue. Persistence happens afterward in a
background worker. If the report returns `not_found`, first verify the database
and schema, then search the gateway log for the UUID and these events:

```text
scheduler_submission_enqueued
scheduler_submission_dequeued
dag_plan_built
dag_persist_start
dag_persisted
Job submission failed
Failed to process job
```

Also inspect `/api/debug` under `result.queue_status` for submission-worker
count, active workers, queue size, and pending requests. A restart after queue
acceptance but before persistence can discard an unprocessed request.

Use this report for incident lookup, not as an unbounded dashboard query. The
history portions can scan `job_history` and `kv_store_worker_history` on
installations that do not have lookup indexes for their ID and key columns.
Keep the read-only transaction timeout in place and review `EXPLAIN` before
automating it.

## Throughput

Execute `throughput_analysis.sql` once using any PostgreSQL client. It installs
three `STABLE` SQL functions; it does not depend on `psql` variables or
metacommands.

Call the functions with a bounded lookback in hours:

```sql
SELECT *
FROM marie_scheduler.monitor_system_throughput(24, NULL);

SELECT *
FROM marie_scheduler.monitor_planner_throughput(24, NULL);

SELECT *
FROM marie_scheduler.monitor_task_throughput(24, NULL);
```

Filter to one query planner when comparing a specific workflow:

```sql
SELECT *
FROM marie_scheduler.monitor_system_throughput(
    72,
    '<query-planner-name>'
);
```

The functions provide three levels of detail:

1. Overall and hourly system throughput.
2. Window-total and hourly throughput by query planner.
3. Window-total and hourly throughput by query-plan task and endpoint.

The measurement units are intentionally different:

| Metric | Database source | Meaning |
| --- | --- | --- |
| Plans submitted | `dag.created_on` | Query-plan runs accepted by the scheduler |
| Plans completed | Completed `dag.completed_on` | End-to-end successful query-plan runs |
| Tasks completed | Completed `job.completed_on` | Successful query-plan task nodes, including control nodes |
| Executor tasks completed | Completed `job.completed_on` excluding scheduler-local control endpoints | Successful work dispatched to executors |

`job.data.metadata.name` is the task name from the query plan.
`job.data.metadata.on` is the executor endpoint. `dag.planner` is the
authoritative query-plan name.

The current clock-hour row is marked `partial`. Compare completed clock hours
when looking for regressions. The `window_total` row reports rates normalized
to the elapsed duration of the selected window.

### Throughput limitations

The throughput report uses current retained `job` and `dag` rows. Therefore:

- It measures final task and plan outcomes, not every retry attempt.
- A reset that clears `completed_on` removes the earlier terminal event from
  the current-row report.
- Retention or archival can remove older rows from a long lookback.
- Skipped branch nodes are reported separately and are excluded from the task
  success-rate denominator.
- Scheduler-local `noop`, branch, switch, and control-merger nodes count as
  query-plan tasks but not executor-backed tasks.

Use `job_attempt` and `job_history` when attempt-level throughput or replay
behavior is required.

### Throughput query cost

Hourly completion filters use `completed_on`. Large installations should run
`EXPLAIN (ANALYZE, BUFFERS)` with a short lookback before scheduling this report
at a high frequency. The base schema prioritizes scheduler hot-path indexes;
it does not guarantee a completion-time index on every queue partition.

If the query scans large queue partitions, have the database team evaluate
per-partition `completed_on` indexes or a separate rollup table. Do not create
indexes during an incident without reviewing lock duration, disk space, write
amplification, and the actual query plan.

## Outstanding work

Run with all defaults:

```bash
psql -X -P pager=off \
  -f config/psql/schema/monitoring/outstanding_work_analysis.sql
```

Restrict the report to one queue and reduce detailed output:

```bash
psql -X -P pager=off \
  -v queue_name='<queue-name>' \
  -v lookback_minutes=60 \
  -v result_limit=100 \
  -f config/psql/schema/monitoring/outstanding_work_analysis.sql
```

The report distinguishes:

- work waiting for dependencies;
- work delayed by `start_after`;
- active work with missing or expired run leases;
- terminal worker KV state that disagrees with an active scheduler row; and
- work that is ready but waiting for executor capacity.

The totals operate over retained current rows. On a large installation, filter
by `queue_name` whenever the incident has a known queue.

## Failed job analysis

Set `marie_monitor.job_id` near the top of `job_failure_analysis.sql`, then run
the complete file:

```bash
psql -X -P pager=off \
  -f config/psql/schema/monitoring/job_failure_analysis.sql
```

The report correlates:

- the current scheduler job and worker KV rows;
- structured executor exceptions from worker history;
- scheduler state history;
- durable dispatch and terminal attempts; and
- run-owner and gateway fencing information.

Worker `message` can contain a generic failure description. Prefer the
structured `runtime_env_json.error` fields when present.

The base history tables may have only their primary-key indexes. On a large
installation, inspect indexes on `job_history(id, history_created_on)` and
`kv_store_worker_history(namespace, key, change_time)` before using this query
at dashboard frequency. Add such indexes only through the normal database
change process after measuring storage and write overhead.

## Creation-time and SLA analytics

`example_queries.sql` contains individually executable examples. Select only
the query needed for the investigation instead of running the whole file.

These queries use generated UTC columns from
`config/psql/schema/040_slots_columns.sql`:

```text
slot_idx15_created     day_local_created
slot_idx15_soft        day_local_soft
slot_idx15_hard        day_local_hard
slot_idx15_effective   day_local_effective
```

They group retained jobs by creation or SLA time. They do not represent actual
completion throughput or historical concurrency. Use
`throughput_analysis.sql` for those completion metrics.

Inspect the installed generated columns before using the examples:

```sql
SELECT
    column_name,
    data_type,
    is_generated,
    generation_expression
FROM information_schema.columns
WHERE table_schema = 'marie_scheduler'
  AND table_name = 'job'
  AND (
      column_name LIKE 'slot_idx15_%'
      OR column_name LIKE 'day_local_%'
  )
ORDER BY ordinal_position;
```

## Slot-basis report

`slots_basis.sql` requires a UTC date supplied as a `psql` variable:

```bash
psql -X -P pager=off \
  -v target_day='2026-01-15' \
  -f config/psql/schema/monitoring/slots_basis.sql
```

It includes the current outstanding states: `created`, `retry`, and `active`.
It does not create indexes or change scheduler state.

## Optional monitoring views

Install a view explicitly after the numbered scheduler schema migrations have
been applied:

```bash
psql -X -v ON_ERROR_STOP=1 \
  -f config/psql/schema/monitoring/dag_chain_debug_view.sql
```

The SLA view requires the `day_in_tz()` and `slot_15m()` helpers from
`040_slots_columns.sql`. Query installed views with a specific DAG or a bounded
result set:

```sql
SELECT *
FROM marie_scheduler.dag_chain_debug_view
WHERE dag_id = '<dag-uuid>'::uuid
ORDER BY job_level, job_id;
```

The dependency views and SLA view can touch many retained jobs when queried
without a DAG predicate. They are diagnostic views, not unbounded fleet-wide
dashboard queries.

## Scheduler hot-path analysis

`scheduler_hot_path_index_analysis.sql` expects:

- the `pg_cron` extension and `cron.job` table for schedule inspection; and
- the `pg_stat_statements` extension for accumulated statement statistics.

Its `EXPLAIN ANALYZE` statements execute bounded read-only scheduler scans. Run
them during a representative but controlled period. Use plain `EXPLAIN` when
execution cost itself is a concern.

Check optional dependencies first:

```sql
SELECT
    to_regclass('cron.job') AS cron_job,
    to_regclass('pg_stat_statements') AS stat_statements;
```

## Time zones

Scheduler timestamps are `timestamptz`. Generated `day_local_*` and
`slot_idx15_*` columns currently use UTC despite their historical names.
Throughput reporting explicitly creates its clock-hour buckets in UTC.

Convert timestamps only for display:

```sql
SELECT completed_on AT TIME ZONE '<display-time-zone>' AS completed_local
FROM marie_scheduler.job
WHERE id = '<job-uuid>'::uuid;
```

Keep filtering and bucketing in UTC unless a business report explicitly
requires civil-time buckets and has defined daylight-saving behavior.

## Validation checklist

Before promoting a monitoring query into a recurring dashboard:

1. Confirm every referenced table, column, function, and extension exists.
2. Run with a short lookback and a restrictive queue or planner filter.
3. Inspect `EXPLAIN (ANALYZE, BUFFERS)` on representative data.
4. Verify whether the metric counts plans, tasks, executor tasks, or attempts.
5. Confirm current-hour and retention behavior.
6. Record the expected time zone.
7. Set a dashboard timeout and maximum lookback.
