---
sidebar_position: 10
---

# Scheduler SQL troubleshooting

Use this guide to investigate Marie scheduler state, executor failures, retries,
stuck DAGs, page selection, leases, and worker history in PostgreSQL. Start with
read-only queries. Use the recovery operations only after the active executor
state has been checked and the affected work has been drained or quarantined.

Scheduler JSON can contain document references, request metadata, and error
messages. Keep query results in an access-controlled incident location. Do not
paste full `data`, `output`, signed object URLs, or document content into tickets
or chat systems.

## Query safety

Use a read-only transaction and a statement timeout while investigating:

```sql
BEGIN TRANSACTION READ ONLY;
SET LOCAL statement_timeout = '30s';
SET LOCAL lock_timeout = '3s';

-- Run investigation queries here.

ROLLBACK;
```

The examples use ordinary SQL literals so they work in graphical SQL clients,
application consoles, and `psql`. Replace values such as `<job-uuid>` while
retaining the explicit type cast:

```sql
'<job-uuid>'::uuid
```

Avoid `SELECT *` during an incident. The `data`, `output`, serialized DAG, and
runtime-environment columns can be large.

## 1. Verify scheduler objects and indexes

Confirm that the core tables exist:

```sql
SELECT
    to_regclass('marie_scheduler.queue') AS queue_table,
    to_regclass('marie_scheduler.dag') AS dag_table,
    to_regclass('marie_scheduler.job') AS job_table,
    to_regclass('marie_scheduler.job_history') AS job_history_table,
    to_regclass('marie_scheduler.job_attempt') AS job_attempt_table,
    to_regclass('marie_scheduler.kv_store_worker') AS worker_state_table,
    to_regclass(
        'marie_scheduler.kv_store_worker_history'
    ) AS worker_history_table;
```

Inspect the available indexes before running broad history searches:

```sql
SELECT
    tablename,
    indexname,
    indexdef
FROM pg_indexes
WHERE schemaname = 'marie_scheduler'
  AND tablename IN (
      'job',
      'job_history',
      'job_attempt',
      'kv_store_worker',
      'kv_store_worker_history'
  )
ORDER BY tablename, indexname;
```

The base schema does not guarantee secondary indexes for every history lookup.
Use exact job keys where possible, keep fleet-wide lookbacks short, and inspect
the plan before running a broad production query:

```sql
EXPLAIN (COSTS, VERBOSE)
SELECT kh.change_time
FROM marie_scheduler.kv_store_worker_history kh
WHERE kh.namespace = 'job'
  AND kh.change_time >= now() - INTERVAL '24 hours';
```

Do not use `EXPLAIN ANALYZE` on state-changing statements.

## 2. Inspect queues and state totals

List configured queues and retry policies:

```sql
SELECT
    name,
    policy,
    retry_limit,
    retry_delay,
    retry_backoff,
    expire_seconds,
    retention_minutes,
    dead_letter,
    updated_on
FROM marie_scheduler.queue
ORDER BY name;
```

Summarize current jobs by queue and state:

```sql
SELECT
    name AS queue_name,
    state::text AS state,
    count(*) AS jobs,
    min(created_on) AS oldest_created_on,
    max(created_on) AS newest_created_on
FROM marie_scheduler.job
GROUP BY name, state
ORDER BY name, state;
```

Summarize DAG state:

```sql
SELECT
    state,
    count(*) AS dags,
    min(created_on) AS oldest_created_on,
    max(created_on) AS newest_created_on
FROM marie_scheduler.dag
GROUP BY state
ORDER BY state;
```

Count active DAGs:

```sql
SELECT count(*) AS active_dags
FROM marie_scheduler.dag
WHERE state = 'active';
```

### Measure hourly throughput

Execute `config/psql/schema/monitoring/throughput_analysis.sql` once through
your SQL client to install the read-only monitoring functions. Then run the
level of detail needed for the investigation:

```sql
SELECT *
FROM marie_scheduler.monitor_system_throughput(24, NULL);

SELECT *
FROM marie_scheduler.monitor_planner_throughput(24, NULL);

SELECT *
FROM marie_scheduler.monitor_task_throughput(24, NULL);
```

The first argument is the lookback in hours. The second is an optional query
planner name. For example:

```sql
SELECT *
FROM marie_scheduler.monitor_system_throughput(
    72,
    '<query-planner-name>'
);
```

The report keeps the units separate:

- completed DAGs are end-to-end query-plan throughput;
- completed jobs are query-plan task throughput; and
- completed executor-backed jobs exclude scheduler-local control nodes.

It returns window totals, hourly planner totals, and window/hourly breakdowns
by query-plan task and executor endpoint. The current hour is marked as
partial. See `config/psql/schema/monitoring/QUERY_GUIDE.md` for retention,
index, and retry limitations.

## 3. Locate a submission or job

The UUID in the gateway message `Job submitted with id ...` is the
submission/DAG ID. Depending on the query plan, one generated task may reuse
the same UUID, but other task jobs have their own IDs. Worker status is keyed
by task job ID, so searching worker history with only the DAG ID can correctly
return no rows.

For a one-pass timeline across the DAG, tasks, attempts, and worker status, run
`config/psql/schema/monitoring/submission_lifecycle_analysis.sql`. Replace the
single UUID in its `target` CTE. The statement works in ordinary SQL clients
and does not require `psql` variables.

First establish whether the submitted UUID is a current DAG ID, a task job ID,
or both:

```sql
WITH target(id) AS (
    VALUES ('<submission-or-job-uuid>'::uuid)
)
SELECT
    'dag' AS record_type,
    d.id AS submission_id,
    NULL::uuid AS job_id,
    d.state::text AS state,
    d.created_on
FROM marie_scheduler.dag d
JOIN target t ON t.id = d.id

UNION ALL

SELECT
    'job',
    j.dag_id,
    j.id,
    j.state::text,
    j.created_on
FROM marie_scheduler.job j
JOIN target t
  ON t.id = j.id
  OR t.id = j.dag_id
ORDER BY created_on, record_type;
```

If a task exists but `kv_store_worker_history` is empty, the scheduler has
persisted the task but an executor has not necessarily observed it. Inspect
`state`, `run_attempt_id`, and the `job_attempt` rows before looking for an
executor error.

The gateway currently acknowledges a submission after placing it in an
in-memory scheduler queue; background workers persist the DAG and task rows.
If neither `dag`, `job`, their history tables, nor `job_attempt` contains the
UUID, verify that the SQL client is connected to the correct database and
search the gateway log for submission dequeue, persistence, or background
submission errors.

Find a job by scheduler job ID:

```sql
SELECT
    j.id,
    j.dag_id,
    j.name AS queue_name,
    j.state::text AS state,
    j.priority,
    j.retry_count,
    j.retry_limit,
    j.created_on,
    j.started_on,
    j.completed_on,
    j.start_after,
    j.run_owner,
    j.run_attempt_id,
    j.run_lease_expires_at,
    j.data #>> '{metadata,on}' AS endpoint,
    j.data #>> '{metadata,ref_id}' AS ref_id,
    j.data #>> '{metadata,ref_type}' AS ref_type,
    j.output
FROM marie_scheduler.job j
WHERE j.id = '<job-uuid>'::uuid;
```

Find recent jobs by document reference without returning the complete payload:

```sql
SELECT
    j.id,
    j.dag_id,
    j.name AS queue_name,
    j.state::text AS state,
    j.retry_count,
    j.run_attempt_id,
    j.created_on,
    j.started_on,
    j.completed_on,
    j.data #>> '{metadata,on}' AS endpoint,
    j.data #>> '{metadata,ref_type}' AS ref_type
FROM marie_scheduler.job j
WHERE j.data #>> '{metadata,ref_id}' = '<document-reference>'
ORDER BY j.created_on DESC
LIMIT 100;
```

Find recent jobs for one endpoint and queue:

```sql
SELECT
    j.id,
    j.dag_id,
    j.state::text AS state,
    j.retry_count,
    j.created_on,
    j.started_on,
    j.completed_on,
    j.data #>> '{metadata,ref_id}' AS ref_id
FROM marie_scheduler.job j
WHERE j.name = '<queue-name>'
  AND j.data #>> '{metadata,on}' = '<executor-endpoint>'
  AND j.created_on >= now() - INTERVAL '24 hours'
ORDER BY j.created_on DESC
LIMIT 200;
```

## 4. Determine where a page selection came from

This query shows both page sources stored in the job payload:

```sql
SELECT
    j.id,
    j.dag_id,
    j.name AS queue_name,
    j.state::text AS scheduler_state,
    j.retry_count,
    j.retry_limit,
    j.data #>> '{metadata,on}' AS endpoint,
    j.data #>> '{metadata,ref_id}' AS ref_id,
    j.data #>> '{metadata,ref_type}' AS ref_type,
    j.data #> '{metadata,pages}' AS top_level_pages,
    j.data #> '{metadata,args,pages}' AS args_pages,
    jsonb_path_query_array(
        COALESCE(
            j.data #> '{metadata,features}',
            '[]'::jsonb
        ),
        '$[*] ? (@.type == "pipeline")'
    ) AS pipeline_features,
    CASE
        WHEN (j.data->'metadata') ? 'uri' THEN 'uri'
        WHEN (j.data->'metadata') ? 'srcUrl' THEN 'srcUrl'
        WHEN (j.data->'metadata') ? 'srcFile' THEN 'srcFile'
        WHEN (j.data->'metadata') ? 'data' THEN 'inline-data'
        WHEN (j.data->'metadata') ? 'srcData' THEN 'inline-data'
        WHEN (j.data->'metadata') ? 'srcBase64' THEN 'inline-base64'
        ELSE 'unknown'
    END AS source_kind
FROM marie_scheduler.job j
WHERE j.id = '<job-uuid>'::uuid;
```

Page indexes are zero-based: page `2` means the third page. In the document LLM
pipeline, pages in the matching pipeline feature override the top-level asset
pages. An empty or absent page selection means all pages.

## 5. Inspect current worker status

Read the current worker KV record for one job:

```sql
WITH current_worker AS (
    SELECT
        kv.updated_at,
        kv.value,
        COALESCE(
            NULLIF(kv.value->>'runtime_env_json', '')::jsonb,
            '{}'::jsonb
        ) AS runtime_env
    FROM marie_scheduler.kv_store_worker kv
    WHERE kv.namespace = 'job'
      AND kv.key = 'marie_internal/job_info_' || '<job-uuid>'
      AND kv.is_deleted = false
)
SELECT
    updated_at,
    value->>'status' AS status,
    value->>'message' AS worker_message,
    value->>'run_owner' AS run_owner,
    value->>'run_attempt_id' AS run_attempt_id,
    runtime_env #>> '{attributes,executor}' AS executor,
    runtime_env #>> '{attributes,runtime_name}' AS runtime_name,
    runtime_env #>> '{attributes,host}' AS executor_host,
    runtime_env #>> '{attributes,executor_endpoint}' AS endpoint,
    runtime_env #>> '{error,type}' AS error_type,
    runtime_env #>> '{error,message}' AS error_message
FROM current_worker;
```

The current KV row is only the latest snapshot. Use the history table to see
earlier replicas and state transitions.

## 6. Inspect worker history and structured errors

Show the complete worker lifecycle for one job:

```sql
SELECT
    kh.change_time,
    kh.operation,
    kh.value->>'status' AS status,
    kh.value->>'message' AS worker_message,
    kh.value->>'run_owner' AS run_owner,
    kh.value->>'run_attempt_id' AS run_attempt_id,
    env #>> '{attributes,executor}' AS executor,
    env #>> '{attributes,runtime_name}' AS runtime_name,
    env #>> '{attributes,host}' AS executor_host,
    env #>> '{attributes,executor_endpoint}' AS endpoint,
    env #>> '{error,type}' AS error_type,
    env #>> '{error,message}' AS error_message,
    env #>> '{error,filename}' AS error_file,
    env #>> '{error,name}' AS error_function,
    env #>> '{error,line_no}' AS error_line
FROM marie_scheduler.kv_store_worker_history kh
CROSS JOIN LATERAL (
    SELECT COALESCE(
        NULLIF(kh.value->>'runtime_env_json', '')::jsonb,
        '{}'::jsonb
    ) AS env
) parsed
WHERE kh.namespace = 'job'
  AND kh.key = 'marie_internal/job_info_' || '<job-uuid>'
ORDER BY kh.change_time;
```

`worker_message` can be the generic `Job failed.` message. The structured
exception is stored under `runtime_env_json.error`; use `error_type`,
`error_message`, `error_file`, `error_function`, and `error_line` for diagnosis.

## 7. Inspect every durable executor attempt

```sql
SELECT
    run_attempt_id,
    attempt_state,
    executor,
    run_owner,
    gateway_instance_id,
    activated_at,
    dispatch_started_at,
    dispatch_confirmed_at,
    dispatch_error,
    terminal_at,
    terminal_status,
    terminal_work_state,
    terminal_source,
    terminal_accepted,
    terminal_reject_reason,
    recovery_at,
    recovery_state,
    recovery_reason
FROM marie_scheduler.job_attempt
WHERE job_id = '<job-uuid>'::uuid
ORDER BY activated_at;
```

Interpret common attempt patterns:

| Pattern | Likely meaning |
| --- | --- |
| Activated, dispatched, then terminally completed | Normal attempt. |
| `dispatch_error` without confirmation | Gateway could not confirm dispatch. |
| `RUN_LEASE_EXPIRED` recovery | Worker disappeared or stopped renewing the run lease. |
| Multiple attempts with the same deterministic error | Input or code failure is being retried. |
| `terminal_accepted = false` | A stale or mismatched terminal event was fenced out. |
| Different gateway IDs across attempts | Work moved during retry or gateway failover. |

## 8. Inspect scheduler retry history

```sql
SELECT
    history_created_on,
    state::text AS state,
    retry_count,
    retry_limit,
    run_attempt_id,
    start_after,
    started_on,
    completed_on,
    output
FROM marie_scheduler.job_history
WHERE id = '<job-uuid>'::uuid
ORDER BY history_created_on;
```

The history trigger records meaningful state changes. Lease-only refreshes are
not expected to create history rows.

Compare the current job, attempt audit, scheduler history, and worker history:

```text
job               current authoritative scheduler row
job_history       scheduler state changes and retry progression
job_attempt       durable dispatch, terminal, fencing, and recovery audit
kv_store_worker   latest executor-reported state
worker history    every retained executor state snapshot
```

## 9. Find occurrences of an error

Find recent failures by exception type and function:

```sql
WITH failures AS (
    SELECT
        kh.change_time,
        replace(
            kh.key,
            'marie_internal/job_info_',
            ''
        ) AS job_id,
        kh.value->>'run_attempt_id' AS run_attempt_id,
        env #>> '{attributes,host}' AS executor_host,
        env #>> '{attributes,runtime_name}' AS runtime_name,
        env #>> '{error,type}' AS error_type,
        env #>> '{error,message}' AS error_message,
        env #>> '{error,filename}' AS error_file,
        env #>> '{error,name}' AS error_function,
        env #>> '{error,line_no}' AS error_line
    FROM marie_scheduler.kv_store_worker_history kh
    CROSS JOIN LATERAL (
        SELECT COALESCE(
            NULLIF(kh.value->>'runtime_env_json', '')::jsonb,
            '{}'::jsonb
        ) AS env
    ) parsed
    WHERE kh.namespace = 'job'
      AND kh.value->>'status' = 'FAILED'
      AND kh.change_time >= now() - INTERVAL '7 days'
)
SELECT *
FROM failures
WHERE error_type = '<exception-type>'
  AND error_function = '<function-name>'
ORDER BY change_time DESC
LIMIT 500;
```

Find failures containing part of an error message:

```sql
WITH failures AS (
    SELECT
        kh.change_time,
        replace(kh.key, 'marie_internal/job_info_', '') AS job_id,
        env #>> '{attributes,runtime_name}' AS runtime_name,
        env #>> '{attributes,host}' AS executor_host,
        env #>> '{error,type}' AS error_type,
        env #>> '{error,message}' AS error_message
    FROM marie_scheduler.kv_store_worker_history kh
    CROSS JOIN LATERAL (
        SELECT COALESCE(
            NULLIF(kh.value->>'runtime_env_json', '')::jsonb,
            '{}'::jsonb
        ) AS env
    ) parsed
    WHERE kh.namespace = 'job'
      AND kh.value->>'status' = 'FAILED'
      AND kh.change_time >= now() - INTERVAL '7 days'
)
SELECT *
FROM failures
WHERE error_message ILIKE '%' || '<message-fragment>' || '%'
ORDER BY change_time DESC
LIMIT 500;
```

Summarize failures by host, runtime, exception, and message:

```sql
WITH failures AS (
    SELECT
        replace(kh.key, 'marie_internal/job_info_', '') AS job_id,
        env #>> '{attributes,host}' AS executor_host,
        env #>> '{attributes,runtime_name}' AS runtime_name,
        env #>> '{error,type}' AS error_type,
        env #>> '{error,message}' AS error_message
    FROM marie_scheduler.kv_store_worker_history kh
    CROSS JOIN LATERAL (
        SELECT COALESCE(
            NULLIF(kh.value->>'runtime_env_json', '')::jsonb,
            '{}'::jsonb
        ) AS env
    ) parsed
    WHERE kh.namespace = 'job'
      AND kh.value->>'status' = 'FAILED'
      AND kh.change_time >= now() - INTERVAL '7 days'
)
SELECT
    executor_host,
    runtime_name,
    error_type,
    error_message,
    count(*) AS failure_records,
    count(DISTINCT job_id) AS distinct_jobs
FROM failures
GROUP BY executor_host, runtime_name, error_type, error_message
ORDER BY failure_records DESC
LIMIT 200;
```

History can contain more than one failed snapshot for a single attempt. Report
both `failure_records` and `distinct_jobs`; do not interpret row count alone as
the number of failed requests.

### Page-selection failure workflow

For an error such as `Page index out of range`, use the queries above in this
order:

1. Run **Determine where a page selection came from** for the failed job. Check
   `top_level_pages`, `args_pages`, and the matching pipeline feature. Remember
   that page indexes are zero-based.
2. Run **Inspect every durable executor attempt**. Multiple attempts on
   different runtimes with the same exception indicate a deterministic input
   or code failure rather than one unhealthy executor.
3. Run **Inspect scheduler retry history**. Confirm how `retry_count`, state,
   and `run_attempt_id` changed between attempts.
4. Run **Find occurrences of an error** with `IndexError` and `_select_frames`,
   then use the grouped query to measure distinct affected jobs and runtimes.

Keep the job ID and time window bounded before broadening the final history
search across the fleet.

## 10. Find stuck or expired active work

List the longest-running active jobs:

```sql
SELECT
    id,
    dag_id,
    name AS queue_name,
    started_on,
    now() - started_on AS active_for,
    retry_count,
    run_owner,
    run_attempt_id,
    run_lease_expires_at,
    data #>> '{metadata,on}' AS endpoint,
    data #>> '{metadata,ref_id}' AS ref_id
FROM marie_scheduler.job
WHERE state = 'active'
ORDER BY started_on NULLS FIRST
LIMIT 200;
```

Find active jobs whose run lease is missing or expired:

```sql
SELECT
    id,
    dag_id,
    name AS queue_name,
    started_on,
    run_owner,
    run_attempt_id,
    run_lease_expires_at,
    now() - run_lease_expires_at AS lease_overdue_by
FROM marie_scheduler.job
WHERE state = 'active'
  AND (
      run_lease_expires_at IS NULL
      OR run_lease_expires_at <= now()
  )
ORDER BY run_lease_expires_at NULLS FIRST;
```

Find active jobs without a matching current attempt audit row:

```sql
SELECT
    j.id,
    j.dag_id,
    j.name AS queue_name,
    j.started_on,
    j.run_owner,
    j.run_attempt_id
FROM marie_scheduler.job j
LEFT JOIN marie_scheduler.job_attempt ja
  ON ja.run_attempt_id = j.run_attempt_id
WHERE j.state = 'active'
  AND (
      j.run_attempt_id IS NULL
      OR ja.run_attempt_id IS NULL
  )
ORDER BY j.started_on NULLS FIRST;
```

These queries are observations. The scheduler may recover an expired lease
between the query and any follow-up action.

## 11. Diagnose a DAG that is not progressing

Show the DAG and job-state totals:

```sql
SELECT
    d.id,
    d.name,
    d.state,
    d.planner,
    d.created_on,
    d.started_on,
    d.completed_on,
    count(j.id) AS total_jobs,
    count(j.id) FILTER (
        WHERE j.state IN ('created', 'retry')
    ) AS schedulable_jobs,
    count(j.id) FILTER (WHERE j.state = 'active') AS active_jobs,
    count(j.id) FILTER (WHERE j.state = 'failed') AS failed_jobs,
    count(j.id) FILTER (WHERE j.state = 'completed') AS completed_jobs
FROM marie_scheduler.dag d
LEFT JOIN marie_scheduler.job j ON j.dag_id = d.id
WHERE d.id = '<dag-uuid>'::uuid
GROUP BY d.id;
```

List every job in the DAG without returning the full payload:

```sql
SELECT
    id,
    name AS queue_name,
    job_level,
    state::text AS state,
    retry_count,
    retry_limit,
    start_after,
    started_on,
    completed_on,
    run_attempt_id,
    dependencies,
    data #>> '{metadata,on}' AS endpoint,
    pg_column_size(output) AS output_bytes
FROM marie_scheduler.job
WHERE dag_id = '<dag-uuid>'::uuid
ORDER BY job_level, created_on, id;
```

Show terminal blockers and their latest attempt:

```sql
SELECT
    j.id,
    j.name AS queue_name,
    j.state::text AS state,
    j.retry_count,
    j.retry_limit,
    j.completed_on,
    pg_column_size(j.output) AS output_bytes,
    attempt.run_attempt_id,
    attempt.attempt_state,
    attempt.dispatch_error,
    attempt.terminal_source,
    attempt.terminal_work_state,
    attempt.recovery_reason,
    attempt.terminal_reject_reason
FROM marie_scheduler.job j
LEFT JOIN LATERAL (
    SELECT
        ja.run_attempt_id,
        ja.attempt_state,
        ja.dispatch_error,
        ja.terminal_source,
        ja.terminal_work_state,
        ja.recovery_reason,
        ja.terminal_reject_reason
    FROM marie_scheduler.job_attempt ja
    WHERE ja.job_id = j.id
    ORDER BY ja.updated_on DESC
    LIMIT 1
) attempt ON true
WHERE j.dag_id = '<dag-uuid>'::uuid
  AND j.state::text IN ('failed', 'expired', 'cancelled')
ORDER BY j.completed_on NULLS LAST, j.id;
```

## 12. Check history-table size

Large history tables can make incident queries expensive:

```sql
SELECT
    relname AS table_name,
    pg_size_pretty(
        pg_total_relation_size(
            format('%I.%I', schemaname, relname)::regclass
        )
    ) AS total_size,
    n_live_tup,
    n_dead_tup,
    last_autovacuum,
    last_autoanalyze
FROM pg_stat_user_tables
WHERE schemaname = 'marie_scheduler'
  AND relname IN (
      'job_history',
      'dag_history',
      'job_attempt',
      'kv_store_worker_history'
  )
ORDER BY pg_total_relation_size(
    format('%I.%I', schemaname, relname)::regclass
) DESC;
```

Do not create or drop an index during an active incident without DBA review.
Capture the query plan, table size, row count, and existing indexes first.

## State-changing recovery operations

The following operations are not troubleshooting queries. They modify durable
scheduler state and can cause work to execute again.

Before resetting anything:

1. Capture the affected job IDs, DAG IDs, run-attempt IDs, and current outputs.
2. Confirm the corresponding executor processes are no longer running.
3. Drain or quarantine dispatch for the affected queues.
4. Confirm no other gateway is actively scheduling the same work.
5. Obtain production change approval.

### Preview an active queue reset

This read-only query mirrors the scope of
`reset_active_dags_and_jobs()` without changing state:

```sql
WITH requested_queues AS (
    SELECT ARRAY['<queue-name>']::text[] AS names
), affected_jobs AS (
    SELECT j.id, j.dag_id, j.name, j.run_owner, j.run_attempt_id
    FROM marie_scheduler.job j
    CROSS JOIN requested_queues q
    WHERE j.state = 'active'
      AND j.name = ANY(q.names)
), affected_dags AS (
    SELECT DISTINCT d.id
    FROM marie_scheduler.dag d
    CROSS JOIN requested_queues q
    WHERE d.state = 'active'
      AND EXISTS (
          SELECT 1
          FROM marie_scheduler.job j
          WHERE j.dag_id = d.id
            AND j.name = ANY(q.names)
      )
)
SELECT
    (SELECT count(*) FROM affected_jobs) AS jobs_to_reset,
    (SELECT count(*) FROM affected_dags) AS dags_to_reset,
    (SELECT array_agg(id ORDER BY id) FROM affected_jobs) AS job_ids,
    (SELECT array_agg(id ORDER BY id) FROM affected_dags) AS dag_ids;
```

The function resets active jobs in the named queues to `created`, clears their
lease and run ownership, and resets associated active DAGs to `created`. It does
not prove that the old executor stopped. Resetting a live job can produce
duplicate execution.

### Reset active DAGs and jobs

Run only after the preview and operational checks:

```sql
BEGIN;
SET LOCAL statement_timeout = '30s';
SET LOCAL lock_timeout = '3s';

SELECT marie_scheduler.reset_active_dags_and_jobs(
    ARRAY['<queue-name>']::text[]
);

SELECT
    state::text,
    count(*)
FROM marie_scheduler.job
WHERE name = '<queue-name>'
GROUP BY state
ORDER BY state;

COMMIT;
```

For multiple queues:

```sql
SELECT marie_scheduler.reset_active_dags_and_jobs(
    ARRAY['<queue-a>', '<queue-b>']::text[]
);
```

Do not substitute ad hoc `UPDATE` statements for the scheduler function. Direct
updates can leave leases, run ownership, attempt audit, DAG state, and in-memory
frontier state inconsistent.

## Investigation checklist

For a single failed or stuck job, capture:

```text
Current job state and retry counters
DAG state and blocking jobs
Requested endpoint and page selection
Current worker status
Worker history and structured exception
Every durable run attempt
Scheduler job history
Lease owner and expiration
Whether the same error appears on other hosts or runtimes
Whether recovery changed state during the investigation
```

For executor or host memory incidents, continue with the
[production executor OOM troubleshooting runbook](./executor-oom-troubleshooting.md).
