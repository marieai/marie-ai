# Scheduler High Availability Checks

This directory contains repeatable checks for validating Marie gateway and
scheduler high availability after an active-active gateway run.

The checks are focused on correctness:

- which gateway activated each job attempt
- which gateway accepted terminal events
- whether activated attempts have terminal audit records after the workload drains
- whether active jobs still have valid run attempt identity and run leases
- whether duplicate successful completions were accepted for the same job

Throughput is checked separately from scheduler traces with
`tools/stress/analyze_scheduler_trace.py`.

## Prerequisites

Apply the durable attempt schema before running these checks:

```bash
config/psql/schema/064_durable_attempts.sql
config/psql/schema/065_job_attempt.sql
```

Restart all gateways after applying the schema. New HA runs should populate
`marie_scheduler.job_attempt` with `gateway_instance_id`,
`scheduler_lease_owner`, activation identity, and terminal audit fields.

Install the optional invariant helper before running the shared HA checks or
`scheduler_correctness.py`:

```bash
psql "$DATABASE_URL" -v ON_ERROR_STOP=1 \
  -f config/psql/high-availability/scheduler_attempt_invariant_checks.sql
```

The helper creates one read-only diagnostic function. It is not part of the
scheduler schema version and gateway startup does not require it.

## Run The SQL Checks

Run after the workload has drained. Open
`config/psql/high-availability/ha_scheduler_checks.sql` in your database client
and run the whole file.

The script defaults to the last 24 hours. It reads optional current-session
settings so automation can scope the checked-in SQL without rewriting it:

```sql
CREATE TEMP TABLE ha_check_params AS
SELECT
    COALESCE(
        NULLIF(current_setting('marie.ha_run_start', TRUE), '')::timestamptz,
        now() - INTERVAL '24 hours'
    ) AS run_start,
    COALESCE(
        NULLIF(current_setting('marie.ha_run_end', TRUE), '')::timestamptz,
        now()
    ) AS run_end;
```

For an exact HA run, set a fixed UTC window in the same database session before
running the file:

```sql
SET marie.ha_run_start = '2026-05-19 09:45:00+00';
SET marie.ha_run_end = '2026-05-19 10:00:00+00';
```

Use an explicit window for real HA validation so historical repairs or old
stress runs do not pollute the result. After the optional helper is installed,
the check script creates only temporary tables and does not modify persistent
scheduler data.

From `psql`, the same plain SQL file can be run without variables:

```bash
psql "$DATABASE_URL" \
  -f config/psql/high-availability/ha_scheduler_checks.sql
```

## In-Flight Gateway Kill Checks

For the chaos test where a gateway is killed while it owns an active
long-running attempt, use:

```text
config/psql/high-availability/ha_inflight_gateway_kill_invariants.sql
```

Run it twice if useful:

- Before the kill, to identify active `run_owner`, `run_attempt_id`, and
  `gateway_instance_id`.
- After the kill and recovery window, to validate DB invariants.

The post-kill run should happen after:

```text
run_ttl_seconds + maintenance interval + buffer
```

The script defaults to the last 2 hours. For reviewable results, set
`marie.ha_run_start`, `marie.ha_run_end`, and optionally
`marie.ha_killed_gateway_instance_id` and
`marie.ha_killed_scheduler_lease_owner` in the current session.

Expected post-kill DB invariant results:

- `active_missing_attempt_identity = PASS`
- `expired_active_run_leases = PASS`
- `activated_missing_terminal_or_recovery = PASS`
- `duplicate_accepted_completed_terminal_by_job = PASS`
- `accepted_terminal_missing_terminal_gateway = PASS`
- `recovered_attempt_still_expired_active = PASS`

Terminal rejections are not automatically failures in this test. If an old
attempt reports after recovery, a rejected terminal event is expected as long as
the reject reason shows stale owner or stale attempt behavior.

Semaphore leak checks are not fully answerable from Postgres. Pair the DB
invariant script with an etcd/SemaphoreStore holder/count check for the executor
slot type used by the test.

## Lost Local Terminal Event Reconciliation

For the fault test where durable job-info storage reaches a terminal state but
the gateway-local terminal event is lost, use:

```text
config/psql/high-availability/ha_lost_terminal_event_reconciliation.sql
```

This test creates the mismatch directly:

- `marie_scheduler.job` remains `active`.
- `marie_scheduler.kv_store_worker` for the same job is changed to
  `status = SUCCEEDED`.
- The KV `end_time` is written as 10 minutes old so the scheduler sync path does
  not wait for the default 300 second terminal age guard.
- The script bypasses `JobInfoStorageClientProxy.put_status()`, so no
  gateway-local job event is published.

The file is guarded. By default, `marie.ha_enable_mutation` is unset and
therefore false. Mutation also requires an exact job ID:

```sql
SET marie.ha_target_job_id = '00000000-0000-0000-0000-000000000000';
SET marie.ha_enable_mutation = 'true';
```

With mutation disabled, it selects a candidate and shows the current DB/KV
state. Enabling mutation without `marie.ha_target_job_id` selects no row. After
the mutation, wait one or two scheduler sync cycles, usually 60-120 seconds,
then set `marie.ha_enable_mutation = 'false'` and rerun the file with the same
job ID.

Expected result:

- `job_state = completed`
- `terminal_source = storage_sync`
- `terminal_accepted = true`
- `terminal_gateway_instance_id` is not null
- `terminal_reject_reason` is null

If `terminal_source = job_event`, the normal terminal event path won and this
run did not prove lost-local-event reconciliation. If `terminal_source` stays
null, wait another sync cycle and inspect scheduler logs for `Syncing job` and
`State mismatch`.

## Capacity Holder Log Check

The gateway capacity summary is the quick operational check for semaphore
holders:

```text
[capacity] Slot summary:
SLOT             | CAPACITY | TARGET | USED | AVAIL | HOLDERS | NOTES
extract_executor | 10       | 10     | 1    | 9     | 1       |
```

During a long-running job, `USED` and `HOLDERS` should reflect the active
executor reservation. After the job completes, or after recovery has reclaimed a
failed run, the same slot should return to zero:

```text
extract_executor | 10       | 10     | 0    | 10    | 0       |
```

Read the reconcile summary with the slot summary:

```text
[sem] boot reconcile summary: {'extract_executor': {'before_count': 0,
'after_count': 0, 'deleted_orphans': 0, 'malformed_holders': 0}}
```

If the holder count reaches zero before the next reconcile pass, the normal
release path cleaned up the reservation. If `deleted_orphans` is greater than
zero, reconcile found stale etcd holders and removed them. Either can be valid
after fault injection, but the final post-drain state must be `USED=0` and
`HOLDERS=0` for the tested executor slot.

## Passing Criteria

For an active-active HA run, the important rows are in the `HA check summary`
section.

Expected pass conditions after drain:

- `active_active_gateway_count` passes when at least two gateways activated work.
- `activated_missing_terminal` is `0`.
- `active_missing_attempt_identity` is `0`.
- `expired_active_run_leases` is `0`.
- `duplicate_completed_terminal_by_job` is `0`.
- `activated_without_gateway_instance` is `0`.
- `terminal_rejected` is `0` for normal no-fault runs.

Some non-zero values can be expected in fault-injection runs. For example,
`terminal_rejected` can be valid when intentionally testing stale attempt
fencing. In that case, inspect `terminal_reject_reason` and confirm the
rejection matches the scenario.

## Interpreting Common Results

An attempt row is created atomically when a leased job becomes active. Always
use a scoped run window when comparing gateway ownership or terminal outcomes.

Rows with `terminal_source = job_state_backfill` are acceptable only for
historical repair or rolling-upgrade validation. Fresh HA runs should normally
show terminal events from `job_event`.

Uneven distribution across gateways is not a correctness failure. It can point
to load balancer stickiness, gateway startup timing, executor routing, or one
scheduler loop being more productive than another.

If `activated_missing_terminal` is non-zero after the workload drained, inspect
the detail section. Completed jobs without terminal audit are a bug in terminal
recording or a schema/version mismatch.

If `expired_active_run_leases` is non-zero, maintenance or recovery is not
claiming expired run leases, or a long-running executor is not extending its run
lease.

## Trace Checks

Use the trace analyzer to judge throughput and scheduling latency:

```bash
python tools/stress/analyze_scheduler_trace.py \
  ~/tmp/marie-scheduler-trace.jsonl \
  --report
```

For a healthy correctness run, expect:

- `gateway_dispatch_start` equals the submitted count after drain
- `executor_success_recorded + executor_failed_recorded` equals dispatches
- `executor_failed_recorded = 0` for normal runs

For throughput tuning, look at:

- `gateway->dispatch`
- `candidate->planned`
- `candidate snapshots before selection`
- `dispatch->confirm`

Large `candidate->planned` or repeated candidate snapshots indicate scheduler
selection pressure. That is a performance problem, not by itself an HA
correctness failure.

## Useful Manual Queries

Which gateway processed work:

```sql
SELECT
    gateway_instance_id,
    scheduler_lease_owner,
    COUNT(*) AS attempts,
    COUNT(*) FILTER (WHERE terminal_accepted IS TRUE) AS terminal_accepted,
    COUNT(*) FILTER (WHERE recovery_state IS NOT NULL) AS recovered
FROM marie_scheduler.job_attempt
GROUP BY gateway_instance_id, scheduler_lease_owner
ORDER BY attempts DESC;
```

Expired active run leases:

```sql
SELECT id, state, run_owner, run_attempt_id, run_lease_expires_at
FROM marie_scheduler.job
WHERE state::text = 'active'
  AND run_lease_expires_at < now();
```

Activated attempts missing terminal audit:

```sql
SELECT
    ja.job_id,
    ja.run_attempt_id,
    j.state::text AS job_state,
    ja.gateway_instance_id,
    ja.activated_at,
    ja.terminal_accepted,
    ja.terminal_reject_reason,
    ja.recovery_state
FROM marie_scheduler.job_attempt ja
JOIN marie_scheduler.job j ON j.id = ja.job_id
WHERE ja.terminal_accepted IS DISTINCT FROM TRUE
  AND COALESCE(ja.executor, '') NOT IN ('noop', 'branch', 'switch', 'merger')
  AND ja.recovery_state IS NULL
ORDER BY ja.activated_at DESC
LIMIT 50;
```
