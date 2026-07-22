# Scheduler Reliability Runner

Use `scheduler_reliability_runner.py` for stateful faults that coordinate a
workload, an exact mutation target, recovery, and authoritative verification.
Keep rate sweeps in `run_gateway_benchmark_matrix.sh`.

The runner executes one selected scenario at a time. A scenario passes only
when the injection is observed, recovery is observed before its deadline, the
workload settles, and every mandatory correctness check passes. Disabled,
unavailable, and dry-run scenarios report `skipped`; they never report a pass.

## Configure A Scenario

Copy `scheduler-reliability.config.example.json` and replace the operator helper
paths. Keep credentials out of the JSON. Helper commands inherit the secured
process environment and must emit one JSON object on stdout.

Probe output uses this shape:

```json
{
  "status": "pass",
  "observed": {},
  "artifacts": [
    {"name": "gateway-debug", "path": "/tmp/run-gateway-debug.json"}
  ],
  "query_counters": {"get_job_by_id": 0, "postgresql_statements": 14}
}
```

Valid probe states are `pass`, `fail`, `unavailable`, and `error`. A probe may
instead emit `passed` or `ready` as a boolean. Injection and recovery commands
may emit `effect_applied: false`; the runner treats that as a failed injection.
Commands run as exact argument arrays without a shell.

For active-active scenarios, configure a diagnostics probe for every gateway
endpoint. The report retains pre-fault and post-recovery query-counter snapshots
and deltas per endpoint. It does not sum process-local counters into a false
cluster total.

For the two-scheduler lease-shortfall scenario, configure `query_budget.probe`
to return:

```json
{
  "status": "pass",
  "observed": {
    "missing_leases": 4,
    "counters": {"get_job_by_id": 0, "postgresql_statements": 5}
  }
}
```

Set explicit `max_per_missing_lease` limits for both counters. Exceeding either
limit fails the scenario even when latency and process liveness look healthy.

## Inspect Before Mutation

List the configured scenarios:

```bash
python tools/stress/scheduler_reliability_runner.py \
  --config tools/stress/scheduler-reliability.config.example.json \
  --list
```

Render the selected contract without running any command:

```bash
python tools/stress/scheduler_reliability_runner.py \
  --config /path/to/scheduler-reliability.config.json \
  --scenario gateway-owner-kill-example \
  --dry-run \
  --report /tmp/gateway-owner-kill-dry-run.json
```

Enable a real mutation only after confirming the printed target. Both the
opt-in flag and an exact target match are required:

```bash
python tools/stress/scheduler_reliability_runner.py \
  --config /path/to/scheduler-reliability.config.json \
  --scenario gateway-owner-kill \
  --allow-mutation \
  --confirm-target marie-gateway-1 \
  --report /tmp/gateway-owner-kill.json
```

The injection and recovery command arrays must each contain the exact target as
one argument. Shell commands, broad process kills, container prune operations,
database resets, schema drops, and truncation are rejected.

## Supported Fault Matrix

Each built-in fault has trigger, injection, recovery, and settle deadlines.
Scenario-specific deadlines may be shorter but cannot exceed them.

| Fault | Trigger / inject / recover / settle | Required recovery outcome |
| --- | --- | --- |
| `gateway-owner-kill` | 120s / 15s / 180s / 300s | Accepted work completes once, fails durably within retry policy, receives an accepted recovery audit, or remains intentionally pending with a documented reason; stale terminals remain fenced. |
| `gateway-schedulers-restart` | 120s / 60s / 180s / 300s | The same durable outcome set; no accepted work disappears during the all-scheduler restart. |
| `scheduler-run-lease-expiry` | 120s / 15s / 240s / 360s | The expired attempt receives accepted recovery audit or durable failure and retains no terminal lease. |
| `lost-local-terminal-event` | 120s / 15s / 180s / 300s | Exactly one completion is accepted through `storage_sync`. |
| `executor-kill-before-dispatch-confirmation` | 120s / 15s / 180s / 300s | The unconfirmed attempt releases its lease and executor capacity. |
| `executor-kill-after-dispatch-confirmation` | 120s / 15s / 240s / 360s | The confirmed attempt reaches one durable terminal or accepted recovery outcome. |
| `malformed-executor-event` | 120s / 15s / 60s / 180s | The malformed event is rejected without changing current attempt state. |
| `duplicate-executor-event` | 120s / 15s / 60s / 180s | The duplicate terminal is rejected or ignored idempotently. |
| `delayed-executor-event` | 120s / 15s / 180s / 300s | A still-current event is accepted; a superseded event is rejected. |
| `stale-executor-event` | 120s / 15s / 60s / 180s | The stale event cannot mutate the current attempt. |
| `partial-lease-jobs-by-id` | 120s / 15s / 120s / 240s | Missing leases release database leases, local reservations, and capacity. |
| `activation-failure-after-reservation` | 120s / 15s / 120s / 240s | Activation failure releases the lease, semaphore holder, and capacity. |
| `two-scheduler-lease-shortfall` | 180s / 15s / 120s / 300s | Shortfalls release cleanly and stay within the point-lookup and statement budgets. |
| `postgres-submission-outage` | 120s / 30s / 180s / 300s | Submission is rejected clearly or every accepted item recovers durably. |
| `postgres-dispatch-outage` | 120s / 30s / 180s / 300s | Every accepted item reaches one durable terminal or accepted recovery outcome. |
| `postgres-pool-exhaustion` | 120s / 30s / 180s / 300s | Failure is bounded or recovery completes without losing accepted work. |
| `postgres-latency` | 120s / 30s / 180s / 300s | Deadline failure or recovery occurs without duplicate accepted terminals. |
| `postgres-statement-timeout` | 120s / 30s / 180s / 300s | Timeout recovery leaves no database lease or executor-capacity leak. |
| `rabbitmq-pause` | 120s / 30s / 180s / 300s | Durable terminal state reconciles after the broker returns. |
| `rabbitmq-consumer-disconnect` | 120s / 30s / 180s / 300s | The consumer reconnects and durable terminal state reconciles. |

ETCD discovery and capacity faults remain owned by Slice 05. Reference their
timeline artifact from a reliability scenario instead of adding another ETCD
injector.

## Required Privileges

Grant only the control needed by the selected target:

| Fault group | Operator capability |
| --- | --- |
| Gateway or executor process | Stop and restore one exact container, PID, or supervised service. Do not grant broad host process control to a helper. |
| Scheduler lease and lost-event SQL | Connect to the test database; read scheduler audit tables; update only the exact selected scheduler/KV rows for mutating scenarios. |
| Executor event faults | Publish test events to the selected test queue or call the dedicated test injector. Do not reuse production routing credentials. |
| PostgreSQL service faults | Pause/restart one test service or enable one exact proxy toxic. Pool exhaustion must use a bounded test client count. |
| RabbitMQ faults | Pause/restart one test broker or disconnect one named test consumer through its supervisor or management API. |
| Read-only verification | Read the run manifest, scheduler job/attempt state, PostgreSQL statistics, gateway debug output, and capacity state. |

The final report contains the sanitized configuration hash, all declared
sub-artifact references, fault timestamps, per-gateway snapshots, and the
mandatory verifier results. Helper output is sanitized again before it enters
the report, but helpers must not print credentials in the first place.

## Parameterize The HA SQL

The three HA scripts read optional PostgreSQL session settings. This lets the
runner or an operator reuse the checked-in SQL without editing it.

```sql
SET marie.ha_run_start = '2026-07-21 18:00:00+00';
SET marie.ha_run_end = '2026-07-21 18:15:00+00';
SET marie.ha_killed_gateway_instance_id = 'gateway-instance-1';
SET marie.ha_killed_scheduler_lease_owner = 'scheduler-owner-1';
```

For lost-local-terminal-event mutation, the exact job ID and explicit boolean
are both required. With mutation unset or false, the script is read-only.

```sql
SET marie.ha_target_job_id = '00000000-0000-0000-0000-000000000000';
SET marie.ha_enable_mutation = 'true';
SET marie.ha_extend_run_lease_by = '5 minutes';
SET marie.ha_synthetic_terminal_age = '10 minutes';
```

Run the settings and file in one `psql` session. Standard libpq environment
configuration supplies the database connection and credentials.

```bash
psql -X --set ON_ERROR_STOP=1 \
  -c "SET marie.ha_run_start = '2026-07-21 18:00:00+00'" \
  -c "SET marie.ha_run_end = '2026-07-21 18:15:00+00'" \
  -f config/psql/high-availability/ha_scheduler_checks.sql
```

## External-Service Qualification Commands

Keep real service mutation opt-in and outside ordinary unit tests. Create one
enabled scenario per exact target, then run the matching command:

```bash
# PostgreSQL during submission, dispatch, pool pressure, latency, or timeout
python tools/stress/scheduler_reliability_runner.py --config /path/to/reliability.json --scenario postgres-submission-outage --allow-mutation --confirm-target postgres-test --report /tmp/postgres-submission.json
python tools/stress/scheduler_reliability_runner.py --config /path/to/reliability.json --scenario postgres-dispatch-outage --allow-mutation --confirm-target postgres-test --report /tmp/postgres-dispatch.json
python tools/stress/scheduler_reliability_runner.py --config /path/to/reliability.json --scenario postgres-pool-exhaustion --allow-mutation --confirm-target postgres-test-pool --report /tmp/postgres-pool.json
python tools/stress/scheduler_reliability_runner.py --config /path/to/reliability.json --scenario postgres-latency --allow-mutation --confirm-target postgres-test-proxy --report /tmp/postgres-latency.json
python tools/stress/scheduler_reliability_runner.py --config /path/to/reliability.json --scenario postgres-statement-timeout --allow-mutation --confirm-target postgres-test-proxy --report /tmp/postgres-timeout.json

# RabbitMQ service or one named consumer
python tools/stress/scheduler_reliability_runner.py --config /path/to/reliability.json --scenario rabbitmq-pause --allow-mutation --confirm-target rabbitmq-test --report /tmp/rabbitmq-pause.json
python tools/stress/scheduler_reliability_runner.py --config /path/to/reliability.json --scenario rabbitmq-consumer-disconnect --allow-mutation --confirm-target scheduler-event-consumer-1 --report /tmp/rabbitmq-consumer.json
```

Run the focused unit suite without any service mutation:

```bash
pytest tests/unit/tools/stress/test_scheduler_reliability_runner.py
```
