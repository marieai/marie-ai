# Reproduce the 50K DAGs/Hour Gateway Qualification

Use this runbook to qualify the gateway and PostgreSQL scheduler at the
production offered load of 50,000 DAGs per hour. The trial submits 25,000 DAGs
over approximately 30 minutes while retaining the production limit of 64 active
DAGs.

This is a scheduler-throughput qualification. It uses the existing
`mock_parallel_subgraphs` planner and eight mock executor types; it does not
replace the separate run-lease test whose executor duration exceeds
`run_ttl_seconds`.

## Workload Model

One `mock_parallel_subgraphs` DAG contains 24 graph nodes:

- 15 executable jobs
- 9 scheduler-local control-flow nodes
- 2 executable jobs for each of `mock_executor_a` through `mock_executor_g`
- 1 executable job for `mock_executor_h`

The 30-minute trial therefore creates:

- 25,000 DAGs
- 600,000 total graph nodes
- 375,000 executable jobs
- 13.8889 DAG submissions per second
- approximately 208.33 executable jobs per second at steady state

With 120 executor replicas and a per-job mock duration of 0.4 seconds,
theoretical executor capacity is 300 executable jobs per second. The offered
load consumes approximately 69.4% of that service capacity and leaves headroom
for scheduler, database, and handoff overhead.

At 50,000 DAGs per hour, 64 active DAGs permit an average DAG residence time of
4.61 seconds:

```text
64 active DAGs / 13.8889 DAGs per second = 4.608 seconds
```

The qualification is intended to show whether the complete mock graph can
remain within that production concurrency envelope.

## Configure the Test Topology

Use the actual gateway configuration at:

```text
/mnt/data/marie-ai/config/service/marie-gateway-4.0.0.yml
```

Keep these scheduler settings for the trial:

```yaml
job_scheduler_kwargs:
  run_ttl_seconds: 10
  run_lease_renewal_interval_seconds: 2

  dag_manager:
    min_concurrent_dags: 2
    max_concurrent_dags: 64
```

Do not raise `max_concurrent_dags` for this qualification. The 64-DAG limit is
part of the production workload contract being tested.

Configure 120 total replicas in the existing mock executor topology. Match
replicas to the per-DAG executor demand:

| Executor | Jobs per DAG | Replicas |
| --- | ---: | ---: |
| `mock_executor_a` | 2 | 16 |
| `mock_executor_b` | 2 | 16 |
| `mock_executor_c` | 2 | 16 |
| `mock_executor_d` | 2 | 16 |
| `mock_executor_e` | 2 | 16 |
| `mock_executor_f` | 2 | 16 |
| `mock_executor_g` | 2 | 16 |
| `mock_executor_h` | 1 | 8 |

The total is 120 replicas. Set only the `replicas` values in the mock executor
service configuration. The command below supplies `process_time` through job
metadata, which is the normal per-request parameter path used by
`IntegrationExecutorMock`; do not hardcode the qualification duration into the
executor configuration.

Restart the gateway and mock executor topology after changing replica counts.
The stresser's preflight must observe positive capacity for all eight executor
types before it submits the first DAG.

## Enable Full Scheduler Tracing

Use full tracing for this diagnostic qualification so the artifacts retain the
durable submission, frontier, planner, database lease, semaphore,
dispatch admission, executor receipt, callback, terminal-status, and slot-release
stages needed to investigate a stall. Full tracing can produce millions of
records at this scale, so use a new trace file on a filesystem with sufficient
free space and monitor its growth during the run.

Set these variables in the gateway and mock executor process environments before
starting them:

```bash
export MARIE_SCHEDULER_TRACE_ENABLED=true
export MARIE_SCHEDULER_TRACE_PROFILE=full
export MARIE_SCHEDULER_TRACE_PATH=~/tmp/marie-scheduler-50kph-trace.jsonl
```

Use a new trace path for every qualification run so results from earlier trials
are not mixed into the report.

## Run the 30-Minute Qualification

Run from `~/dev/marieai/marie-ai`. Supply the gateway API key through
the environment; never add it to this runbook, a shell script, or a tracked
configuration.

```bash
cd ~/dev/marieai/marie-ai

: "${GATEWAY_API_KEY:?Export GATEWAY_API_KEY first}"

MARIE_STRESS_RUN_ID="scheduler-50kph-$(date -u +%Y%m%dT%H%M%SZ)"
MARIE_STRESS_REPORT_DIR="${HOME}/tmp/${MARIE_STRESS_RUN_ID}"

mkdir -p "${MARIE_STRESS_REPORT_DIR}"

.venv/bin/python tools/stress/gateway_e2e_stresser.py \
  --config tools/stress/gateway-e2e.config.json \
  --api-key "${GATEWAY_API_KEY}" \
  --s3-uri s3://dummy/stress.txt \
  --job-count 25000 \
  --submit-rate 13.8889 \
  --submit-concurrency 128 \
  --run-id "${MARIE_STRESS_RUN_ID}" \
  --job-name mock_parallel_subgraphs \
  --planner mock_parallel_subgraphs \
  --required-executor mock_executor_a \
  --required-executor mock_executor_b \
  --required-executor mock_executor_c \
  --required-executor mock_executor_d \
  --required-executor mock_executor_e \
  --required-executor mock_executor_f \
  --required-executor mock_executor_g \
  --required-executor mock_executor_h \
  --fault-profile normal \
  --mock-process-time 0.4 \
  --mock-failure-rate 0 \
  --soft-sla-seconds 60 \
  --hard-sla-seconds 3600 \
  --min-soft-sla-compliance-pct 95 \
  --min-hard-sla-compliance-pct 99 \
  --preflight-deadline 120 \
  --preflight-interval 1 \
  --terminal-timeout 3600 \
  --min-submission-acceptance-pct 100 \
  --min-terminal-completion-pct 100 \
  --max-event-timeout-jobs 0 \
  --max-open-jobs 0 \
  --require-event-order \
  --max-duplicate-terminal-events 0 \
  --max-conflicting-terminal-events 0 \
  --job-jsonl "${MARIE_STRESS_REPORT_DIR}/jobs.jsonl" \
  --max-retained-jobs 25000 \
  --max-metric-samples 100000 \
  --max-events-per-job 16 \
  --progress-interval 30 \
  --debug-sample-interval 10 \
  --live-report "${MARIE_STRESS_REPORT_DIR}/live.html" \
  --verify-correctness \
  --correctness-config tools/stress/scheduler-db.config.example.json \
  --correctness-timeout 1800 \
  --correctness-report "${MARIE_STRESS_REPORT_DIR}/correctness.json" \
  --trace-mode full \
  --report "${MARIE_STRESS_REPORT_DIR}/final.json"
```

`--job-count 25000` guarantees the requested cohort size. At
`--submit-rate 13.8889`, submission takes approximately 30 minutes. The process
can continue after the submission window while it drains accepted work and runs
the PostgreSQL correctness verifier. `--terminal-timeout 3600` permits up to one
additional hour for that drain; needing a material drain period is itself a
throughput failure even when every DAG eventually completes.

During preflight, the stresser prints the first unmet readiness condition,
prints again immediately if the condition changes, and repeats an unchanged
condition at most once every 10 seconds until the deadline.

## Monitor the Trial

Open the live report printed by the command, or inspect:

```text
~/tmp/<run-id>/live.html
```

During the steady-state portion of the run, verify:

- accepted submissions continue tracking 13.8889 DAGs per second
- terminal throughput converges toward submission throughput
- active DAGs never exceed 64
- request, scheduler, and ready-job queues do not grow monotonically
- every executor type continues releasing and reacquiring capacity
- PostgreSQL pool acquisition wait does not trend upward

Do not declare a pass solely because the backlog drains after submission stops.
The system must sustain the offered rate without unbounded queue growth.

## Pass Criteria

The stresser must exit zero, and the artifacts must show:

- 25,000 of 25,000 submissions accepted
- 100% terminal completion for accepted DAGs
- no terminal event timeouts
- no unexplained open DAGs
- lifecycle events in scheduled, started, terminal order
- no duplicate terminal events
- no conflicting completed and failed terminal events
- at least 95% soft-SLA compliance at 15 seconds
- at least 99% hard-SLA compliance at 45 seconds
- PostgreSQL correctness verification passed
- no sustained queue growth during the steady-state window

A nonzero stresser exit, failed correctness report, sustained backlog growth, or
completion throughput below the offered rate fails the qualification.

## Analyze the Scheduler Trace

After the stresser and correctness verifier finish, run:

```bash
cd ~/dev/marieai/marie-ai

.venv/bin/python tools/stress/analyze_scheduler_trace.py \
  ~/tmp/marie-scheduler-50kph-trace.jsonl \
  --report
```

Review dispatch rate, completion rate, frontier-to-candidate delay, dispatch
batch size and interval, compatible free slots, terminal feedback latency,
PostgreSQL pool wait, and executor slot refill idle time. Compare the trace
window with the final gateway report before attributing a low aggregate rate to
the scheduler; the trace can include startup or idle periods outside the
30-minute submission window.

## Keep the Run-Lease Qualification Separate

This throughput profile uses a 0.4-second executor duration, so its executor jobs
do not cross the 10-second run TTL. The existing run-lease qualification must
still run with a per-request duration greater than 10 seconds.

Do not set all 375,000 executable jobs in this throughput trial to 12 seconds.
At 13.8889 DAGs per second, that workload needs approximately 2,500 continuously
available executor slots:

```text
13.8889 DAGs/s * 15 jobs/DAG * 12 seconds/job = 2,500 slots
```

One hundred twenty replicas cannot drain that workload at the production
submission rate.
Use the production-rate profile in this runbook to qualify throughput and the
long-duration profile to qualify run-lease renewal and restart recovery. Both
must pass before closing the scheduler hangup investigation.
