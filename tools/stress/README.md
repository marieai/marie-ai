# Marie-AI Stress Testing Tools

This directory contains stress testing tools for testing the Marie gateway and networking components.

## Tools

### 1. `gateway_e2e_stresser.py`

End-to-end scheduler stress harness for real document jobs.

This tool is for validating the full path:

- pre-staged S3 selection or optional local upload
- optional S3/MinIO upload when you want full-pipeline benchmarking
- planner-aware `job submit` through the gateway
- RabbitMQ scheduler event tracking
- end-to-end latency and failure reporting

Use it when the goal is to test:

- gateway scheduling behavior under load
- queueing and start latency
- planner selection and submission correctness
- downstream LLM / annotator failure handling
- terminal completion vs failure rates

#### Features

- **Real asset staging**: uploads source files to the configured S3/MinIO bucket
- **Existing S3 asset mode**: submits pre-staged `s3://` assets without uploading
- **Planner-aware submit**: sets `planner`, `ref_id`, `ref_type`, and `uri` in metadata
- **Per-submit visibility**: logs the selected source, planned `s3://` URI, request ID, and returned job ID for every real submission
- **Scheduler event tracking**: listens for `*.scheduled`, `*.started`, `*.completed`, `*.failed`
- **Companion metadata support**: uploads `<file>.meta.json` sidecars when present
- **Flexible load model**: submit either a fixed `--job-count` or run for a wall-clock `--run-time` at a target rate
- **Live progress monitoring**: configurable console progress cadence and optional live JSON or HTML snapshots during the run
- **Latency breakdowns**: submit, scheduling, queue wait, execution, and end-to-end timing
- **SLA verification**: stamps `soft_sla` / `hard_sla` onto each request and reports compliance
- **Mock executor failure injection**: stamps `failure_rate`, `failure_mode`, and deterministic `force_fail` controls for mock-executor runs
- **AIMock fault profile integration**: can switch the mock backend into `normal`, `timeout`, `error`, or randomized `chaos`

#### Usage

Enable the scheduler JSONL trace when investigating unexplained SLA gaps. Set
these variables on the gateway and every executor process you want in the same
timeline:

```bash
export MARIE_SCHEDULER_TRACE_ENABLED=true
export MARIE_SCHEDULER_TRACE_PATH=~/tmp/marie-scheduler-trace.jsonl
export MARIE_SCHEDULER_TRACE_PROFILE=full
```

The trace is disabled by default and is best-effort: write failures never block
job scheduling. The default profile is `compact`, which is meant for long
endurance runs and keeps submit, dispatch, terminal, batch, priority-refresh,
DAG-sync, and pool-pressure signals. Use `MARIE_SCHEDULER_TRACE_PROFILE=full`
for short bottleneck investigations that need scheduler submission queue
enqueue/dequeue, DAG persistence, frontier insertion, planner selection,
frontier and DB leasing, semaphore reservation, gateway dispatch confirmation,
executor request receipt, RUNNING/SUCCEEDED/FAILED status writes, callbacks, and
slot release. The full profile splits end-to-end latency into submission queue
wait, DAG persistence, frontier wait, dispatch wait, executor service time, and
terminal status/slot-release delay. Frontier wait is further split into
candidate visibility, planner selection, frontier take, DB lease, semaphore
reservation, and activation. Planner and frontier phases are emitted as batch
events with `job_ids`; the analyzer expands them per job without writing one
trace line per selected job.

After a run, summarize the slowest handoffs:

```bash
python tools/stress/analyze_scheduler_trace.py \
    ~/tmp/marie-scheduler-trace.jsonl \
    --sort frontier_to_dispatch \
    --limit 25
```

Or print an aggregate report with event rates, executor utilization, planner
pressure, control-flow balance, and latency percentiles:

```bash
python tools/stress/analyze_scheduler_trace.py \
    ~/tmp/marie-scheduler-trace.jsonl \
    --report
```

```bash
# Full end-to-end extract test using the local stress config example
python tools/stress/gateway_e2e_stresser.py \
    --config tools/stress/gateway-e2e.config.example.json \
    --input-dir ~/.marie/generators \
    --job-count 25 \
    --job-name extract \
    --planner extract

# Submit TIFFs at 4 jobs/sec and write an HTML report
python tools/stress/gateway_e2e_stresser.py \
    --config tools/stress/gateway-e2e.config.example.json \
    --s3-uri-manifest tools/stress/gateway-e2e.s3-uri-manifest.example.txt \
    --job-count 50 \
    --job-name gen5_extract \
    --planner extract \
    --soft-sla-seconds 30 \
    --hard-sla-seconds 90 \
    --soft-sla-step-seconds 10 \
    --hard-sla-step-seconds 20 \
    --sla-step-every-jobs 25 \
    --sla-step-cycle 4 \
    --min-hard-sla-compliance-pct 99 \
    --submit-rate 4 \
    --report /tmp/gateway-e2e-report.html

# Run for one hour at 10 jobs/sec against pre-staged S3 assets
python tools/stress/gateway_e2e_stresser.py \
    --config tools/stress/gateway-e2e.config.example.json \
    --s3-uri-manifest tools/stress/gateway-e2e.s3-uri-manifest.example.txt \
    --run-time 1h \
    --job-name gen5_extract \
    --planner extract \
    --submit-rate 10 \
    --progress-interval 2 \
    --live-report /tmp/gateway-e2e-live.html \
    --report /tmp/gateway-e2e-hourly-report.html

# Run against AIMock/LiteLLM with randomized chaos mode
python tools/stress/gateway_e2e_stresser.py \
    --config tools/stress/gateway-e2e.config.example.json \
    --s3-uri 's3://marie/gen5_extract/sample-001.tif' \
    --job-count 10 \
    --job-name gen5_extract \
    --planner extract \
    --fault-profile chaos \
    --aimock-admin-url http://localhost:4011

# Force every fifth mock-executor job to fail while keeping the rest at a 10% failure rate
python tools/stress/gateway_e2e_stresser.py \
    --config tools/stress/gateway-e2e.config.example.json \
    --s3-uri-manifest tools/stress/gateway-e2e.s3-uri-manifest.example.txt \
    --job-count 50 \
    --job-name mock_parallel_subgraphs \
    --planner mock_parallel_subgraphs \
    --mock-failure-rate 0.10 \
    --mock-failure-mode exception \
    --force-failure-every 5 \
    --report /tmp/gateway-e2e-failures.html

# Exercise the real LLM annotator executor path with AIMock behind OpenAIEngine
# The mock_annotator_llm template purges mock-llm output so repeated runs submit
# fresh LLM dispatch work instead of reusing agent-output/mock-llm.
python tools/stress/gateway_e2e_stresser.py \
    --config tools/stress/gateway-e2e.config.json \
    --api-key "$GATEWAY_API_KEY" \
    --input-dir ~/.marie/generators \
    --job-count 1 \
    --job-name gen5_extract \
    --planner mock_annotator_llm \
    --llm-pool-id document-small \
    --ref-type stress \
    --project-id mock-annotator-llm-stress \
    --request-template tools/stress/mock_annotator_llm.invoke.json \
    --dry-run

# Preview exactly what would be submitted without uploading or calling the gateway
python tools/stress/gateway_e2e_stresser.py \
    --config tools/stress/gateway-e2e.config.example.json \
    --input-dir ~/.marie/generators \
    --job-count 1 \
    --job-name gen5_extract \
    --planner extract \
    --dry-run
```

#### Quick Start

The usual scheduler-stress path is:

1. start the mock LLM backend
2. start LiteLLM pointed at that mock backend
3. make sure gateway + RabbitMQ + MinIO/S3 are already running
4. run `gateway_e2e_stresser.py` against pre-staged `s3://` assets

The repo-local base files for this tool are:

- `tools/stress/gateway-e2e.config.example.json`
- `tools/stress/gateway-e2e.s3-uri-manifest.example.txt`

These files are examples only. They are not required.

- `gateway-e2e.config.example.json` is a local starter config for the tool
- `gateway-e2e.s3-uri-manifest.example.txt` is a sample manifest showing the expected `s3://` URI format

Create the shared Docker network once:

```bash
docker network create --driver=bridge marie_default 2>/dev/null || true
```

Start the programmatic AIMock backend with admin control enabled:

```bash
cd Dockerfiles
docker compose -f docker-compose.mock-llm-programmatic.yml up -d

# Verify the admin endpoint
curl http://localhost:4011/fault-profile
```

Start LiteLLM and point it at the programmatic AIMock service on the Docker network:

```bash
AIMOCK_URL=http://aimock-programmatic:4010 \
docker compose --env-file ./config/.env.dev \
  -f ./Dockerfiles/docker-compose.litellm.yml \
  --project-directory . \
  up -d

# Verify LiteLLM
curl http://localhost:4000/health
```

If LiteLLM is running with host networking, use:

```bash
AIMOCK_URL=http://127.0.0.1:4010 \
docker compose --env-file ./config/.env.dev \
  -f ./Dockerfiles/docker-compose.litellm.yml \
  --project-directory . \
  up -d
```

Networking rule of thumb:

- LiteLLM on Docker bridge network talking to `aimock-programmatic` container:
  `AIMOCK_URL=http://aimock-programmatic:4010`
- LiteLLM on host network talking to AIMock bound on the same host:
  `AIMOCK_URL=http://127.0.0.1:4010`

If you want to inspect or change the active AIMock profile manually:

```bash
# Read the active profile
curl http://localhost:4011/fault-profile

# Force timeout mode
curl -X POST http://localhost:4011/fault-profile \
  -H 'Content-Type: application/json' \
  -d '{"profile":"timeout"}'

# Return to normal mode
curl -X POST http://localhost:4011/fault-profile \
  -H 'Content-Type: application/json' \
  -d '{"profile":"normal"}'
```

#### Input Modes

`gateway_e2e_stresser.py` supports two input modes.

It also supports two load-shaping modes:

- `--job-count N` for a fixed number of submissions
- `--run-time 30s|2m|1h` for a duration-based run at `--submit-rate`

For real-time monitoring during the run:

- `--progress-interval` controls how often the tool logs progress to the terminal
- `--live-report` rewrites a lightweight status snapshot on the same cadence
- `--live-report-format` can force `json` or `html`; otherwise the tool infers the format from the file extension

Pre-staged S3 mode:

- use `--s3-uri` or `--s3-uri-manifest`
- the tool skips upload and submits directly against existing `s3://` objects
- this is the preferred mode for scheduler, queueing, timeout, and LLM failure testing
- local files are not required in this mode

Upload mode:

- use `--input-dir`, `--input-glob`, or `--input-manifest`
- the tool uploads local files to S3/MinIO before submission
- use this only when you want a true end-to-end ingest benchmark

#### Dry-Run Mode

Use `--dry-run` when you want to inspect the fully resolved submit plan before
the tool touches S3 or the gateway.

What `--dry-run` does:

- resolves the exact input file(s) or `s3://` URI(s) that would be used
- computes the destination `s3_uri` for local upload mode
- builds the final metadata, including SLA fields
- builds the exact request payload body the tool would submit
- prints the whole plan as JSON to stdout
- in `--run-time` mode, previews only the first few would-be submissions instead of materializing the full duration run

What `--dry-run` does not do:

- does not upload local files
- does not call the gateway
- does not switch AIMock fault profiles
- does not wait for scheduler events

On a normal run without `--dry-run`, the tool also logs a one-line submit summary
for each job so you can see exactly what was sent:

```text
Submitted job_index=0 request_id=job-0-... job_id=<gateway-job-id> planner=extract input_mode=upload source=/path/to/sample.tif s3_uri=s3://marie/extract/...
```

Example:

```bash
python tools/stress/gateway_e2e_stresser.py \
  --config tools/stress/gateway-e2e.config.example.json \
  --input-dir ~/.marie/generators \
  --job-count 1 \
  --job-name gen5_extract \
  --planner extract \
  --dry-run \
  --report /tmp/gateway-e2e-dry-run.html
```

When `--run-time` is used with `--dry-run`, the tool reports:

- `run_mode: duration`
- `run_time_seconds`
- `estimated_job_count`
- `preview_job_count`

Use `--dry-run-preview-count` to control how many preview submissions are emitted.

#### Live Monitoring

Use `--progress-interval` to control the console update cadence.

Example:

```bash
python tools/stress/gateway_e2e_stresser.py \
  --config tools/stress/gateway-e2e.config.example.json \
  --s3-uri-manifest tools/stress/gateway-e2e.s3-uri-manifest.example.txt \
  --run-time 30m \
  --job-name gen5_extract \
  --planner extract \
  --submit-rate 10 \
  --progress-interval 2 \
  --live-report /tmp/gateway-e2e-live.html
```

The live report is rewritten on each interval and can be either:

- JSON when the output path ends with `.json`
- HTML when the output path ends with `.html`
- an explicitly forced format via `--live-report-format json|html`

The live report focuses on aggregate run health:

- target submit rate vs observed created/completed throughput
- open jobs, inflight jobs, and pending-submit backlog
- terminal success rate and submit acceptance rate
- observed latency summaries for submit, scheduling, queue wait, execution, and end-to-end
- queue and dispatcher signals from the latest debug sample
- recent failures only, instead of a general recent-job list

The final HTML report now carries the same SLA context forward and adds:

- `SLA Outcome` with configured jobs, met jobs, missed jobs, overdue-open counts, and compliance
- `Worst SLA Misses` with the highest-lateness jobs across soft and hard deadlines
- the existing per-job table for full drill-down

Debug sampling is off by default. To enable it, pass a positive interval:

```bash
python tools/stress/gateway_e2e_stresser.py \
  --config tools/stress/gateway-e2e.config.example.json \
  --s3-uri-manifest tools/stress/gateway-e2e.s3-uri-manifest.example.txt \
  --run-time 15m \
  --job-name gen5_extract \
  --planner extract \
  --submit-rate 5 \
  --debug-sample-interval 5 \
  --live-report /tmp/gateway-e2e-live.html
```

The stresser polls:

- `http://<gateway-host>:<http-port>/api/debug`

Important:

- debug sampling requires a reachable gateway HTTP port
- if you are using `--protocol grpc`, you still need `--http-port` set correctly for `/api/debug`
- `0` means disabled, which is why the report currently shows it as off unless you opt in

Example inspection for JSON:

```bash
watch -n 2 'jq . /tmp/gateway-e2e-live.json'
```

Example HTML run:

```bash
python tools/stress/gateway_e2e_stresser.py \
  --config tools/stress/gateway-e2e.config.example.json \
  --s3-uri-manifest tools/stress/gateway-e2e.s3-uri-manifest.example.txt \
  --run-time 30m \
  --job-name gen5_extract \
  --planner extract \
  --submit-rate 10 \
  --progress-interval 2 \
  --live-report /tmp/gateway-e2e-live.html
```

Open `/tmp/gateway-e2e-live.html` in a browser and it will auto-refresh.

The live report includes fields like:

- `source_path`
- `input_mode`
- `s3_uri`
- `metadata`
- `request_payload`
- `transport`

#### Sample `--s3-uri-manifest`

The manifest is just a plain text file with one existing `s3://` URI per line:

```text
s3://marie/gen5_extract/sample-001.tif
s3://marie/gen5_extract/sample-002.tif
s3://marie/gen5_extract/sample-003.tif
```

The tool does not create these S3 objects. They must already exist in S3/MinIO.
The point of the manifest is to let the stress run reuse a prepared asset set
without re-uploading data on every run.

How it is used:

- if you pass `--s3-uri`, the run reuses that single object for every submitted job
- if you pass `--s3-uri-manifest`, the tool reads the listed URIs and cycles through them round-robin as submissions increase
- if the total number of submissions is larger than the number of manifest entries, the tool wraps and reuses the listed URIs

Use `--s3-uri` when one object is enough. Use `--s3-uri-manifest` when you want
to spread the run across a set of existing staged assets.

Example sample file:

```bash
cat tools/stress/gateway-e2e.s3-uri-manifest.example.txt
```

#### Common Runs

Scheduler and LLM stress against pre-staged S3 assets:

```bash
source ~/environment/marie-3.12/bin/activate

python tools/stress/gateway_e2e_stresser.py \
  --config tools/stress/gateway-e2e.config.example.json \
  --s3-uri-manifest tools/stress/gateway-e2e.s3-uri-manifest.example.txt \
  --job-count 1000 \
  --job-name gen5_extract \
  --planner extract \
  --submit-rate 4 \
  --fault-profile normal \
  --aimock-admin-url http://localhost:4011 \
  --report /tmp/gateway-e2e-report.html
```


python tools/stress/gateway_e2e_stresser.py   --config tools/stress/gateway-e2e.config.example.json   --input-dir ~/.marie/generators   --run-time 5m   --job-name extract   --planner extract   --submit-concurrency 1   --submit-rate 1   --soft-sla-seconds 30   --hard-sla-seconds 120   --min-soft-sla-compliance-pct 95   --min-hard-sla-compliance-pct 99   --progress-interval 5   --terminal-timeout 1800   --fault-profile normal   --live-report ~/tmp/gateway-e2e-8h-live.html   --report ~/tmp/gateway-e2e-8h-final.json



Eight-hour scheduler endurance run for 10 extract executors with a 1-second
mock workload:

```bash
source ~/environment/marie-3.12/bin/activate
mkdir -p ~/tmp

export MARIE_SCHEDULER_TRACE_ENABLED=true
export MARIE_SCHEDULER_TRACE_PATH=~/tmp/marie-scheduler-trace-8h.jsonl
export MARIE_SCHEDULER_TRACE_PROFILE=compact

python tools/stress/gateway_e2e_stresser.py \
  --config tools/stress/gateway-e2e.config.example.json \
  --input-dir ~/.marie/generators \
  --run-time 8h \
  --job-name extract \
  --planner extract \
  --submit-concurrency 100 \
  --submit-rate 8 \
  --soft-sla-seconds 30 \
  --hard-sla-seconds 120 \
  --min-soft-sla-compliance-pct 95 \
  --min-hard-sla-compliance-pct 99 \
  --progress-interval 30 \
  --terminal-timeout 1800 \
  --fault-profile normal \
  --live-report ~/tmp/gateway-e2e-8h-live.html \
  --report ~/tmp/gateway-e2e-8h-final.json
```

Preconditions:

- run exactly 10 `extract_executor` slots
- configure the mock extract workload to sleep for 1 second
- keep the AIMock / executor profile in `normal` mode
- prefer pre-staged S3 inputs for pure scheduler tests; the command above uses
  upload mode because it matches the local generator input path

The `--submit-rate 8` target leaves headroom under the theoretical 10 jobs/sec
capacity of 10 one-second executor slots. Use `--submit-rate 10` for a
saturation run. Avoid `--submit-rate 250` for this 8-hour SLA run unless the
goal is intentionally to create a large backlog and measure overload behavior.

After the run, summarize the scheduler trace:

```bash
python tools/stress/analyze_scheduler_trace.py \
  ~/tmp/marie-scheduler-trace-8h.jsonl \
  --sort frontier_to_dispatch \
  --limit 25 \
  --report
```

The compact trace profile is intended for long endurance runs. It keeps submit,
dispatch, terminal, batch, priority-refresh, DAG-sync, and pool-pressure signals
while dropping per-stage debug events that can turn an 8-hour trace into a very
large JSONL file. Use `MARIE_SCHEDULER_TRACE_PROFILE=full` only for shorter
bottleneck investigations where detailed stage timings are required.

#### Benchmark Matrix

Use a rate sweep to find the first sustained submit rate where:

- soft SLA drops below target
- hard SLA drops below target
- queue wait starts growing across the run
- completed throughput stops tracking submit rate

For the 1-second extract mock, start with fixed SLAs and sweep by executor count.

Recommended sweep bands:

| Executor Slots | Submit Rates to Sweep |
| --- | --- |
| `1` | `0.6 0.7 0.8 0.9 1.0` |
| `2` | `1.0 1.2 1.4 1.6 1.8 2.0` |

Run each point for at least `2m` with the same SLA contract:

- `--soft-sla-seconds 15`
- `--hard-sla-seconds 45`
- `--min-soft-sla-compliance-pct 95`
- `--min-hard-sla-compliance-pct 99`

The repo includes a small runner for this sweep:

```bash
bash tools/stress/run_gateway_benchmark_matrix.sh \
  --executor-count 1 \
  --input-dir ~/.marie/generators \
  --report-dir ~/tmp/gateway-matrix-1x
```

For two executor slots:

```bash
bash tools/stress/run_gateway_benchmark_matrix.sh \
  --executor-count 2 \
  --input-dir ~/.marie/generators \
  --report-dir ~/tmp/gateway-matrix-2x
```

To override the default sweep:

```bash
bash tools/stress/run_gateway_benchmark_matrix.sh \
  --executor-count 1 \
  --input-dir ~/.marie/generators \
  --rates "0.65 0.75 0.85 0.95" \
  --report-dir ~/tmp/gateway-matrix-custom
```

Each run writes:

- one HTML report per rate
- one live HTML report per rate while that point is running
- one raw console log per rate
- a `summary.txt` file listing the rate, pass/fail status, and output paths
- a stable `current-live.html` symlink pointing at the active rate's live report

While the matrix is running, keep this open in a browser:

```text
~/tmp/gateway-matrix-1x/current-live.html
```

Or watch the current point from the shell:

```bash
ls -l ~/tmp/gateway-matrix-1x/current-live.html
```

How to read the matrix:

- `PASS`: both SLA thresholds met and the stresser exited `0`
- `FAIL`: at least one SLA threshold missed and the stresser exited `2`
- first failing rate: practical SLA ceiling for that executor count
- last passing rate: safe operating point for that executor count

When the goal is pure scheduler capacity, prefer pre-staged `s3://` assets over local upload mode so S3 transfer variance does not contaminate the ceiling measurement.

The SLA flags support both fixed and incremental deadlines:

- `--soft-sla-seconds` / `--hard-sla-seconds`: base deadlines from submit start
- `--soft-sla-step-seconds` / `--hard-sla-step-seconds`: increment applied per SLA bucket
- `--sla-step-every-jobs`: how many jobs share the same bucket before the increment advances
- `--sla-step-cycle`: optional wraparound after N buckets

#### SLA Modes

Fixed SLA mode:

- every job gets the same `soft_sla` and `hard_sla`
- use only `--soft-sla-seconds` and `--hard-sla-seconds`
- best when you want one contractual target for the whole run

Incremental SLA mode:

- jobs are grouped into SLA buckets
- each bucket shifts the deadline by the configured step size
- best when you want one run to simulate mixed urgency classes

Bucket formula:

- `bucket_index = floor(job_index / sla_step_every_jobs)`
- if `--sla-step-cycle` is set, `bucket_index = bucket_index % sla_step_cycle`
- `soft_offset = soft_sla_seconds + bucket_index * soft_sla_step_seconds`
- `hard_offset = hard_sla_seconds + bucket_index * hard_sla_step_seconds`

Two useful patterns:

- tighter deadlines over time: `--soft-sla-step-seconds -5 --hard-sla-step-seconds -15`
- looser deadlines over time: `--soft-sla-step-seconds 5 --hard-sla-step-seconds 15`

Timeout-profile run:

```bash
python tools/stress/gateway_e2e_stresser.py \
  --config tools/stress/gateway-e2e.config.example.json \
  --s3-uri-manifest tools/stress/gateway-e2e.s3-uri-manifest.example.txt \
  --job-count 250 \
  --job-name gen5_extract \
  --planner extract \
  --submit-rate 2 \
  --fault-profile timeout \
  --aimock-admin-url http://localhost:4011 \
  --report /tmp/gateway-timeout-report.html
```

SLA verification run:

```bash
python tools/stress/gateway_e2e_stresser.py \
  --config tools/stress/gateway-e2e.config.example.json \
  --s3-uri-manifest tools/stress/gateway-e2e.s3-uri-manifest.example.txt \
  --job-count 200 \
  --job-name gen5_extract \
  --planner extract \
  --submit-rate 4 \
  --soft-sla-seconds 20 \
  --hard-sla-seconds 60 \
  --min-soft-sla-compliance-pct 95 \
  --min-hard-sla-compliance-pct 99 \
  --report /tmp/gateway-sla-report.html
```

If a configured SLA verification threshold is missed, the tool exits with status code `2`.

Incremental deadline run:

```bash
python tools/stress/gateway_e2e_stresser.py \
  --config tools/stress/gateway-e2e.config.example.json \
  --s3-uri-manifest tools/stress/gateway-e2e.s3-uri-manifest.example.txt \
  --job-count 300 \
  --job-name gen5_extract \
  --planner extract \
  --submit-rate 5 \
  --soft-sla-seconds 15 \
  --hard-sla-seconds 45 \
  --soft-sla-step-seconds 5 \
  --hard-sla-step-seconds 15 \
  --sla-step-every-jobs 25 \
  --sla-step-cycle 4 \
  --report /tmp/gateway-sla-stepped-report.html
```

That example yields four repeating SLA classes:

- jobs `0-24`: soft `15s`, hard `45s`
- jobs `25-49`: soft `20s`, hard `60s`
- jobs `50-74`: soft `25s`, hard `75s`
- jobs `75-99`: soft `30s`, hard `90s`

Then the pattern wraps and repeats for the rest of the run.

Decrementing deadline run:

```bash
python tools/stress/gateway_e2e_stresser.py \
  --config tools/stress/gateway-e2e.config.example.json \
  --s3-uri-manifest tools/stress/gateway-e2e.s3-uri-manifest.example.txt \
  --job-count 200 \
  --job-name gen5_extract \
  --planner extract \
  --submit-rate 4 \
  --soft-sla-seconds 40 \
  --hard-sla-seconds 120 \
  --soft-sla-step-seconds -5 \
  --hard-sla-step-seconds -15 \
  --sla-step-every-jobs 20 \
  --sla-step-cycle 5 \
  --report /tmp/gateway-sla-tightening-report.html
```

That pattern starts relaxed and gets tighter every 20 jobs:

- jobs `0-19`: soft `40s`, hard `120s`
- jobs `20-39`: soft `35s`, hard `105s`
- jobs `40-59`: soft `30s`, hard `90s`
- jobs `60-79`: soft `25s`, hard `75s`
- jobs `80-99`: soft `20s`, hard `60s`

Then it wraps and repeats.

Randomized monkey/chaos run:

```bash
python tools/stress/gateway_e2e_stresser.py \
  --config tools/stress/gateway-e2e.config.example.json \
  --s3-uri-manifest tools/stress/gateway-e2e.s3-uri-manifest.example.txt \
  --job-count 500 \
  --job-name gen5_extract \
  --planner extract \
  --submit-rate 3 \
  --fault-profile chaos \
  --aimock-admin-url http://localhost:4011 \
  --report /tmp/gateway-chaos-report.html
```

True full-pipeline run with local upload:

```bash
python tools/stress/gateway_e2e_stresser.py \
  --config tools/stress/gateway-e2e.config.example.json \
  --input-dir ~/.marie/generators \
  --job-count 100 \
  --job-name gen5_extract \
  --planner extract \
  --submit-rate 2 \
  --report /tmp/gateway-upload-report.html
```

Cycle work across document-size LLM pools:

```bash
python tools/stress/gateway_e2e_stresser.py \
  --config tools/stress/gateway-e2e.config.example.json \
  --s3-uri-manifest /tmp/staged-documents.txt \
  --job-count 60 \
  --job-name gen5_extract \
  --planner extract \
  --llm-pool-cycle document-small,document-medium,document-large \
  --debug-sample-interval 5
```

#### Important options

| Option | Description |
|--------|-------------|
| `--config` | Stress config JSON with `api_base_url`, `api_key`, `storage`, and `queue` |
| `--api-key` | Gateway API key override; use this when the stress runner cannot read the gateway service YAML |
| `--input-dir` / `--input-glob` / `--input-manifest` | Local source file discovery for upload mode |
| `--s3-uri` / `--s3-uri-manifest` | Pre-staged S3 objects for submit-only mode |
| `--job-count` | Number of jobs to submit in fixed-count mode |
| `--run-time` | Duration-based mode, for example `30s`, `2m`, or `1h` |
| `--progress-interval` | Console progress and live report refresh cadence in seconds |
| `--live-report` | Path to a live status report rewritten during the run |
| `--live-report-format` | Override live report format; default `auto` infers from `.json` or `.html` |
| `--job-name` | Gateway submit name / scheduler queue name |
| `--planner` | Planner to place in metadata |
| `--llm-pool-id` | Fixed LLM dispatch pool ID to place in `metadata.pool_id`, for example `document-small` |
| `--llm-pool-cycle` | Comma-separated LLM dispatch pool IDs to cycle through `metadata.pool_id` by generated job index |
| `--purge-annotators` | Comma-separated annotator names to purge before annotation, for example `mock-llm` |
| `--submit-rate` | Target submit rate in jobs per second |
| `--dry-run-preview-count` | Number of sample submissions to show when `--dry-run` is combined with `--run-time` |
| `--fault-profile` | Run label and AIMock control target: `normal`, `timeout`, `error`, `chaos` |
| `--aimock-admin-url` | AIMock admin endpoint used to switch the active fault profile before the run |
| `--soft-sla-seconds` | Relative soft SLA target from submit start |
| `--hard-sla-seconds` | Relative hard SLA target from submit start |
| `--soft-sla-step-seconds` | Soft SLA increment applied per SLA bucket |
| `--hard-sla-step-seconds` | Hard SLA increment applied per SLA bucket |
| `--sla-step-every-jobs` | Number of jobs per SLA bucket before incrementing |
| `--sla-step-cycle` | Optional number of buckets before the incremental pattern wraps |
| `--min-soft-sla-compliance-pct` | Optional soft SLA verification threshold |
| `--min-hard-sla-compliance-pct` | Optional hard SLA verification threshold |
| `--submit-concurrency` | Concurrent upload+submit workers |
| `--submit-rate` | Job submit rate in jobs/sec |
| `--terminal-timeout` | Max wait for terminal scheduler events |
| `--request-template` | JSON file containing metadata or a full `invoke_action` template |
| `--report` | Write the final report as JSON or HTML |
| `--report-format` | Override final report format; default `auto` infers from `.json` or `.html` |
| `--dry-run` | Print the fully resolved submit plan and payload(s) as JSON without uploading or submitting |

#### Output

The report breaks timing into:

- **submit latency**: gateway `job submit` request/response
- **scheduling latency**: submit response to `*.scheduled`
- **queue wait**: submit response to `*.started` (or `*.scheduled` when no start event exists)
- **execution latency**: `*.started` to terminal event
- **end-to-end latency**: submit start to terminal event

When SLA options are enabled the report also includes:

- **soft SLA compliance**: how many submitted jobs completed within `soft_sla`
- **hard SLA compliance**: how many submitted jobs completed within `hard_sla`
- **SLA lateness**: how late missed jobs were beyond the configured deadline
- **verification result**: pass/fail against the configured minimum compliance thresholds
- **SLA bucket index**: which incremental deadline bucket the job belonged to
- **resolved SLA offsets**: the actual `soft_sla_offset_seconds` and `hard_sla_offset_seconds` used for that job

The JSON report includes per-job SLA fields:

- `sla_bucket_index`
- `soft_sla_offset_seconds`
- `hard_sla_offset_seconds`
- `soft_sla`
- `hard_sla`
- `soft_sla_status`
- `hard_sla_status`
- `soft_sla_met`
- `hard_sla_met`
- `soft_sla_lateness_ms`
- `hard_sla_lateness_ms`

This is the primary tool to use when intentionally restarting LiteLLM,
annotators, or other downstream services to see how scheduler outcomes change.

For scheduler and LLM fault testing, prefer:

- `--s3-uri` or `--s3-uri-manifest` when the asset is already staged and you want to isolate queueing, scheduling, submit, and failure behavior
- `--fault-profile chaos` when using the programmatic AIMock stack

`chaos` is the randomized monkey-test mode in this setup.

#### Stopping The Mock Stack

```bash
docker compose -f ./Dockerfiles/docker-compose.mock-llm-programmatic.yml down
docker compose -f ./Dockerfiles/docker-compose.litellm.yml down
```

---

### 2. `gateway_stresser.py`

A stress tester for the Marie gateway that supports both gRPC and HTTP protocols.

#### Features

- **Multi-protocol support**: gRPC and HTTP (via aiohttp)
- **Configurable load**: Concurrency, request rate, duration
- **Authentication**: Bearer token support (default API key included)
- **Metrics collection**: Latency percentiles (p50, p95, p99), success rates, throughput
- **Health checks**: Automatic connectivity testing before stress test

#### Usage

```bash
# Basic HTTP test (uses /api/v1/invoke and default API key)
python gateway_stresser.py --protocol http --http-port 51000

# Basic gRPC test
python gateway_stresser.py --protocol grpc --gateway-port 52000

# High load test
python gateway_stresser.py --protocol http --http-port 51000 \
    --concurrency 50 --request-rate 100 --duration 120

# Test specific endpoint
python gateway_stresser.py --protocol http --http-port 51000 --endpoint /extract

# With custom request parameters
python gateway_stresser.py --protocol http --http-port 51000 \
    --parameters '{"invoke_action": {"action_type": "command", "command": "job", "action": "submit", "name": "test"}}'

# With reusable mock planner payload
python gateway_stresser.py --protocol http --http-port 51000 \
    --parameters "$(cat tools/stress/mock_parallel_subgraphs.invoke.json)"

# With reusable mock annotator LLM payload
python gateway_stresser.py --protocol http --http-port 51000 \
    --parameters "$(cat tools/stress/mock_annotator_llm.invoke.json)"

# The payload can vary values per request
# Supported placeholders: {{request_id}}, {{timestamp}}, {{timestamp_ms}}, {{api_key}}

# Compare gRPC vs HTTP performance
python gateway_stresser.py --protocol grpc --gateway-port 52000 --duration 60
python gateway_stresser.py --protocol http --http-port 51000 --duration 60
```

#### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--gateway-host` | `localhost` | Gateway host |
| `--gateway-port` | `52000` | Gateway gRPC port |
| `--http-port` | same as gateway-port | Gateway HTTP port |
| `--protocol` | `grpc` | Protocol: `grpc`, `http`, `websocket` |
| `--endpoint` | `/api/v1/invoke` | Endpoint to test |
| `--concurrency` | `10` | Number of concurrent workers |
| `--request-rate` | `10.0` | Target requests per second |
| `--timeout` | `30.0` | Request timeout in seconds |
| `--duration` | `60.0` | Test duration in seconds |
| `--warmup` | `5.0` | Warmup period in seconds |
| `--batch-size` | `1` | Documents per request |
| `--api-key` | (default key) | API key for authentication |
| `--parameters` | `None` | JSON string of request parameters |
| `--target-executor` | `None` | Target executor name |
| `-v, --verbose` | `False` | Enable verbose logging |

#### Output

For `/api/v1/invoke` job submissions, the tool now treats gateway application
errors as failures even when the HTTP status is `200`. This catches cases where
the gateway returns:

```json
{"parameters":{"status":"error","msg":"..."}}
```

instead of silently counting them as successful requests.

The tool provides real-time progress updates and a final report:

```
======================================================================
GATEWAY STRESS TEST REPORT
======================================================================

Test Duration: 60.0 seconds

--- Request Summary ---
Total Requests: 600
Successful: 598
Failed: 2
Timeouts: 0
Success Rate: 99.67%
Throughput: 10.00 req/s

--- Latency Statistics (ms) ---
Min: 5.23
Max: 125.67
Avg: 15.42
P50: 12.34
P95: 45.67
P99: 89.12
Std Dev: 12.45

======================================================================
RESULT: EXCELLENT - Gateway performing well under load
======================================================================
```

---

### 4. `etcd_outage_simulator.py`

Injects ETCD outages by calling `docker pause` / `docker unpause` against a
running ETCD container such as `etcd-single`.

#### Features

- **Repeatable outage cycles**: Pause/recover ETCD for a fixed number of cycles
- **Safe cleanup**: Attempts to unpause the container if the script is interrupted
- **Jitter support**: Add randomness to outage and recovery windows
- **Dry-run mode**: Validate timing and workflow without touching Docker

#### Usage

```bash
# Single 10 second ETCD outage against the default etcd-single container
python tools/stress/etcd_outage_simulator.py --pause-seconds 10 --recover-seconds 20

# Three outage cycles with jitter
python tools/stress/etcd_outage_simulator.py \
    --cycles 3 \
    --pause-seconds 8 \
    --recover-seconds 15 \
    --pause-jitter 2 \
    --recover-jitter 3

# Preview the outage schedule without running docker commands
python tools/stress/etcd_outage_simulator.py --dry-run --cycles 2
```

#### Typical workflow for reconnect testing

```bash
# Terminal 1: keep gateway traffic flowing
python tools/stress/gateway_stresser.py --protocol http --http-port 51000 --duration 120

# Terminal 2: inject ETCD outages
python tools/stress/etcd_outage_simulator.py --cycles 3 --pause-seconds 10 --recover-seconds 20

# Terminal 3: watch logs or ETCD keys
docker exec etcd-single etcdctl get "/marie/gateway/marie" --prefix=true
```

---

## Prerequisites

1. **Marie gateway must be running**:
   ```bash
   marie server --start --uses config/service/marie.yml
   ```

2. **Python dependencies**:
   ```bash
   pip install aiohttp grpcio
   ```

## Typical Test Workflow

1. **Start the gateway**:
   ```bash
   marie server --start --uses /mnt/data/marie-ai/config/service/extract/marie-extract-4.0.0.yml
   ```

2. **Run basic connectivity test**:
   ```bash
   python tools/stress/gateway_stresser.py --protocol http --http-port 51000 --duration 10
   ```

3. **Run load test**:
   ```bash
   python tools/stress/gateway_stresser.py --protocol http --http-port 51000 \
       --concurrency 50 --request-rate 100 --duration 300
   ```

4. **Compare protocols**:
   ```bash
   # gRPC test
   python tools/stress/gateway_stresser.py --protocol grpc --gateway-port 52000 --duration 60

   # HTTP test
   python tools/stress/gateway_stresser.py --protocol http --http-port 51000 --duration 60
   ```

## Interpreting Results

| Success Rate | Verdict | Meaning |
|--------------|---------|---------|
| >= 99% | EXCELLENT | Gateway performing well under load |
| >= 95% | GOOD | Minor issues detected |
| >= 90% | FAIR | Some reliability concerns |
| < 90% | POOR | Significant issues detected |

### Key Metrics to Watch

- **P95/P99 Latency**: High values indicate tail latency issues
- **Throughput**: Should match or exceed target request rate
- **Error Types**: Connection errors vs timeout errors vs HTTP errors
- **Success Rate**: Should be > 95% for production readiness
