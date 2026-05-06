# Marie-AI Stress Testing Tools

This directory contains stress testing tools for testing the Marie gateway and networking components.

## Tools

### 1. `gateway_e2e_stresser.py`

End-to-end scheduler stress harness for real document jobs.

This tool is for validating the full path:

- local file discovery or pre-staged S3 selection
- optional S3/MinIO upload
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
- **Scheduler event tracking**: listens for `*.scheduled`, `*.started`, `*.completed`, `*.failed`
- **Companion metadata support**: uploads `<file>.meta.json` sidecars when present
- **Finite load model**: submits a configurable number of jobs at a target rate
- **Latency breakdowns**: submit, scheduling, queue wait, execution, and end-to-end timing
- **AIMock fault profile integration**: can switch the mock backend into `normal`, `timeout`, `error`, or randomized `chaos`

#### Usage

```bash
# Full end-to-end extract test using the existing grapnel config
python tools/stress/gateway_e2e_stresser.py \
    --config /home/gbugaj/dev/workflow/grapnel-g5/config.dev.json \
    --input-dir /mnt/data/marie-ai/generators \
    --job-count 25 \
    --job-name gen5_extract \
    --planner extract

# Submit TIFFs at 4 jobs/sec and write a JSON report
python tools/stress/gateway_e2e_stresser.py \
    --config /home/gbugaj/dev/workflow/grapnel-g5/config.dev.json \
    --s3-uri-manifest /tmp/stress-s3-uris.txt \
    --job-count 50 \
    --job-name gen5_extract \
    --planner extract \
    --submit-rate 4 \
    --report-json /tmp/gateway-e2e-report.json

# Run against AIMock/LiteLLM with randomized chaos mode
python tools/stress/gateway_e2e_stresser.py \
    --config /home/gbugaj/dev/workflow/grapnel-g5/config.dev.json \
    --s3-uri 's3://marie/gen5_extract/sample-001.tif' \
    --job-count 10 \
    --job-name gen5_extract \
    --planner extract \
    --fault-profile chaos \
    --aimock-admin-url http://localhost:4011
```

#### Quick Start

The usual scheduler-stress path is:

1. start the mock LLM backend
2. start LiteLLM pointed at that mock backend
3. make sure gateway + RabbitMQ + MinIO/S3 are already running
4. run `gateway_e2e_stresser.py` against pre-staged `s3://` assets

Create the shared Docker network once:

```bash
docker network create --driver=bridge marie_default 2>/dev/null || true
```

Start the programmatic AIMock backend with admin control enabled:

```bash
cd /home/gbugaj/dev/marieai/marie-ai/Dockerfiles
docker compose -f docker-compose.mock-llm-programmatic.yml up -d

# Verify the admin endpoint
curl http://localhost:4011/fault-profile
```

Start LiteLLM and point it at the programmatic AIMock service on the Docker network:

```bash
cd /home/gbugaj/dev/marieai/marie-ai
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
cd /home/gbugaj/dev/marieai/marie-ai
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

Upload mode:

- use `--input-dir`, `--input-glob`, or `--input-manifest`
- the tool uploads local files to S3/MinIO before submission
- use this when you want a true end-to-end ingest benchmark

Pre-staged S3 mode:

- use `--s3-uri` or `--s3-uri-manifest`
- the tool skips upload and submits directly against existing `s3://` objects
- use this when you want to isolate scheduler, queueing, LiteLLM, and failure behavior

#### Sample `--s3-uri-manifest`

Create a simple text file with one `s3://` URI per line:

```text
s3://marie/gen5_extract/sample-001.tif
s3://marie/gen5_extract/sample-002.tif
s3://marie/gen5_extract/sample-003.tif
```

Example:

```bash
cat >/tmp/stress-s3-uris.txt <<'EOF'
s3://marie/gen5_extract/sample-001.tif
s3://marie/gen5_extract/sample-002.tif
s3://marie/gen5_extract/sample-003.tif
EOF
```

#### Common Runs

Scheduler and LLM stress against pre-staged S3 assets:

```bash
source ~/environments/marie-3.12/bin/activate

python /home/gbugaj/dev/marieai/marie-ai/tools/stress/gateway_e2e_stresser.py \
  --config /home/gbugaj/dev/workflow/grapnel-g5/config.dev.json \
  --s3-uri-manifest /tmp/stress-s3-uris.txt \
  --job-count 1000 \
  --job-name gen5_extract \
  --planner extract \
  --submit-rate 4 \
  --fault-profile normal \
  --aimock-admin-url http://localhost:4011 \
  --report-json /tmp/gateway-e2e-report.json
```

Timeout-profile run:

```bash
python /home/gbugaj/dev/marieai/marie-ai/tools/stress/gateway_e2e_stresser.py \
  --config /home/gbugaj/dev/workflow/grapnel-g5/config.dev.json \
  --s3-uri-manifest /tmp/stress-s3-uris.txt \
  --job-count 250 \
  --job-name gen5_extract \
  --planner extract \
  --submit-rate 2 \
  --fault-profile timeout \
  --aimock-admin-url http://localhost:4011 \
  --report-json /tmp/gateway-timeout-report.json
```

Randomized monkey/chaos run:

```bash
python /home/gbugaj/dev/marieai/marie-ai/tools/stress/gateway_e2e_stresser.py \
  --config /home/gbugaj/dev/workflow/grapnel-g5/config.dev.json \
  --s3-uri-manifest /tmp/stress-s3-uris.txt \
  --job-count 500 \
  --job-name gen5_extract \
  --planner extract \
  --submit-rate 3 \
  --fault-profile chaos \
  --aimock-admin-url http://localhost:4011 \
  --report-json /tmp/gateway-chaos-report.json
```

True full-pipeline run with local upload:

```bash
python /home/gbugaj/dev/marieai/marie-ai/tools/stress/gateway_e2e_stresser.py \
  --config /home/gbugaj/dev/workflow/grapnel-g5/config.dev.json \
  --input-dir /mnt/data/marie-ai/generators \
  --job-count 100 \
  --job-name gen5_extract \
  --planner extract \
  --submit-rate 2 \
  --report-json /tmp/gateway-upload-report.json
```

#### Important options

| Option | Description |
|--------|-------------|
| `--config` | Grapnel-style JSON config with `api_base_url`, `api_key`, `storage`, and `queue` |
| `--input-dir` / `--input-glob` / `--input-manifest` | Local source file discovery for upload mode |
| `--s3-uri` / `--s3-uri-manifest` | Pre-staged S3 objects for submit-only mode |
| `--job-count` | Number of jobs to submit |
| `--job-name` | Gateway submit name / scheduler queue name |
| `--planner` | Planner to place in metadata |
| `--fault-profile` | Run label and AIMock control target: `normal`, `timeout`, `error`, `chaos` |
| `--aimock-admin-url` | AIMock admin endpoint used to switch the active fault profile before the run |
| `--submit-concurrency` | Concurrent upload+submit workers |
| `--submit-rate` | Job submit rate in jobs/sec |
| `--terminal-timeout` | Max wait for terminal scheduler events |
| `--request-template` | JSON file containing metadata or a full `invoke_action` template |
| `--report-json` | Write structured report for later analysis |

#### Output

The report breaks timing into:

- **submit latency**: gateway `job submit` request/response
- **scheduling latency**: submit response to `*.scheduled`
- **queue wait**: submit response to `*.started` (or `*.scheduled` when no start event exists)
- **execution latency**: `*.started` to terminal event
- **end-to-end latency**: submit start to terminal event

This is the primary tool to use when intentionally restarting LiteLLM,
annotators, or other downstream services to see how scheduler outcomes change.

For scheduler and LLM fault testing, prefer:

- `--s3-uri` or `--s3-uri-manifest` when the asset is already staged and you want to isolate queueing, scheduling, submit, and failure behavior
- `--fault-profile chaos` when using the programmatic AIMock stack

`chaos` is the randomized monkey-test mode in this setup.

#### Stopping The Mock Stack

```bash
cd /home/gbugaj/dev/marieai/marie-ai
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

### 3. `networking_stresser.py`

A comprehensive networking stress tester with chaos engineering capabilities.

#### Features

- **Executor simulation**: Simulates N executor servers
- **Chaos controller**: Randomly cycles executors up/down
- **Multiple test modes**: Circuit breaker, load balancer, chaos, flood testing
- **Real-time metrics**: Periodic reporting during test execution

#### Usage

```bash
# Full integration test with 5 executors
python networking_stresser.py --duration 120

# Test with 10 executors and aggressive chaos
python networking_stresser.py --num-executors 10 \
    --chaos-interval-min 2 --chaos-interval-max 5

# Test circuit breaker behavior
python networking_stresser.py --mode circuit_breaker_test

# High request rate testing
python networking_stresser.py --request-rate 100 --mode request_flood

# Disable chaos (stable environment)
python networking_stresser.py --no-chaos --duration 60
```

#### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--gateway-host` | `localhost` | Gateway host |
| `--gateway-port` | `52000` | Gateway port |
| `--num-executors` | `5` | Number of simulated executors |
| `--executor-base-port` | `54000` | Base port for executors |
| `--no-chaos` | `False` | Disable chaos (executor up/down) |
| `--chaos-interval-min` | `5` | Min seconds between chaos events |
| `--chaos-interval-max` | `15` | Max seconds between chaos events |
| `--chaos-down-duration-min` | `3` | Min seconds executor stays down |
| `--chaos-down-duration-max` | `10` | Max seconds executor stays down |
| `--chaos-max-down-ratio` | `0.5` | Max ratio of executors that can be down |
| `--request-rate` | `10` | Requests per second |
| `--duration` | `60` | Test duration in seconds |
| `--mode` | `full_integration` | Test mode |

#### Test Modes

- `circuit_breaker_test`: Tests circuit breaker behavior under failure conditions
- `load_balancer_test`: Tests load distribution across replicas
- `chaos_test`: Aggressive chaos testing with frequent executor failures
- `full_integration`: Complete integration test with all features
- `request_flood`: High request rate stress testing

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
