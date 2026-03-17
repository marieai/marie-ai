# Log Collector -- Docker Container Log Collection

## Overview

The log collector is an OpenTelemetry Collector sidecar that ingests Docker container logs and exports them to any OTLP-compatible backend. It uses the `otel/opentelemetry-collector-contrib:0.114.0` image with a Marie-AI-specific configuration that parses structured fields from Marie-AI log output.

**Data flow:**

```
Docker containers -> JSON logs -> OTel Collector -> OTLP backend (HyperDX, Grafana, etc.)
```

The collector reads Docker JSON log files directly from `/var/lib/docker/containers/`, parses the Marie-AI log format, extracts structured fields (job IDs, statuses, durations, components), and exports enriched logs via OTLP HTTP.

## Quick Start

```bash
docker run -d --name log-collector \
  -v /var/lib/docker/containers:/var/lib/docker/containers:ro \
  -v /var/run/docker.sock:/var/run/docker.sock:ro \
  -v ./config/clickstack/log-collector-config.yml:/etc/otel/config.yml:ro \
  -e OTEL_EXPORTER_OTLP_ENDPOINT=http://your-otlp-backend:4318 \
  -e DEPLOYMENT_ENV=production \
  --user 0:0 \
  otel/opentelemetry-collector-contrib:0.114.0 \
  --config=/etc/otel/config.yml
```

The collector must run as root (`--user 0:0`) to read Docker container log files.

Verify it is running:

```bash
# Health check endpoint
curl http://localhost:13133/

# View collector metrics
curl http://localhost:8888/metrics
```

## What Gets Collected

### Sources

- **Docker JSON logs** from all containers at `/var/lib/docker/containers/*/*.log`
- Containers named `*hyperdx*` and `*otel*` are excluded to avoid feedback loops

### Multiline Support

Log lines starting with `INFO`, `WARN`, `ERROR`, `DEBUG`, `CRITICAL`, or `FATAL` are treated as the start of a new log entry. Continuation lines (stack traces, multi-line messages) are joined to the preceding entry.

### Processing Pipeline

1. **Docker JSON parser** -- Extracts timestamp and log message from Docker's JSON log wrapper
2. **Container ID extraction** -- Parses container ID from the log file path
3. **Marie-AI log parser** -- Extracts `log_level`, `log_timestamp`, `component`, `thread_id`, and `message` from the Marie-AI log format
4. **Field extractors** -- Parse job_id, status, executor, duration, DAG info, and other structured fields from log messages
5. **Severity mapping** -- Maps `log_level` to OpenTelemetry severity levels
6. **Resource enrichment** -- Sets `service.name` from component, copies job_id and executor to resource attributes
7. **Batching** -- Groups logs (batch size 500, max 1000, timeout 5s) for efficient export

### Memory Limits

The collector is configured with a 400 MiB memory limit and 100 MiB spike buffer. Docker resource limits default to 512 MiB (configurable via `LOG_COLLECTOR_MEMORY_LIMIT`).

## Custom Configuration

Override the default configuration by mounting your own config file:

```bash
docker run -d --name log-collector \
  -v /var/lib/docker/containers:/var/lib/docker/containers:ro \
  -v /var/run/docker.sock:/var/run/docker.sock:ro \
  -v /path/to/custom-config.yml:/etc/otel/config.yml:ro \
  -e OTEL_EXPORTER_OTLP_ENDPOINT=http://backend:4318 \
  --user 0:0 \
  otel/opentelemetry-collector-contrib:0.114.0 \
  --config=/etc/otel/config.yml
```

The default configuration is at `config/clickstack/log-collector-config.yml` in the Marie-AI repository.

## Using with HyperDX

In the Marie-AI all-in-one deployment, the log collector is preconfigured to export to HyperDX. For standalone use with an existing HyperDX instance:

```bash
docker run -d --name log-collector \
  -v /var/lib/docker/containers:/var/lib/docker/containers:ro \
  -v /var/run/docker.sock:/var/run/docker.sock:ro \
  -v ./config/clickstack/log-collector-config.yml:/etc/otel/config.yml:ro \
  -e OTEL_EXPORTER_OTLP_ENDPOINT=http://hyperdx-host:4318 \
  --user 0:0 \
  otel/opentelemetry-collector-contrib:0.114.0 \
  --config=/etc/otel/config.yml
```

If HyperDX is on the same Docker network:

```bash
docker run -d --name log-collector \
  --network marie_default \
  -v /var/lib/docker/containers:/var/lib/docker/containers:ro \
  -v /var/run/docker.sock:/var/run/docker.sock:ro \
  -v ./config/clickstack/log-collector-config.yml:/etc/otel/config.yml:ro \
  -e OTEL_EXPORTER_OTLP_ENDPOINT=http://marie-hyperdx:4318 \
  --user 0:0 \
  otel/opentelemetry-collector-contrib:0.114.0 \
  --config=/etc/otel/config.yml
```

## Using with Other Backends

### Grafana / Loki

Modify the exporter in the config to use `loki` instead of `otlphttp`:

```yaml
exporters:
  loki:
    endpoint: http://loki-host:3100/loki/api/v1/push
    labels:
      attributes:
        log_level: ""
        component: ""
        job_status: ""
```

Replace the exporter in the pipeline:

```yaml
service:
  pipelines:
    logs:
      receivers: [filelog/docker]
      processors: [memory_limiter, resource, transform, batch]
      exporters: [loki]
```

### Datadog

Use the `datadog` exporter:

```yaml
exporters:
  datadog:
    api:
      key: ${DD_API_KEY}
      site: datadoghq.com

service:
  pipelines:
    logs:
      receivers: [filelog/docker]
      processors: [memory_limiter, resource, transform, batch]
      exporters: [datadog]
```

### Generic OTLP Backend

The default `otlphttp` exporter works with any OTLP-compatible backend. Set the endpoint via environment variable:

```bash
-e OTEL_EXPORTER_OTLP_ENDPOINT=http://your-backend:4318
```

## Marie-AI Log Fields

The log collector parses the Marie-AI log format and extracts structured fields:

```
INFO   2026-01-12 14:59:55,272:            : PostgreSQLJobScheduler[]@ 7 Message here
```

### Core Fields

| Field | Description | Example |
|-------|-------------|---------|
| `log_level` | Log severity | `INFO`, `ERROR`, `DEBUG`, `WARN`, `CRITICAL` |
| `log_timestamp` | Event timestamp | `2026-01-12 14:59:55,272` |
| `component` | Source component | `PostgreSQLJobScheduler`, `JobManager`, `JobSupervisor` |
| `thread_id` | Thread identifier | `7` |
| `message` | Log message body | (full text after parsing) |

### Job Tracking Fields

| Field | Description | Example |
|-------|-------------|---------|
| `job_id` | Job UUID | `069650b6-360c-7d34-8000-d565715a63a1` |
| `submission_id` | Submission UUID | `069650b6-360c-7d34-8000-d565715a63a1` |
| `job_status` | Job lifecycle state | `completed`, `failed`, `scheduled`, `started`, `succeeded`, `enqueued` |
| `executor_name` | Executor handling the job | `corr_routing_executor`, `patient_indexing_executor` |
| `entrypoint` | Executor endpoint | `corr_routing_executor://document/classify` |

### Performance Fields

| Field | Description | Example |
|-------|-------------|---------|
| `duration_seconds` | Processing time in seconds | `0.26`, `2.65` |
| `signal_time` | Signal processing time | `0.011` |
| `post_signal_time` | Post-signal processing time | `0.010` |
| `etcd_lease_time` | etcd lease acquisition time | `0.005` |
| `etcd_put_time` | etcd put operation time | `0.003` |
| `etcd_total_time` | Total etcd operation time | `0.008` |

### Scheduler Fields

| Field | Description | Example |
|-------|-------------|---------|
| `dag_id` | DAG identifier (UUID) | `069650b6-...` |
| `dag_status` | DAG lifecycle state | `completed` |
| `scheduled_count` | Jobs scheduled in batch | `5` |
| `candidate_count` | Candidate jobs considered | `12` |
| `used_slots` | Currently used executor slots | `3` |
| `total_slots` | Total available executor slots | `8` |

### Network Fields

| Field | Description | Example |
|-------|-------------|---------|
| `target_host` | Request target IP | `192.168.1.10` |
| `target_port` | Request target port | `51000` |
| `target_deployment` | Target deployment name | `gateway` |

### Event Fields

| Field | Description | Example |
|-------|-------------|---------|
| `event_type` | Event classification | `corr.completed` |
| `api_key` | API key prefix (first 10 chars) | `sk-1234abcd` |
| `planner_name` | Planner used | `default` |

## HyperDX Search Examples

Once logs are flowing into HyperDX, use these search queries:

```
# Find all errors
severity:error

# Find job failures
job_status:failed OR job_status:FAILED

# Find a specific job by UUID
job_id:069650b6-360c-7d34-8000-d565715a63a1

# Find by component
component:PostgreSQLJobScheduler
component:JobManager
component:JobSupervisor

# Find by executor
executor_name:corr_routing_executor

# Find slow jobs (>1 second)
duration_seconds:>1

# Find DAG completions
dag_status:completed

# Find scheduler summaries
message:"Scheduling summary"

# Find gateway requests
component:gateway

# Combine filters
severity:error AND component:JobManager
executor_name:corr_routing_executor AND job_status:failed
```

## Troubleshooting

### Collector not receiving logs

Check that Docker is using the `json-file` logging driver (the default):

```bash
docker info --format '{{.LoggingDriver}}'
# Should output: json-file
```

Verify log files exist:

```bash
ls /var/lib/docker/containers/*/
```

### Collector health check failing

```bash
# Check collector logs
docker logs log-collector

# Test health endpoint
curl http://localhost:13133/
```

### High memory usage

Reduce batch size and memory limits in the config:

```yaml
processors:
  batch:
    send_batch_size: 200
    send_batch_max_size: 500
    timeout: 10s
  memory_limiter:
    limit_mib: 256
    spike_limit_mib: 50
```

### Logs not appearing in backend

Enable the debug exporter to see what the collector is processing:

```yaml
exporters:
  debug:
    verbosity: detailed

service:
  pipelines:
    logs:
      exporters: [otlphttp, debug]
```

Then check collector logs:

```bash
docker logs -f log-collector
```
