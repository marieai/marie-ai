---
sidebar_position: 12
---

# LLM tracking and observability

Marie-AI exposes LLM and runtime observability through `marie.instrumentation`, using OpenTelemetry for transport and OpenInference for LLM/agent span semantics. The standard runtime path is OTLP collector to ClickHouse.

## Architecture overview

The current tracking system follows this path:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        LLM Tracking Architecture                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Application                                                                │
│   ┌─────────────┐                                                            │
│   │ Tracker /   │──► OpenTelemetry + OpenInference spans                     │
│   │ Tracer API  │                                                            │
│   └──────┬──────┘                                                            │
│          │                                                                   │
│          ▼                                                                   │
│   OTLP Collector                                                             │
│   ┌─────────────┐                                                            │
│   │ Receive     │──► enrich / batch / transform                              │
│   │ OTLP data   │                                                            │
│   └──────┬──────┘                                                            │
│          │                                                                   │
│          ▼                                                                   │
│   ClickHouse                                                                 │
│   ┌─────────────┐                                                            │
│   │ otel_traces │                                                            │
│   │ otel_logs   │                                                            │
│   │ otel_metrics│                                                            │
│   └─────────────┘                                                            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Components

| Component | Purpose |
|-----------|---------|
| **`marie.instrumentation`** | Captures traces, observations, and spans from LLM and agent flows |
| **OpenInference** | Standardizes LLM, agent, tool, and retriever attributes |
| **OTLP collector** | Receives, batches, and exports telemetry |
| **ClickHouse** | Analytics database for dashboards and queries |
| **Console exporter** | Local debugging path when you do not want to send OTLP |

### Data flow

1. The application creates spans through `get_tracker()` or `get_tracer()`
2. `marie.instrumentation` applies OpenInference attributes to the emitted spans
3. The runtime exports OTLP data to the collector
4. The collector batches and transforms telemetry as needed
5. ClickHouse stores traces, logs, and metrics for analytics

## Configuration

Configure LLM tracking in your YAML config file:

```yaml
llm_tracking:
  enabled: true
  exporter: otel  # or "console" for local debugging
  project_id: my-project
  debug: false
  console_spans: false
```

Register the OTLP exporter in process startup:

```python
from marie.instrumentation import configure_from_yaml, register

configure_from_yaml(config["llm_tracking"])
register(
    project_name="marie-prod",
    endpoint="http://localhost:4317",
    batch=True,
)
```

Or use the standard environment variable:

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
```

## Usage

### Basic tracking

```python
from marie.instrumentation import get_tracker

# Get the singleton tracker instance
tracker = get_tracker()

# Create a trace for a user request
trace_id = tracker.create_trace(
    name="document-processing",
    user_id="user-123",
    session_id="session-456",
    metadata={"document_type": "invoice"},
)

# Start tracking an LLM generation
observation_id = tracker.generation(
    trace_id=trace_id,
    name="extract-fields",
    model="gpt-4",
    input={"prompt": "Extract invoice fields..."},
)

# End the generation with output
tracker.end(
    observation_id=observation_id,
    output={"fields": {"amount": "$500", "date": "2024-01-15"}},
    usage={
        "prompt_tokens": 150,
        "completion_tokens": 50,
        "total_tokens": 200,
    },
)

# Update trace with final results
tracker.update_trace(
    trace_id=trace_id,
    output={"status": "success", "fields_extracted": 5},
    metadata={"latency_seconds": 2.5},
)
```

### Error handling

```python
try:
    # LLM call that might fail
    response = llm_client.complete(prompt)
    tracker.end(observation_id, output=response)
except Exception as e:
    # Track the error
    tracker.error(
        observation_id=observation_id,
        error=e,
        level="ERROR",
    )
```

### Spans for non-LLM operations

```python
# Track non-LLM operations (database queries, API calls, etc.)
span_id = tracker.span(
    trace_id=trace_id,
    name="fetch-document",
    input={"document_id": "doc-123"},
)

# ... perform operation ...

tracker.end(
    observation_id=span_id,
    output={"document_size": 1024},
)
```

## ClickHouse storage

The production observability path writes into ClickHouse OTel tables through the collector.

Common tables:

| Table | Purpose |
|-------|---------|
| `otel.otel_traces` | Trace and span storage |
| `otel.otel_logs` | Application and runtime logs |
| `otel.otel_metrics_gauge` | Point-in-time metrics |
| `otel.otel_metrics_sum` | Counters and cumulative metrics |
| `otel.otel_metrics_histogram` | Latency and distribution metrics |

The collector wiring for these tables lives in `config/clickstack/otel-collector-config.yml`.

## Runtime wiring

The current implementation lives under `marie.instrumentation` and emits OpenTelemetry/OpenInference data.

### Collector path

The standard runtime path is:

```text
application
  -> marie.instrumentation
  -> OTLP collector
  -> ClickHouse
```

The collector wiring in this repo lives in `config/clickstack/otel-collector-config.yml`, and the local stack reference is documented in `config/clickstack/README.md`.

## Analytics queries

Query ClickHouse directly for traces and usage analytics:

```sql
-- Slowest spans in the last hour
SELECT
    ServiceName,
    SpanName,
    TraceId,
    Duration / 1000000 AS duration_ms
FROM otel.otel_traces
WHERE Timestamp > now() - INTERVAL 1 HOUR
ORDER BY Duration DESC
LIMIT 20;

-- Error spans by service
SELECT
    ServiceName,
    count() AS error_count
FROM otel.error_traces_mv
WHERE Timestamp > now() - INTERVAL 1 HOUR
GROUP BY ServiceName
ORDER BY error_count DESC;

-- Request latency histogram series
SELECT
    toStartOfMinute(TimeUnix) AS minute,
    sum(Count) AS requests,
    avg(Sum / Count) * 1000 AS avg_latency_ms
FROM otel.otel_metrics_histogram
WHERE MetricName = 'marie_gateway_request_seconds'
  AND TimeUnix > now() - INTERVAL 1 HOUR
GROUP BY minute
ORDER BY minute DESC;
```

## Cleanup

Retention is primarily controlled at the collector and ClickHouse layer. The local collector config already sets a TTL on ClickHouse exports:

```yaml
exporters:
  clickhouse:
    ttl: 72h
```

If your application stores large payloads outside the telemetry path, manage those separately with normal object-store lifecycle rules.

## Troubleshooting

### No spans arriving in ClickHouse

Check that the OTLP collector is up and reachable:

```bash
curl http://localhost:13133/health
```

Common causes:
- `OTEL_EXPORTER_OTLP_ENDPOINT` points at the wrong host or port
- the collector is not running
- ClickHouse is unavailable to the collector

### Console output only

If you only want local debugging output, use the console exporter:

```yaml
llm_tracking:
  enabled: true
  exporter: console
```

## Exporter types

### Console exporter (development)

For local development and debugging:

```yaml
llm_tracking:
  enabled: true
  exporter: console
```

Events are printed locally instead of being exported through OTLP.

### OTel exporter (production)

For the current production path:

```yaml
llm_tracking:
  enabled: true
  exporter: otel
```

Export the spans to the collector with `register(..., endpoint=...)` or the standard OTLP environment variables.

## Integration guidance

For application code, prefer one of these patterns:

- Use `get_tracer()` for new decorator-based instrumentation around agents, tools, and chains.
- Use `get_tracker()` when you need the imperative trace and observation API.
- Register OTLP once during process startup so all spans flow to the same collector and ClickHouse backend.
