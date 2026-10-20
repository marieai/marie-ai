# ClickStack - Observability Stack for Marie AI

ClickStack provides logs, traces, and metrics collection using OpenTelemetry and ClickHouse.

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Marie-AI      │     │  OTel Collector │     │   ClickHouse    │
│   Gateway       │────▶│   (HyperDX)     │────▶│   (otel db)     │
│   Executors     │     │   Port 4317     │     │   Port 8123     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
                                                ┌─────────────────┐
                                                │   HyperDX UI    │
                                                │   Port 8080     │
                                                └─────────────────┘
```

## Directory Structure

```
config/
├── clickhouse/
│   └── init-clickhouse.sql    # Bootstrap: users + schema (runs on container init)
└── clickstack/
    ├── README.md              # This file
    ├── otel-collector-config.yml   # OTel Collector configuration
    ├── log-collector-config.yml    # Docker log collector sidecar
    └── schema/
        └── observability.sql       # OTel table definitions (reference)
```

## Quick Start

```bash
cd ~/dev/marieai/marie-ai

# Create network (if not exists)
docker network create --driver=bridge marie_default 2>/dev/null || true

# Start ClickHouse (includes bootstrap)
# NOTE: --project-directory . is REQUIRED because the compose file lives in
# Dockerfiles/ but volume mounts use paths relative to the repo root.
# Without it, Docker creates empty directories instead of mounting files.
docker compose --env-file ./config/.env.dev \
  -f ./Dockerfiles/docker-compose.clickhouse.yml \
  --project-directory . up -d

# Start ClickStack (HyperDX + OTel Collector)
docker compose --env-file ./config/.env.dev \
  -f ./Dockerfiles/docker-compose.clickstack.yml \
  --project-directory . up -d

# Verify services
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
```

## Full Prompt Trace Payloads

Runtime Fabric LLM dispatch spans intentionally store full prompt and completion
payloads in OpenInference `input.value` and `output.value` attributes. The local
collector is tuned for those larger spans:

- OTLP gRPC receive messages up to 128 MiB.
- Collector batches are capped at 200 telemetry items.
- The Marie Python SDK defaults trace export batches to 32 spans.

Override the SDK batch size only when you know the average prompt size:

```bash
OTEL_BSP_MAX_EXPORT_BATCH_SIZE=16
MARIE_OTEL_GRPC_MAX_MESSAGE_BYTES=134217728
```

## Connection Credentials

### ClickHouse

| Setting | Value |
|---------|-------|
| Host | `localhost` (external) / `marie-clickhouse` (Docker) |
| HTTP Port | `8123` |
| Native Port | `9000` |
| Database | `telemetry` |
| Username | `marie` |
| Password | Existing deployment credential |

**JDBC URL:**
```
jdbc:clickhouse://localhost:8123/telemetry
```

**CLI:**
```bash
docker exec -it marie-clickhouse clickhouse-client --user marie --database telemetry
```

### HyperDX UI

| Setting | Value |
|---------|-------|
| URL | http://localhost:8080 |
| Email | `marie@marie.local` |
| Password | Existing deployment credential |

## OTel Tables

| Table | Description |
|-------|-------------|
| `telemetry.logs` | Application and system logs |
| `telemetry.spans` | Distributed traces and spans |
| `telemetry.metrics_gauge` | Point-in-time metric values |
| `telemetry.metrics_sum` | Cumulative/delta counters |
| `telemetry.metrics_histogram` | Histogram distributions |
| `telemetry.metrics_exponential_histogram` | High-cardinality histograms |
| `telemetry.metrics_summary` | Summary statistics |
| `telemetry.error_logs` | Materialized view: ERROR/FATAL logs |
| `telemetry.error_spans` | Materialized view: Failed spans |

## Useful Queries

### Recent Errors
```sql
SELECT * FROM telemetry.error_logs
ORDER BY Timestamp DESC
LIMIT 100;
```

### Error Count by Service (Last Hour)
```sql
SELECT ServiceName, count() as error_count
FROM telemetry.error_logs
WHERE Timestamp > now() - INTERVAL 1 HOUR
GROUP BY ServiceName
ORDER BY error_count DESC;
```

### Slowest Traces (Last Hour)
```sql
SELECT ServiceName, SpanName, TraceId, Duration/1000000 as duration_ms
FROM telemetry.spans
WHERE Timestamp > now() - INTERVAL 1 HOUR
ORDER BY Duration DESC
LIMIT 20;
```

### Gateway Request Latency
```sql
SELECT
  toStartOfMinute(TimeUnix) as minute,
  sum(Count) as requests,
  avg(Sum / Count) * 1000 as avg_latency_ms
FROM telemetry.metrics_histogram
WHERE MetricName = 'marie_gateway_request_seconds'
  AND TimeUnix > now() - INTERVAL 1 HOUR
GROUP BY minute
ORDER BY minute DESC;
```

### Executor Slot Utilization
```sql
SELECT
  Attributes['executor'] as executor,
  max(Value) as capacity
FROM telemetry.metrics_gauge
WHERE MetricName = 'marie_executor_slot_capacity'
  AND TimeUnix > now() - INTERVAL 5 MINUTE
GROUP BY executor;
```

## Configuring Marie-AI Gateway

Enable metrics export in your gateway config:

```yaml
jtype: MarieGateway
with:
  metrics: true
  metrics_exporter_host: localhost  # or marie-hyperdx in Docker
  metrics_exporter_port: 4317

  tracing: true
  traces_exporter_host: localhost
  traces_exporter_port: 4317
```

Or via environment variables:
```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
export OTEL_SERVICE_NAME=marie-gateway
```

## Ports Reference

| Service | Port | Purpose |
|---------|------|---------|
| ClickHouse HTTP | 8123 | Query interface, JDBC |
| ClickHouse Native | 9000 | CLI client, high-perf |
| ClickHouse MySQL | 9004 | MySQL wire protocol |
| OTel Collector gRPC | 4317 | OTLP metric/trace ingestion |
| OTel Collector HTTP | 4318 | OTLP HTTP endpoint |
| HyperDX UI | 8080 | Web dashboard |
| OTel Health | 13133 | Health check endpoint |

## Troubleshooting

### No metrics in ClickHouse
```bash
# 1. Check gateway has metrics enabled
grep -E "metrics|exporter" config/service/gateway.yml

# 2. Check OTel collector is receiving data
docker logs marie-hyperdx 2>&1 | grep "otlp" | tail -10

# 3. Test collector health
curl http://localhost:13133/health
```

### Authentication failed

Use the configured system account (`marie` by default) and its existing deployment credential. The container entrypoint owns that identity. The initialization SQL does not elevate a limited SQL user; an old limited account cannot grant itself new privileges. Fresh storage must be initialized with the configured system identity. Preserve application/provider configuration when replacing telemetry storage.

### Reset ClickHouse (WARNING: deletes all data)
```bash
docker compose -f ./Dockerfiles/docker-compose.clickhouse.yml down --volumes
docker volume rm marie.clickhouse_data marie.clickhouse_logs 2>/dev/null
docker compose --env-file ./config/.env.dev \
  -f ./Dockerfiles/docker-compose.clickhouse.yml \
  --project-directory . up -d
```

## Data Retention

Retention is declared in `config/clickhouse/workspace-isolation/schema.sql` (raw signals 30 days, annotations 90 days, event counts 7 days). Change the reviewed schema and regenerate/qualify its manifest before altering deployed tenant tables. Ad hoc DDL can invalidate scoped-reader readiness.

## Related Files

- `Dockerfiles/docker-compose.clickhouse.yml` - ClickHouse container
- `Dockerfiles/docker-compose.clickstack.yml` - HyperDX + OTel Collector
- `config/.env.dev` - Environment variables (credentials)
- `config/clickhouse/init-clickhouse.sql` - Bootstrap script

## Fresh telemetry cutover

The canonical store is `telemetry`; `workspace_query` contains scoped INVOKER views. `marie` holds private system projections/pricing. The separate `query` namespace is the existing administrator-only shared SQL interface. These are namespaces on the same ClickHouse server, not additional database services. Do not create a Project hierarchy or treat legacy project attributes as trusted workspace identity.

Compose initialization and the existing `clickhouse-query-schema` service install the canonical schema and private system objects. The collector uses explicit canonical table names with schema creation disabled. Run schema updates before changing/restarting ingestion. The schema service reuses existing system credentials; the dictionary's server-side named collection reuses them internally, without another password setting.

Already persisted HyperDX sources do not follow new defaults automatically. After installing the schema, run Studio's `packages/api/src/operator/cutover-telemetry.ts --connection UUID` using its existing operator environment. It preserves source IDs/connections and policy records while updating supported legacy source mappings, then updates a stored Studio `otel` database setting to `telemetry`. Restart the API to clear cached settings. Inspect unsupported custom source mappings manually.

Do not drop `otel` until all writers, Studio readers and persisted provider sources have been retargeted and positively exercised, including actual alert evaluation and an isolated callback. `DROP DATABASE otel` is the final retirement step, not schema installation. Never drop HyperDX's state store, Studio Postgres, destinations or policies as part of fresh telemetry. Trusted tenant ingress and tenant activation remain separate release gates.
