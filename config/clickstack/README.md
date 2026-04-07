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

## Connection Credentials

### ClickHouse

| Setting | Value |
|---------|-------|
| Host | `localhost` (external) / `marie-clickhouse` (Docker) |
| HTTP Port | `8123` |
| Native Port | `9000` |
| Database | `otel` |
| Username | `marie` |
| Password | `marie123` |

**JDBC URL:**
```
jdbc:clickhouse://localhost:8123/otel?user=marie&password=marie123
```

**CLI:**
```bash
docker exec -it marie-clickhouse clickhouse-client -u marie --password marie123 -d otel
```

### HyperDX UI

| Setting | Value |
|---------|-------|
| URL | http://localhost:8080 |
| Email | `marie@marie.local` |
| Password | `MarieAI@2026!` |

## OTel Tables

| Table | Description |
|-------|-------------|
| `otel_logs` | Application and system logs |
| `otel_traces` | Distributed traces and spans |
| `otel_metrics_gauge` | Point-in-time metric values |
| `otel_metrics_sum` | Cumulative/delta counters |
| `otel_metrics_histogram` | Histogram distributions |
| `otel_metrics_exponential_histogram` | High-cardinality histograms |
| `otel_metrics_summary` | Summary statistics |
| `error_logs_mv` | Materialized view: ERROR/FATAL logs |
| `error_traces_mv` | Materialized view: Failed spans |

## Useful Queries

### Recent Errors
```sql
SELECT * FROM otel.error_logs_mv
ORDER BY Timestamp DESC
LIMIT 100;
```

### Error Count by Service (Last Hour)
```sql
SELECT ServiceName, count() as error_count
FROM otel.error_logs_mv
WHERE Timestamp > now() - INTERVAL 1 HOUR
GROUP BY ServiceName
ORDER BY error_count DESC;
```

### Slowest Traces (Last Hour)
```sql
SELECT ServiceName, SpanName, TraceId, Duration/1000000 as duration_ms
FROM otel.otel_traces
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
FROM otel.otel_metrics_histogram
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
FROM otel.otel_metrics_gauge
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
```bash
# Verify marie user exists
docker exec marie-clickhouse clickhouse-client \
  --query "SELECT name FROM system.users"

# Re-run bootstrap if needed
docker exec -i marie-clickhouse clickhouse-client \
  < config/clickhouse/init-clickhouse.sql
```

### Reset ClickHouse (WARNING: deletes all data)
```bash
docker compose -f ./Dockerfiles/docker-compose.clickhouse.yml down --volumes
docker volume rm marie_clickhouse_data marie_clickhouse_logs 2>/dev/null
docker compose --env-file ./config/.env.dev \
  -f ./Dockerfiles/docker-compose.clickhouse.yml \
  --project-directory . up -d
```

## Data Retention

All tables have a 30-day TTL configured. To modify:

```sql
ALTER TABLE otel.otel_logs
MODIFY TTL toDateTime(Timestamp) + INTERVAL 90 DAY;
```

## Related Files

- `Dockerfiles/docker-compose.clickhouse.yml` - ClickHouse container
- `Dockerfiles/docker-compose.clickstack.yml` - HyperDX + OTel Collector
- `config/.env.dev` - Environment variables (credentials)
- `config/clickhouse/init-clickhouse.sql` - Bootstrap script
