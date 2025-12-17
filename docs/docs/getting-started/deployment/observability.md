---
sidebar_position: 4
---

# Observability

Monitor and troubleshoot Marie-AI deployments with a comprehensive observability stack. This guide covers metrics collection, log aggregation, tracing, and alerting.

## Overview

Marie-AI supports a full observability stack:

| Component | Purpose | Default Port |
|-----------|---------|--------------|
| Prometheus | Metrics collection and storage | 9090 |
| Grafana | Visualization and dashboards | 3000 |
| Loki | Log aggregation | 3100 |
| Jaeger | Distributed tracing | 16686 |
| Alertmanager | Alert routing and notifications | 9093 |

```text
┌─────────────────────────────────────────────────────────────────────┐
│                        Marie-AI Services                            │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐               │
│  │ Gateway │  │Executor │  │Executor │  │Executor │               │
│  │ :54322  │  │  :8001  │  │  :8002  │  │  :8003  │               │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘               │
│       │            │            │            │                     │
│       └────────────┴────────────┴────────────┘                     │
│                         │                                          │
└─────────────────────────┼──────────────────────────────────────────┘
                          │ metrics/logs/traces
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Observability Stack                              │
│  ┌──────────────┐  ┌──────────┐  ┌────────┐  ┌──────────────────┐  │
│  │ OTel         │  │Prometheus│  │ Loki   │  │     Jaeger       │  │
│  │ Collector    │──│  :9090   │  │ :3100  │  │     :16686       │  │
│  │ :4317/:4318  │  └────┬─────┘  └───┬────┘  └────────┬─────────┘  │
│  └──────────────┘       │            │               │             │
│                         └────────────┴───────────────┘             │
│                                      │                             │
│                              ┌───────┴───────┐                     │
│                              │   Grafana     │                     │
│                              │    :3000      │                     │
│                              └───────────────┘                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Quick start

Deploy the monitoring stack with Docker Compose:

```bash
# Create the network if it doesn't exist
docker network create --driver=bridge public

# Start the monitoring stack
docker-compose -f Dockerfiles/docker-compose.monitoring.yml up -d
```

Access the dashboards:

| Service | URL | Default Credentials |
|---------|-----|---------------------|
| Grafana | http://localhost:3000 | admin / (see config) |
| Prometheus | http://localhost:9090 | - |
| Jaeger | http://localhost:16686 | - |
| Loki | http://localhost:3100 | - |
| Alertmanager | http://localhost:9093 | - |

## Metrics with Prometheus

Prometheus collects and stores time-series metrics from Marie-AI services.

### Configuration

The Prometheus configuration defines scrape targets and alerting rules:

```yaml
# config/prometheus/prometheus.yml
global:
  scrape_interval: 20s
  scrape_timeout: 20s

scrape_configs:
  - job_name: 'otel-collector'
    scrape_interval: 500ms
    static_configs:
      - targets: ['otel-collector:8889']
      - targets: ['otel-collector:8888']

  # Add Marie-AI services
  - job_name: 'marie-gateway'
    static_configs:
      - targets: ['gateway:54322']

  - job_name: 'marie-executors'
    static_configs:
      - targets: ['executor-extract:8001', 'executor-ocr:8002']

alerting:
  alertmanagers:
    - scheme: http
      static_configs:
        - targets: ['alertmanager:9093']
```

### Available metrics

Marie-AI exposes metrics through the OpenTelemetry collector:

| Metric | Type | Description |
|--------|------|-------------|
| `marie_jobs_total` | Counter | Total jobs processed |
| `marie_jobs_active` | Gauge | Currently active jobs |
| `marie_jobs_failed_total` | Counter | Total failed jobs |
| `marie_job_duration_seconds` | Histogram | Job processing duration |
| `marie_executor_slots_used` | Gauge | Executor slot utilization |
| `marie_executor_slots_available` | Gauge | Available executor slots |

### Additional metric sources

| Source | Endpoint | Description |
|--------|----------|-------------|
| cAdvisor | http://localhost:8077/metrics | Container resource metrics |
| DCGM | http://localhost:9400/metrics | GPU metrics (NVIDIA) |
| Node Exporter | http://localhost:9100/metrics | Host system metrics |

### Example queries

Common PromQL queries for monitoring Marie-AI:

```promql
# Job throughput (jobs per minute)
rate(marie_jobs_total[5m]) * 60

# Average job duration
histogram_quantile(0.95, rate(marie_job_duration_seconds_bucket[5m]))

# Executor utilization percentage
marie_executor_slots_used / (marie_executor_slots_used + marie_executor_slots_available) * 100

# Failed job rate
rate(marie_jobs_failed_total[5m]) / rate(marie_jobs_total[5m]) * 100
```

## Grafana dashboards

Grafana provides visualization for metrics from Prometheus and logs from Loki.

### Setup

Grafana is configured through provisioning files:

```yaml
# config/grafana/provisioning/datasources/datasources.yml
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true

  - name: Loki
    type: loki
    access: proxy
    url: http://loki:3100
```

### Creating dashboards

Import the Marie-AI dashboard or create custom panels:

1. Open Grafana at http://localhost:3000
2. Go to **Dashboards** → **Import**
3. Paste the dashboard JSON or use a dashboard ID

Example panel configuration for job throughput:

```json
{
  "title": "Job Throughput",
  "type": "graph",
  "targets": [
    {
      "expr": "rate(marie_jobs_total[5m]) * 60",
      "legendFormat": "Jobs/min"
    }
  ]
}
```

### Recommended dashboard panels

| Panel | Query | Purpose |
|-------|-------|---------|
| Job throughput | `rate(marie_jobs_total[5m]) * 60` | Jobs processed per minute |
| Active jobs | `marie_jobs_active` | Current workload |
| Error rate | `rate(marie_jobs_failed_total[5m])` | Failure tracking |
| P95 latency | `histogram_quantile(0.95, ...)` | Performance monitoring |
| Executor slots | `marie_executor_slots_used` | Capacity utilization |

## Logging with Loki

Loki aggregates logs from all Marie-AI services for centralized analysis.

### Configuration

Loki stores logs locally with a 24-hour retention index:

```yaml
# config/grafana/loki/loki-config.yaml
auth_enabled: false

server:
  http_listen_port: 3100
  grpc_listen_port: 9096

common:
  path_prefix: /tmp/loki
  storage:
    filesystem:
      chunks_directory: /tmp/loki/chunks
      rules_directory: /tmp/loki/rules
  replication_factor: 1
  ring:
    instance_addr: 127.0.0.1
    kvstore:
      store: inmemory

schema_config:
  configs:
    - from: 2020-10-24
      store: boltdb-shipper
      object_store: filesystem
      schema: v11
      index:
        prefix: index_
        period: 24h

ruler:
  alertmanager_url: http://alertmanager:9093
```

### Sending logs to Loki

Configure Marie-AI services to send logs via Promtail or direct push:

```yaml
# docker-compose addition for promtail
promtail:
  image: grafana/promtail:latest
  volumes:
    - /var/log:/var/log
    - ./config/promtail:/etc/promtail
  command: -config.file=/etc/promtail/config.yml
```

### Querying logs

Use LogQL in Grafana to search logs:

```logql
# All logs from gateway
{service="marie-gateway"}

# Error logs only
{service=~"marie.*"} |= "ERROR"

# Job failures with job ID
{service="marie-scheduler"} |~ "job.*failed" | json | job_id != ""

# Logs from specific executor
{service="marie-executor", executor="extract"}
```

### Log labels

Recommended labels for Marie-AI logs:

| Label | Example | Description |
|-------|---------|-------------|
| `service` | `marie-gateway` | Service name |
| `executor` | `extract` | Executor type |
| `level` | `INFO`, `ERROR` | Log level |
| `job_id` | `uuid` | Job identifier |
| `dag_id` | `uuid` | DAG identifier |

## Distributed tracing with Jaeger

Jaeger provides distributed tracing for request flow visualization.

### Configuration

The OpenTelemetry collector forwards traces to Jaeger:

```yaml
# config/opentelemetry/otel-collector-config.yml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
      http:
        endpoint: 0.0.0.0:4318

exporters:
  otlp:
    endpoint: "jaeger:4317"
    tls:
      insecure: true

  prometheus:
    endpoint: "0.0.0.0:8889"
    resource_to_telemetry_conversion:
      enabled: true

processors:
  batch:

service:
  pipelines:
    traces:
      receivers: [otlp]
      exporters: [otlp]
      processors: [batch]
    metrics:
      receivers: [otlp]
      processors: [batch]
      exporters: [prometheus]
```

### Viewing traces

1. Open Jaeger UI at http://localhost:16686
2. Select service (e.g., `marie-gateway`)
3. Search for traces by operation or tags
4. Analyze span timing and dependencies

### Trace context

Marie-AI propagates trace context through:
- gRPC metadata
- HTTP headers (`traceparent`, `tracestate`)
- Job metadata for async operations

## Alerting

Alertmanager routes alerts from Prometheus to notification channels.

### Configuration

```yaml
# config/alertmanager/alertmanager.yml
route:
  receiver: 'default'
  repeat_interval: 4h
  group_by: [alertname]
  routes:
    - match:
        severity: critical
      receiver: 'critical'

receivers:
  - name: 'default'
    email_configs:
      - smarthost: 'smtp.example.com:587'
        auth_username: 'alerts@example.com'
        auth_password: '${SMTP_PASSWORD}'
        from: 'alerts@example.com'
        to: 'team@example.com'

  - name: 'critical'
    email_configs:
      - to: 'oncall@example.com'
    # Add PagerDuty, Slack, etc.
```

### Alert rules

Define alert rules in Prometheus:

```yaml
# config/prometheus/alert_rules.yml
groups:
  - name: marie-ai
    rules:
      - alert: HighJobFailureRate
        expr: rate(marie_jobs_failed_total[5m]) / rate(marie_jobs_total[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High job failure rate"
          description: "Job failure rate is above 10%"

      - alert: ExecutorCapacityLow
        expr: marie_executor_slots_available < 2
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Low executor capacity"
          description: "Less than 2 executor slots available"

      - alert: JobProcessingSlowdown
        expr: histogram_quantile(0.95, rate(marie_job_duration_seconds_bucket[5m])) > 300
        for: 15m
        labels:
          severity: critical
        annotations:
          summary: "Job processing slowdown"
          description: "P95 job duration exceeds 5 minutes"
```

## Health checks

Marie-AI exposes health endpoints for monitoring service status.

### Gateway health

```bash
# Liveness check
curl http://localhost:54322/health

# Readiness check
curl http://localhost:54322/ready

# Detailed status
curl http://localhost:54322/api/debug
```

### Executor health

Each executor exposes health endpoints:

```bash
curl http://localhost:8001/health
```

### Kubernetes probes

Configure health probes in Kubernetes deployments:

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 54322
  initialDelaySeconds: 30
  periodSeconds: 10

readinessProbe:
  httpGet:
    path: /ready
    port: 54322
  initialDelaySeconds: 5
  periodSeconds: 5
```

## Kubernetes monitoring

For Kubernetes deployments, use the Prometheus Operator for automatic discovery.

### Installing the stack

```bash
helm install prometheus prometheus-community/kube-prometheus-stack \
  --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false
```

### ServiceMonitor

Create a ServiceMonitor to scrape Marie-AI metrics:

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: marie-ai
  labels:
    app: marie-ai
spec:
  selector:
    matchLabels:
      app: marie-ai
  endpoints:
    - port: metrics
      interval: 15s
```

### Pod annotations

Alternatively, use annotations for Prometheus discovery:

```yaml
metadata:
  annotations:
    prometheus.io/scrape: "true"
    prometheus.io/port: "8000"
    prometheus.io/path: "/metrics"
```

## Troubleshooting with observability

### Common issues

| Symptom | Check | Solution |
|---------|-------|----------|
| No metrics | Prometheus targets | Verify service is reachable |
| Missing logs | Promtail targets | Check log path configuration |
| No traces | Collector logs | Verify OTLP endpoint |
| Alerts not firing | Alertmanager status | Check route configuration |

### Debug endpoints

```bash
# Prometheus targets
curl http://localhost:9090/api/v1/targets

# Promtail targets
curl http://localhost:9080/targets

# Alertmanager status
curl http://localhost:9093/api/v1/status
```

### Log levels

Increase logging verbosity for debugging:

```yaml
# In docker-compose or deployment
environment:
  - MARIE_LOG_LEVEL=DEBUG
  - PROMETHEUS_LOG_LEVEL=debug
```

## Production considerations

### Storage

Configure persistent storage for production:

```yaml
volumes:
  prometheus_data:
    driver: local
    driver_opts:
      type: none
      device: /data/prometheus
      o: bind

  loki_data:
    driver: local
    driver_opts:
      type: none
      device: /data/loki
      o: bind
```

### Retention

Configure data retention policies:

```yaml
# Prometheus
command: >
  --storage.tsdb.retention.time=30d
  --storage.tsdb.retention.size=50GB

# Loki
limits_config:
  retention_period: 30d
```

### High availability

For production deployments:

1. **Prometheus**: Use Thanos or Cortex for HA and long-term storage
2. **Loki**: Deploy in distributed mode with object storage
3. **Alertmanager**: Run multiple instances with clustering

## Next steps

- [Docker deployment](./docker.md) - Container deployment options
- [Kubernetes deployment](./kubernetes.md) - Production Kubernetes setup
- [Configuration](../configuration/config.md) - Service configuration options
