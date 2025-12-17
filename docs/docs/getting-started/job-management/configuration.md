---
sidebar_position: 5
---

# Configuration

Configure the Marie-AI job scheduler for your deployment requirements.

## Scheduler configuration

The scheduler is configured through the Gateway YAML configuration:

```yaml
gateway:
  uses: MarieGateway
  with:
    # PostgreSQL connection for job scheduler
    job_scheduler_kwargs:
      provider: postgresql
      hostname: localhost
      port: 5432
      username: marie
      password: ${POSTGRES_PASSWORD}
      database: marie

    # Key-value store (same database, different use)
    kv_store_kwargs:
      provider: postgresql
      hostname: localhost
      port: 5432
      username: marie
      password: ${POSTGRES_PASSWORD}
      database: marie

    # Service discovery
    discovery_host: localhost
    discovery_port: 2379
    discovery_service_name: marie
```

## Scheduler options

| Option | Default | Description |
|--------|---------|-------------|
| `queue_names` | `["default"]` | List of queue names to monitor |
| `distributed_scheduler` | `false` | Enable distributed lease management |
| `max_workers` | `5` | Thread pool size for DB operations |
| `lease_ttl_seconds` | `5` | Job lease timeout |
| `run_ttl_seconds` | `60` | Active job execution timeout |
| `maintenance_interval` | `60` | Maintenance task frequency (seconds) |

## DAG manager settings

Configure DAG processing behavior:

```yaml
gateway:
  with:
    dag_manager:
      min_concurrent_dags: 1
      max_concurrent_dags: 16
      cache_ttl_seconds: 5
      dag_cache_size: 5000
      frontier_batch_size: 1000
```

| Option | Default | Description |
|--------|---------|-------------|
| `min_concurrent_dags` | `1` | Minimum DAGs to process simultaneously |
| `max_concurrent_dags` | `16` | Maximum DAGs to process simultaneously |
| `cache_ttl_seconds` | `5` | DAG cache entry lifetime |
| `dag_cache_size` | `5000` | Maximum cached DAG entries |
| `frontier_batch_size` | `1000` | Candidate jobs per poll cycle |

## Heartbeat and monitoring

Configure scheduler health monitoring:

```yaml
gateway:
  with:
    heartbeat:
      interval: 5.0
      window_minutes: 10
      trend_points: 12
      recent_window_minutes: 1
      max_retries: 3
      error_backoff: 5.0
      enable_trend_arrows: true
      enable_per_queue_stats: true
      enable_executor_stats: true
      log_active_dags: false
```

| Option | Default | Description |
|--------|---------|-------------|
| `interval` | `5.0` | Heartbeat loop interval (seconds) |
| `window_minutes` | `10` | Rolling throughput window |
| `trend_points` | `12` | Data points for trend calculation |
| `recent_window_minutes` | `1` | Recent throughput window |
| `max_retries` | `3` | Max retries for heartbeat operations |
| `error_backoff` | `5.0` | Backoff time after errors |
| `enable_trend_arrows` | `true` | Show trend indicators in logs |
| `enable_per_queue_stats` | `true` | Log per-queue statistics |
| `enable_executor_stats` | `true` | Log executor slot information |
| `log_active_dags` | `false` | Log active DAG IDs (debug) |

## Queue configuration

Queues organize jobs and can have custom policies. Create queues via SQL:

```sql
INSERT INTO marie_scheduler.queue (name, retry_limit, expire_in_seconds)
VALUES ('high-priority', 5, 7200);
```

| Column | Type | Description |
|--------|------|-------------|
| `name` | `text` | Unique queue name |
| `retry_limit` | `integer` | Default retry limit for jobs |
| `expire_in_seconds` | `integer` | Default TTL for jobs |

## Database setup

### Schema creation

The scheduler uses the `marie_scheduler` schema. Initialize it:

```sql
CREATE SCHEMA IF NOT EXISTS marie_scheduler;
```

### Required tables

The scheduler requires these tables:

| Table | Purpose |
|-------|---------|
| `job` | Active job records |
| `job_history` | Historical job records |
| `dag` | DAG definitions |
| `dag_history` | Historical DAG records |
| `queue` | Queue definitions |
| `schedule` | Cron/scheduled jobs |
| `archive` | Archived completed jobs |
| `subscription` | Pub/sub channels |
| `version` | Schema version |

### Connection pool

The scheduler uses a connection pool for database access:

```yaml
job_scheduler_kwargs:
  provider: postgresql
  hostname: localhost
  port: 5432
  username: marie
  password: ${POSTGRES_PASSWORD}
  database: marie
  # Pool settings (if supported)
  pool_size: 10
  max_overflow: 20
```

## ETCD configuration

For distributed scheduling and service discovery:

```yaml
gateway:
  with:
    discovery_host: localhost
    discovery_port: 2379
    discovery_service_name: marie
```

### ETCD cluster

For production, configure an ETCD cluster:

```yaml
gateway:
  with:
    discovery_host: etcd-0.etcd,etcd-1.etcd,etcd-2.etcd
    discovery_port: 2379
    discovery_service_name: marie
```

## Environment variables

Use environment variables for sensitive configuration:

```yaml
gateway:
  with:
    job_scheduler_kwargs:
      hostname: ${POSTGRES_HOST:-localhost}
      port: ${POSTGRES_PORT:-5432}
      username: ${POSTGRES_USER:-marie}
      password: ${POSTGRES_PASSWORD}
      database: ${POSTGRES_DB:-marie}
```

### Required variables

| Variable | Description |
|----------|-------------|
| `POSTGRES_PASSWORD` | Database password |
| `POSTGRES_HOST` | Database hostname (optional) |
| `POSTGRES_PORT` | Database port (optional) |
| `POSTGRES_USER` | Database username (optional) |
| `POSTGRES_DB` | Database name (optional) |

## Complete example

Full scheduler configuration:

```yaml
jtype: Flow
with:
  protocol: [grpc, http]
  port: [54321, 54322]

gateway:
  uses: MarieGateway
  with:
    # Database connections
    job_scheduler_kwargs:
      provider: postgresql
      hostname: ${POSTGRES_HOST:-localhost}
      port: ${POSTGRES_PORT:-5432}
      username: ${POSTGRES_USER:-marie}
      password: ${POSTGRES_PASSWORD}
      database: ${POSTGRES_DB:-marie}

    kv_store_kwargs:
      provider: postgresql
      hostname: ${POSTGRES_HOST:-localhost}
      port: ${POSTGRES_PORT:-5432}
      username: ${POSTGRES_USER:-marie}
      password: ${POSTGRES_PASSWORD}
      database: ${POSTGRES_DB:-marie}

    # Service discovery
    discovery_host: ${ETCD_HOST:-localhost}
    discovery_port: ${ETCD_PORT:-2379}
    discovery_service_name: marie

    # Scheduler settings
    distributed_scheduler: true
    max_workers: 10
    lease_ttl_seconds: 5
    run_ttl_seconds: 120

    # DAG manager
    dag_manager:
      min_concurrent_dags: 2
      max_concurrent_dags: 32
      cache_ttl_seconds: 10
      dag_cache_size: 10000
      frontier_batch_size: 2000

    # Heartbeat
    heartbeat:
      interval: 5.0
      window_minutes: 10
      enable_per_queue_stats: true
      enable_executor_stats: true

executors:
  - name: extract
    uses: ExtractExecutor
    replicas: 4
```

## Tuning recommendations

### High throughput

For maximum job throughput:

```yaml
dag_manager:
  max_concurrent_dags: 64
  frontier_batch_size: 5000

heartbeat:
  interval: 2.0
```

### Low latency

For minimal job latency:

```yaml
dag_manager:
  min_concurrent_dags: 4
  frontier_batch_size: 100

heartbeat:
  interval: 1.0
```

### Resource constrained

For limited resources:

```yaml
dag_manager:
  max_concurrent_dags: 8
  dag_cache_size: 1000
  frontier_batch_size: 500

max_workers: 3
```

## Next steps

- [API reference](./api.md) - REST and Python APIs
- [Maintenance](./maintenance.md) - Database operations and cleanup
