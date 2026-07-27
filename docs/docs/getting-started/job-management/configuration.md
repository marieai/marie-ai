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
| `queue_names` | Required | List of queue names to monitor |
| `job_event_worker_count` | `8` | Keyed publisher workers; events for one job always use one worker |
| `job_event_queue_size` | `1024` | Total bounded publisher capacity; job-event producers block when it is full |
| `lease_ttl_seconds` | `5` | Job lease timeout |
| `run_ttl_seconds` | `60` | Active job execution timeout |
| `maintenance_interval` | `60` | Maintenance task frequency (seconds) |

The submission call returns only after the DAG and its jobs commit to
PostgreSQL. Ingress concurrency is therefore bounded by the gateway and
database connection limits rather than a scheduler-local submission queue.

## DAG manager settings

Configure DAG processing behavior:

```yaml
gateway:
  with:
    job_scheduler_kwargs:
      dag_manager:
        max_concurrent_dags: 16
        dag_cache_size: 5000
        frontier_batch_size: 1000
```

| Option | Default | Description |
|--------|---------|-------------|
| `max_concurrent_dags` | `16` | Maximum DAGs to process simultaneously |
| `dag_cache_size` | `5000` | Maximum cached DAG entries |
| `frontier_batch_size` | `1000` | Candidate jobs per poll cycle |

## Scheduler diagnostics

Use `GET /api/debug` for scheduler runtime diagnostics. The response includes
scheduler state, queue depth, state counts, active DAGs, dispatch counters, and
the frontier summary. Scheduler phase transitions and rejected stale attempts
are also emitted through structured scheduler traces.

Unknown `job_scheduler_kwargs` and `dag_manager` keys fail startup. Durable
database leasing is always enabled; the retired `distributed_scheduler`,
`scheduler_mode`, `hard_sla_policy`, and scheduler `heartbeat` settings are not
supported.

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
  min_pool_size: 1
  max_pool_size: 10
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
      queue_names: [extract]
      job_event_worker_count: 8
      job_event_queue_size: 1024
      lease_ttl_seconds: 5
      run_ttl_seconds: 120
      dag_manager:
        max_concurrent_dags: 32
        dag_cache_size: 10000
        frontier_batch_size: 2000

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

executors:
  - name: extract
    uses: ExtractExecutor
    replicas: 4
```

## Tuning recommendations

### High throughput

For maximum job throughput:

```yaml
job_scheduler_kwargs:
  dag_manager:
    max_concurrent_dags: 64
    frontier_batch_size: 5000
```

### Low latency

For minimal job latency:

```yaml
job_scheduler_kwargs:
  dag_manager:
    max_concurrent_dags: 16
    frontier_batch_size: 100
```

### Resource constrained

For limited resources:

```yaml
job_scheduler_kwargs:
  dag_manager:
    max_concurrent_dags: 8
    dag_cache_size: 1000
    frontier_batch_size: 500
```

## Next steps

- [API reference](./api.md) - REST and Python APIs
- [Maintenance](./maintenance.md) - Database operations and cleanup
