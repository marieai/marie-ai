---
sidebar_position: 7
---

# Maintenance

Database maintenance operations for the Marie-AI job scheduler. These SQL commands help you query, clean up, and manage job data.

:::warning
These operations modify database data directly. Always back up your database before running destructive commands like `TRUNCATE`.
:::

## Querying jobs

### List all jobs

View all jobs in the scheduler:

```sql
SELECT * FROM marie_scheduler.job;
```

### Filter by metadata

Query jobs by metadata fields stored in the `data` column:

```sql
SELECT *
FROM marie_scheduler.job
WHERE data->'metadata'->>'doc_type' = 'invoice'
  AND data->'metadata'->>'doc_id' = 'doc_id_0001';
```

### Filter by state

Query jobs in specific states:

```sql
-- Active jobs
SELECT * FROM marie_scheduler.job WHERE state = 'active';

-- Failed jobs
SELECT * FROM marie_scheduler.job WHERE state = 'failed';

-- Jobs pending retry
SELECT * FROM marie_scheduler.job WHERE state = 'retry';
```

### Filter by DAG

Query jobs belonging to a specific DAG:

```sql
SELECT * FROM marie_scheduler.job WHERE dag_id = 'my-dag-id';
```

### Job statistics

Get job counts by state:

```sql
SELECT state, COUNT(*) as count
FROM marie_scheduler.job
GROUP BY state
ORDER BY count DESC;
```

## Cleanup operations

### Purge all jobs

Remove all job data (use with caution):

```sql
-- Clear current jobs
TRUNCATE marie_scheduler.job;

-- Clear job history
TRUNCATE marie_scheduler.job_history;

-- Clear worker state (if used)
TRUNCATE kv_store_worker;
TRUNCATE kv_store_worker_history;
```

### Purge completed jobs

Remove only completed jobs older than a specific date:

```sql
DELETE FROM marie_scheduler.job
WHERE state = 'completed'
  AND completed_on < NOW() - INTERVAL '7 days';
```

### Purge failed jobs

Remove failed jobs after investigation:

```sql
DELETE FROM marie_scheduler.job
WHERE state = 'failed'
  AND completed_on < NOW() - INTERVAL '30 days';
```

### Archive old jobs

Move old jobs to the archive table:

```sql
INSERT INTO marie_scheduler.archive
SELECT * FROM marie_scheduler.job
WHERE state IN ('completed', 'failed', 'cancelled', 'expired')
  AND completed_on < NOW() - INTERVAL '7 days';

DELETE FROM marie_scheduler.job
WHERE state IN ('completed', 'failed', 'cancelled', 'expired')
  AND completed_on < NOW() - INTERVAL '7 days';
```

## Queue management

### Create a queue

Create a new queue with custom settings:

```sql
-- Queue with 1 retry
SELECT marie_scheduler.create_queue('extract', '{"retry_limit":1}'::json);

-- Queue with 2 retries
SELECT marie_scheduler.create_queue('high-priority', '{"retry_limit":2}'::json);
```

### List queues

View all configured queues:

```sql
SELECT * FROM marie_scheduler.queue;
```

### Update queue settings

Modify queue retry limits:

```sql
UPDATE marie_scheduler.queue
SET retry_limit = 3
WHERE name = 'extract';
```

## DAG operations

### Get next job in DAG

Find the next ready job in a DAG (jobs with all dependencies completed):

```sql
SELECT j.*
FROM marie_scheduler.job AS j
WHERE j.state < 'active'
  AND NOT EXISTS (
      SELECT 1
      FROM marie_scheduler.job AS d
      WHERE d.id IN (
          SELECT value::uuid
          FROM jsonb_array_elements_text(j.dependencies)
      )
      AND d.state != 'completed'
  )
ORDER BY j.priority DESC, j.created_on ASC
LIMIT 1;
```

### List DAGs

View all DAGs:

```sql
SELECT * FROM marie_scheduler.dag;
```

### DAG progress

Check progress of a specific DAG:

```sql
SELECT
    dag_id,
    COUNT(*) as total_jobs,
    COUNT(*) FILTER (WHERE state IN ('completed', 'skipped')) as completed,
    COUNT(*) FILTER (WHERE state = 'active') as active,
    COUNT(*) FILTER (WHERE state = 'failed') as failed,
    ROUND(
        100.0 * COUNT(*) FILTER (WHERE state IN ('completed', 'skipped', 'failed', 'cancelled', 'expired'))
        / NULLIF(COUNT(*), 0),
        1
    ) as progress_pct
FROM marie_scheduler.job
WHERE dag_id = 'my-dag-id'
GROUP BY dag_id;
```

## Monitoring queries

### Stuck jobs

Find jobs that have been active too long:

```sql
SELECT *
FROM marie_scheduler.job
WHERE state = 'active'
  AND started_on < NOW() - INTERVAL '1 hour';
```

### Retry analysis

Analyze job retry patterns:

```sql
SELECT
    name,
    AVG(retry_count) as avg_retries,
    MAX(retry_count) as max_retries,
    COUNT(*) as total_jobs
FROM marie_scheduler.job
WHERE retry_count > 0
GROUP BY name
ORDER BY avg_retries DESC;
```

### Throughput metrics

Calculate job throughput:

```sql
SELECT
    DATE_TRUNC('hour', completed_on) as hour,
    COUNT(*) as completed_jobs
FROM marie_scheduler.job
WHERE state = 'completed'
  AND completed_on > NOW() - INTERVAL '24 hours'
GROUP BY DATE_TRUNC('hour', completed_on)
ORDER BY hour;
```

## Schema reference

### Job table columns

| Column | Type | Description |
|--------|------|-------------|
| `id` | `uuid` | Unique job identifier |
| `dag_id` | `uuid` | DAG association |
| `name` | `text` | Executor endpoint |
| `priority` | `integer` | Scheduling priority |
| `data` | `jsonb` | Job payload |
| `state` | `text` | Current state |
| `retry_count` | `integer` | Current retry count |
| `retry_limit` | `integer` | Maximum retries |
| `created_on` | `timestamptz` | Creation time |
| `started_on` | `timestamptz` | Execution start |
| `completed_on` | `timestamptz` | Completion time |
| `soft_sla` | `timestamptz` | Target completion |
| `hard_sla` | `timestamptz` | Deadline |
| `dependencies` | `jsonb` | Dependent job IDs |

## Best practices

1. **Schedule regular cleanup**: Run purge operations during low-traffic periods

2. **Monitor table sizes**: Large job tables impact query performance

3. **Use archive table**: Move old jobs to archive instead of deleting

4. **Index commonly queried fields**: Add indexes for frequent filter conditions

5. **Back up before maintenance**: Always have a backup before destructive operations

## Related documentation

- [Job lifecycle](./job-lifecycle.md) - Understanding job states
- [Configuration](./configuration.md) - Database setup
- [API reference](./api.md) - Programmatic access
