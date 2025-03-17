---
sidebar_position: 3
---

# Maintenance

## list all jobs

```sql
select * from marie_scheduler.job
```

```sql
SELECT *
FROM marie_scheduler.job
WHERE data->'metadata'->>'doc_type' = 'doc_type'
AND data->'metadata'->>'doc_id' = 'doc_id_0001'
```

## Purge all jobs

```sql

TRUNCATE marie_scheduler.job;
TRUNCATE marie_scheduler.job_history;
TRUNCATE kv_store_worker;
TRUNCATE kv_store_worker_history;

```

## Create queue

```sql
SELECT marie_scheduler.create_queue('extract', '{"retry_limit":1}'::json)
SELECT marie_scheduler.create_queue('extract', '{"retry_limit":2}'::json)
```


## Get the next job in DAG( Directed Acyclic Graph)

```sql
SELECT j.*
FROM marie_scheduler.job AS j
WHERE j.state < 'active'
  AND NOT EXISTS (
      SELECT 1
      FROM marie_scheduler.job AS d
      WHERE d.id IN (
          SELECT value::uuid
          FROM jsonb_array_elements_text(j.dependencies)  -- or jsonb_array_elements() if storing as JSONB
      )
      AND d.state != 'completed'
  )
ORDER BY j.priority DESC, j.created_on ASC
LIMIT 1
```