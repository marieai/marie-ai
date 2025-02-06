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
