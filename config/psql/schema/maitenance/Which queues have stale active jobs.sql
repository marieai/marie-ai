

  -- 1. Which queues have stale active jobs?
  SELECT name, COUNT(*) AS active_count, MIN(started_on) AS oldest_started_on
  FROM marie_scheduler.job
  WHERE state = 'active'
  GROUP BY name
  ORDER BY active_count DESC;

  -- 2. Active jobs whose run lease is already expired
  SELECT id, name, dag_id, started_on, run_owner, run_lease_expires_at
  FROM marie_scheduler.job
  WHERE state = 'active'
    AND run_lease_expires_at IS NOT NULL
    AND run_lease_expires_at < NOW()
  ORDER BY run_lease_expires_at ASC;

  -- 3. Compare DB active state vs KV job status
  -- Adjust kv schema if your table lives in public instead of marie_scheduler.
  SELECT
    j.id,
    j.name,
    j.dag_id,
    j.started_on,
    j.run_lease_expires_at,
    kv.value->>'status' AS kv_status,
    to_timestamp(((kv.value->>'end_time')::bigint) / 1000.0) AS kv_end_time
  FROM marie_scheduler.job j
  LEFT JOIN marie_scheduler.kv_store_worker kv
    ON kv.namespace = 'job'
   AND kv.key = 'marie_internal/job_info_' || j.id::text
   AND kv.is_deleted = FALSE
  WHERE j.state = 'active'
  ORDER BY j.started_on ASC;

