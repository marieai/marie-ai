CREATE OR REPLACE FUNCTION marie_scheduler.lease_jobs_by_id(
  p_job_name text,
  p_ids uuid[],
  p_owner text,
  p_ttl_seconds integer
)
RETURNS TABLE(id uuid, lease_epoch bigint)
LANGUAGE sql AS $$
  UPDATE marie_scheduler.job j
  SET state='leased',
      lease_owner = p_owner,
      lease_expires_at = now() + make_interval(secs => p_ttl_seconds),
      lease_epoch = j.lease_epoch + 1
  WHERE j.name = p_job_name
    AND j.id   = ANY(p_ids)
    AND j.state IN ('created','retry','leased')
    AND (j.lease_expires_at IS NULL OR j.lease_expires_at < now())
  RETURNING j.id, j.lease_epoch;
$$;
