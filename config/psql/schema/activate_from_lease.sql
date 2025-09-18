CREATE OR REPLACE FUNCTION marie_scheduler.activate_from_lease(
  p_ids uuid[],
  p_owner text,
  p_run_owner text,
  p_run_ttl_seconds integer
)
RETURNS SETOF uuid
LANGUAGE sql AS $$
  UPDATE marie_scheduler.job j
  SET state='active',
      started_on = COALESCE(j.started_on, now()),
      lease_owner = NULL,
      lease_expires_at = NULL,
      run_owner = p_run_owner,
      run_lease_expires_at = now() + make_interval(secs => p_run_ttl_seconds)
  WHERE j.id = ANY(p_ids)
    AND j.state='leased'
    AND j.lease_owner = p_owner
  RETURNING j.id;
$$;
