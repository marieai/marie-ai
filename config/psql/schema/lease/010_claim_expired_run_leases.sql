CREATE OR REPLACE FUNCTION {schema}.claim_expired_run_leases(
  _limit int DEFAULT 1000
)
RETURNS TABLE (
  id uuid,
  name text,
  dag_id uuid,
  previous_state text,
  retry_count int,
  retry_limit int,
  retry_delay int,
  retry_backoff boolean,
  start_after timestamptz,
  run_owner text,
  run_attempt_id uuid,
  run_lease_expires_at timestamptz
)
LANGUAGE sql
AS $$
  SELECT
    j.id,
    j.name,
    j.dag_id,
    j.state::text AS previous_state,
    j.retry_count,
    j.retry_limit,
    j.retry_delay,
    j.retry_backoff,
    j.start_after,
    j.run_owner,
    j.run_attempt_id,
    j.run_lease_expires_at
  FROM {schema}.job j
  WHERE j.state = 'active'
    AND j.run_owner IS NOT NULL
    AND j.run_attempt_id IS NOT NULL
    AND j.run_lease_expires_at IS NOT NULL
    AND j.run_lease_expires_at <= now()
  ORDER BY j.run_lease_expires_at, j.id
  LIMIT _limit
  FOR UPDATE OF j SKIP LOCKED;
$$;
