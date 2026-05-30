CREATE OR REPLACE FUNCTION {schema}.extend_run_lease(
  _ids            uuid[],
  _run_owner      text,
  _run_attempt_id uuid,
  _extend_by      interval DEFAULT '5 minutes'
)
RETURNS uuid[] LANGUAGE sql AS
$$
  WITH ok AS (
    SELECT j.id
    FROM {schema}.job j
    JOIN unnest(_ids) u(id) ON u.id = j.id
    WHERE j.state = 'active'
      AND j.run_owner = _run_owner
      AND j.run_attempt_id = _run_attempt_id
      AND (j.run_lease_expires_at IS NULL OR j.run_lease_expires_at > now() - interval '1 minute')
  ), upd AS (
    UPDATE {schema}.job j
    SET run_lease_expires_at = now() + _extend_by
    FROM ok
    WHERE j.id = ok.id
      AND j.state = 'active'
      AND j.run_owner = _run_owner
      AND j.run_attempt_id = _run_attempt_id
    RETURNING j.id
  )
  SELECT COALESCE(array_agg(id), '{}') FROM upd;
$$;
