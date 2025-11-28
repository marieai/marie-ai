CREATE OR REPLACE FUNCTION {schema}.lease_jobs_by_id(
  _ids   uuid[],
  _ttl   interval DEFAULT '2 minutes',
  _owner text     DEFAULT COALESCE(inet_client_addr()::text, 'unknown'),
  _name  text     DEFAULT NULL           -- when provided, restrict to this job.name
)
RETURNS uuid[] LANGUAGE sql AS
$$
  WITH cand AS (
    SELECT j.id
    FROM {schema}.job j
    JOIN unnest(_ids) u(id) ON u.id = j.id
    WHERE j.state IN ('created','retry')
      AND (_name IS NULL OR j.name = _name)
      AND (j.lease_expires_at IS NULL OR j.lease_expires_at <= now())
  ),
  upd AS (
    UPDATE {schema}.job j
    SET lease_owner      = _owner,
        lease_expires_at = now() + _ttl,
        lease_epoch      = j.lease_epoch + 1
    FROM cand
    WHERE j.id = cand.id
    RETURNING j.id
  )
  SELECT COALESCE(array_agg(id), '{}') FROM upd;
$$;
