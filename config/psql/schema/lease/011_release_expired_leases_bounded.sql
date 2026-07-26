CREATE OR REPLACE FUNCTION {schema}.release_expired_leases(
  _max_rows integer DEFAULT 1000
)
RETURNS integer
LANGUAGE sql
AS $$
  WITH cand AS (
    SELECT j.id, j.name
    FROM {schema}.job j
    WHERE j.state IN ('created','retry')
      AND j.lease_owner IS NOT NULL
      AND j.lease_expires_at IS NOT NULL
      AND j.lease_expires_at <= now()
    ORDER BY j.lease_expires_at, j.id
    LIMIT COALESCE(_max_rows, 1000)
    FOR UPDATE OF j SKIP LOCKED
  ),
  upd AS (
    UPDATE {schema}.job j
    SET lease_owner      = NULL,
        lease_expires_at = NULL
    FROM cand
    WHERE j.name = cand.name
      AND j.id = cand.id
    RETURNING 1
  )
  SELECT COUNT(*) FROM upd;
$$;
