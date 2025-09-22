CREATE OR REPLACE FUNCTION marie_scheduler.release_expired_leases(
  _max_rows integer DEFAULT NULL
)
RETURNS integer
LANGUAGE sql
AS $$
  WITH cand AS (
    SELECT id
    FROM marie_scheduler.job
    WHERE state IN ('created','retry')
      AND lease_owner IS NOT NULL
      AND lease_expires_at IS NOT NULL
      AND lease_expires_at <= now()
    ORDER BY lease_expires_at
    LIMIT COALESCE(_max_rows, 2147483647)
  ),
  upd AS (
    UPDATE marie_scheduler.job j
    SET lease_owner      = NULL,
        lease_expires_at = NULL
    FROM cand
    WHERE j.id = cand.id
    RETURNING j.id
  )
  SELECT COUNT(*) FROM upd;
$$;
