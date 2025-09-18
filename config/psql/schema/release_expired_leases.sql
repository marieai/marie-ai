CREATE OR REPLACE FUNCTION marie_scheduler.release_expired_leases()
RETURNS integer
LANGUAGE sql
AS $$
  WITH upd AS (
    UPDATE marie_scheduler.job
    SET state = 'retry', leased_by = NULL, leased_until = NULL
    WHERE state = 'leased' AND leased_until <= now()
    RETURNING 1
  )
  SELECT COUNT(*) FROM upd;
$$;
