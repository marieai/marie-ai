CREATE OR REPLACE FUNCTION marie_scheduler.reap_expired_leases(_limit int DEFAULT 1000)
RETURNS integer LANGUAGE sql AS
$$
  WITH exp AS (
    SELECT id
    FROM marie_scheduler.job
    WHERE lease_expires_at IS NOT NULL
      AND lease_expires_at <= now()
    LIMIT _limit
  ), upd AS (
    UPDATE marie_scheduler.job j
    SET lease_owner      = NULL,
        lease_expires_at = NULL
    FROM exp
    WHERE j.id = exp.id
    RETURNING 1
  )
  SELECT COALESCE(count(*), 0) FROM upd;
$$;
