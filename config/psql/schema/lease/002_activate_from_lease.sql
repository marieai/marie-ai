CREATE OR REPLACE FUNCTION {schema}.activate_from_lease(
  _ids      uuid[],
  _run_owner text,
  _run_ttl   interval DEFAULT '5 minutes'
)
RETURNS uuid[] LANGUAGE sql AS
$$
  WITH ok AS (
    SELECT j.id
    FROM {schema}.job j
    JOIN unnest(_ids) u(id) ON u.id = j.id
    WHERE j.lease_expires_at IS NOT NULL
      AND j.lease_expires_at > now()
  ), upd AS (
    UPDATE {schema}.job j
    SET state                 = 'active',
        started_on            = COALESCE(j.started_on, now()),
        run_owner             = _run_owner,
        run_lease_expires_at  = now() + _run_ttl,
        -- clear the acquisition lease
        lease_owner           = NULL,
        lease_expires_at      = NULL
    FROM ok
    WHERE j.id = ok.id
    RETURNING j.id
  )
  SELECT COALESCE(array_agg(id), '{}') FROM upd;
$$;
