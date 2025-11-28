CREATE OR REPLACE FUNCTION {schema}.release_lease(_ids uuid[])
RETURNS uuid[] LANGUAGE sql AS
$$
  WITH upd AS (
    UPDATE {schema}.job j
    SET lease_owner      = NULL,
        lease_expires_at = NULL
    FROM unnest(_ids) u(id)
    WHERE j.id = u.id
      AND j.lease_expires_at IS NOT NULL
    RETURNING j.id
  )
  SELECT COALESCE(array_agg(id), '{}') FROM upd;
$$;
