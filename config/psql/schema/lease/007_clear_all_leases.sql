CREATE OR REPLACE FUNCTION {schema}.clear_all_leases()
RETURNS integer
LANGUAGE sql
AS $$
  WITH upd AS (
    UPDATE {schema}.job
    SET
      lease_owner           = NULL,
      lease_expires_at      = NULL,
      lease_epoch           = lease_epoch + 1,  -- bump CAS
      run_owner             = NULL,
      run_lease_expires_at  = NULL
    WHERE lease_owner IS NOT NULL
       OR lease_expires_at IS NOT NULL
       OR run_owner IS NOT NULL
       OR run_lease_expires_at IS NOT NULL
    RETURNING 1
  )
  SELECT COUNT(*) FROM upd;
$$;
