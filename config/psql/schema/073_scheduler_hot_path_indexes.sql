-- Index maintenance scans by expiration time so empty cycles do not scan the
-- ready and active working sets.

CREATE INDEX IF NOT EXISTS job_u_expired_acquisition_lease_idx
    ON {schema}.job (lease_expires_at, id)
    WHERE state IN ('created', 'retry')
      AND lease_owner IS NOT NULL
      AND lease_expires_at IS NOT NULL;

CREATE INDEX IF NOT EXISTS job_u_expired_run_lease_idx
    ON {schema}.job (run_lease_expires_at, id)
    WHERE state = 'active'
      AND run_owner IS NOT NULL
      AND run_attempt_id IS NOT NULL
      AND run_lease_expires_at IS NOT NULL;

CREATE INDEX IF NOT EXISTS dag_admission_order_idx
    ON {schema}.dag (
        priority DESC,
        (COALESCE(soft_sla, hard_sla)),
        created_on,
        id
    )
    WHERE state IN ('created', 'active');
