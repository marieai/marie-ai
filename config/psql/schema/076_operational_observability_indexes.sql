-- File: 076_operational_observability_indexes.sql
-- Description: Bounded operational history and flow query indexes

CREATE INDEX IF NOT EXISTS job_history_operational_event_idx
    ON {schema}.job_history (history_created_on DESC, history_id DESC);

CREATE INDEX IF NOT EXISTS dag_history_operational_event_idx
    ON {schema}.dag_history (history_created_on DESC, history_id DESC);

CREATE INDEX IF NOT EXISTS job_attempt_operational_updated_idx
    ON {schema}.job_attempt (updated_on DESC, run_attempt_id DESC);

CREATE INDEX IF NOT EXISTS job_attempt_operational_activated_idx
    ON {schema}.job_attempt (activated_at DESC, run_attempt_id DESC);

CREATE INDEX IF NOT EXISTS job_attempt_operational_terminal_idx
    ON {schema}.job_attempt (terminal_at DESC, run_attempt_id DESC)
    WHERE terminal_at IS NOT NULL;

CREATE INDEX IF NOT EXISTS job_attempt_operational_recovery_idx
    ON {schema}.job_attempt (recovery_at DESC, run_attempt_id DESC)
    WHERE recovery_at IS NOT NULL;

CREATE INDEX IF NOT EXISTS job_attempt_operational_facets_idx
    ON {schema}.job_attempt (gateway_instance_id, executor);

-- Time-dependent attention ranks are evaluated at read time within group 3.
CREATE INDEX IF NOT EXISTS job_attempt_operational_attention_idx
    ON {schema}.job_attempt (
        (CASE
            WHEN terminal_accepted IS FALSE OR (
                terminal_at IS NOT NULL AND (
                    terminal_accepted IS FALSE OR (
                        terminal_status IS NOT NULL
                        AND terminal_work_state IS NOT NULL
                        AND LOWER(terminal_status) <> LOWER(terminal_work_state)
                    )
                )
            ) THEN 0
            WHEN terminal_at IS NOT NULL AND (
                (
                    terminal_gateway_instance_id IS NOT NULL
                    AND gateway_instance_id IS NOT NULL
                    AND terminal_gateway_instance_id <> gateway_instance_id
                )
                OR (
                    terminal_scheduler_lease_owner IS NOT NULL
                    AND terminal_scheduler_lease_owner <> scheduler_lease_owner
                )
            ) THEN 1
            WHEN recovery_at IS NOT NULL THEN 2
            WHEN terminal_at IS NULL AND recovery_at IS NULL THEN 3
            ELSE 4
        END),
        updated_on DESC,
        run_attempt_id DESC
    );

CREATE INDEX IF NOT EXISTS job_u_operational_created_idx
    ON {schema}.job (created_on DESC);

CREATE INDEX IF NOT EXISTS job_u_operational_started_idx
    ON {schema}.job (started_on DESC)
    WHERE started_on IS NOT NULL;

CREATE INDEX IF NOT EXISTS job_u_operational_completed_idx
    ON {schema}.job (completed_on DESC)
    WHERE completed_on IS NOT NULL;

CREATE INDEX IF NOT EXISTS job_u_operational_terminal_attempt_idx
    ON {schema}.job (created_on DESC, id)
    INCLUDE (state, run_attempt_id)
    WHERE run_attempt_id IS NOT NULL
      AND state IN ('completed', 'skipped', 'failed', 'expired', 'cancelled');
