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

CREATE INDEX IF NOT EXISTS job_operational_created_idx
    ON {schema}.job (created_on DESC);

CREATE INDEX IF NOT EXISTS job_operational_started_idx
    ON {schema}.job (started_on DESC)
    WHERE started_on IS NOT NULL;

CREATE INDEX IF NOT EXISTS job_operational_completed_idx
    ON {schema}.job (completed_on DESC)
    WHERE completed_on IS NOT NULL;
