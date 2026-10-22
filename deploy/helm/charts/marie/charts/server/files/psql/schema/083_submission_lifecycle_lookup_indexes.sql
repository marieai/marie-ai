-- File: 083_submission_lifecycle_lookup_indexes.sql
-- Description: Add indexed history lookups for submission lifecycle analysis

CREATE INDEX IF NOT EXISTS dag_history_lifecycle_id_idx
    ON {schema}.dag_history (id);

CREATE INDEX IF NOT EXISTS job_history_lifecycle_id_idx
    ON {schema}.job_history (id);

CREATE INDEX IF NOT EXISTS job_history_lifecycle_dag_id_idx
    ON {schema}.job_history (dag_id);
