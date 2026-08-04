-- File: 081_operational_worker_history_index.sql
-- Description: Worker execution-history lookup for operational consoles

CREATE INDEX IF NOT EXISTS kv_store_worker_history_job_key_time_idx
    ON {schema}.kv_store_worker_history (
        key,
        change_time DESC,
        history_id DESC
    )
    WHERE namespace = 'job';
