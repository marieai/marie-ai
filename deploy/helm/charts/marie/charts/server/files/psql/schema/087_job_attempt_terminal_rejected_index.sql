-- File: 087_job_attempt_terminal_rejected_index.sql
-- Description: Support ordered operational lookup of rejected terminal attempts

CREATE INDEX IF NOT EXISTS job_attempt_terminal_rejected_updated_idx
    ON {schema}.job_attempt (updated_on DESC, run_attempt_id DESC)
    WHERE terminal_accepted IS FALSE;
