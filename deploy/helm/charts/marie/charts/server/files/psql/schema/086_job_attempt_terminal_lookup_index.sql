-- File: 086_job_attempt_terminal_lookup_index.sql
-- Description: Cover terminal-attempt checks in operational job pages

CREATE INDEX IF NOT EXISTS job_attempt_terminal_lookup_idx
    ON {schema}.job_attempt (run_attempt_id)
    INCLUDE (terminal_accepted, terminal_work_state)
    WHERE terminal_accepted IS FALSE
       OR terminal_work_state IS NOT NULL;
