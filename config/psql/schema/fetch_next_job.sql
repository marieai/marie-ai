
--- -- This view selects jobs that are ready to be processed, meaning they are not active,
--- -- have no unmet dependencies, and their DAG is not completed.

CREATE OR REPLACE FUNCTION marie_scheduler.fetch_next_job(
    job_name TEXT,
    batch_size INT DEFAULT 1,
    mark_active BOOLEAN DEFAULT TRUE
)
RETURNS SETOF marie_scheduler.job
LANGUAGE plpgsql
AS $$
BEGIN
    IF mark_active THEN
        RETURN QUERY
        UPDATE marie_scheduler.job j
        SET state = 'active'::marie_scheduler.job_state,
            started_on = now(),
            retry_count = CASE
                WHEN started_on IS NOT NULL THEN retry_count + 1
                ELSE retry_count
            END
        FROM (
            SELECT *
            FROM marie_scheduler.ready_jobs_view
            WHERE name = job_name
              AND state < 'active'
            ORDER BY dag_id, job_level ASC, priority DESC, created_on, id
            LIMIT batch_size
        ) rj
        WHERE j.id = rj.id
        RETURNING j.*;
    ELSE
        RETURN QUERY
        SELECT *
        FROM marie_scheduler.ready_jobs_view
        WHERE name = job_name
          AND state < 'active'
        ORDER BY dag_id, job_level ASC, priority DESC, created_on, id
        LIMIT batch_size;
    END IF;
END;
$$