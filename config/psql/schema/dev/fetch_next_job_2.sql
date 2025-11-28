CREATE OR REPLACE FUNCTION marie_scheduler.fetch_next_job(
    job_name TEXT,
    batch_size INTEGER DEFAULT 1
)
RETURNS SETOF marie_scheduler.job
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    WITH candidate_jobs AS (
        SELECT j.*
        FROM marie_scheduler.job j
        JOIN marie_scheduler.dag d ON d.id = j.dag_id
        WHERE j.name = job_name
          AND j.state IN ('created', 'retry')  -- Only consider jobs that can run
          AND j.start_after < now()
          AND d.state NOT IN ('completed', 'failed', 'cancelled')  -- Exclude problematic DAGs
          ORDER BY j.job_level DESC, j.priority DESC
          LIMIT 500
    ),
    unblocked_jobs AS (
        SELECT j.*
        FROM candidate_jobs j
        WHERE NOT EXISTS (
            SELECT 1
            FROM marie_scheduler.job_dependencies dep
            JOIN marie_scheduler.job d2 ON d2.id = dep.depends_on_id
            WHERE dep.job_id = j.id
              AND d2.state NOT IN ('completed')  -- Dependencies must be completed
        )
        -- Exclude jobs from DAGs with any failed dependencies
        AND NOT EXISTS (
            SELECT 1
            FROM marie_scheduler.job_dependencies dep
            JOIN marie_scheduler.job d2 ON d2.id = dep.depends_on_id
            WHERE dep.job_id = j.id
              AND d2.state = 'failed'
        )
    )
    SELECT *
    FROM unblocked_jobs
    ;
    --ORDER BY job_level ASC, priority DESC
    --LIMIT batch_size;
END;
$$;