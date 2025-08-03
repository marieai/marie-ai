CREATE OR REPLACE VIEW marie_scheduler.ready_jobs_view AS
WITH candidate_jobs AS (
    SELECT j.*
    FROM marie_scheduler.job j
    INNER JOIN marie_scheduler.dag d ON d.id = j.dag_id
    WHERE j.state < 'active'
      AND j.start_after < now()
      AND d.state != 'completed'
),
unblocked_jobs AS (
    SELECT j.*
    FROM candidate_jobs j
    WHERE NOT EXISTS (
        SELECT 1
        FROM marie_scheduler.job_dependencies dep
        JOIN marie_scheduler.job d2
          ON d2.name = dep.depends_on_name
         AND d2.id = dep.depends_on_id
        WHERE dep.job_name = j.name
          AND dep.job_id = j.id
          AND d2.state <> 'completed'
    )
)
SELECT *
FROM unblocked_jobs;
