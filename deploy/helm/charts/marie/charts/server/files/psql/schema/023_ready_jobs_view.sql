CREATE OR REPLACE VIEW {schema}.ready_jobs_view AS
WITH candidate_jobs AS (
    SELECT j.*
    FROM {schema}.job j
    INNER JOIN {schema}.dag d ON d.id = j.dag_id
    WHERE j.state::text IN ('created', 'retry')
      AND j.start_after < now()
      AND d.state::text != 'completed'
),
unblocked_jobs AS (
    SELECT j.*
    FROM candidate_jobs j
    WHERE NOT EXISTS (
        SELECT 1
        FROM {schema}.job_dependencies dep
        JOIN {schema}.job d2
          ON d2.name = dep.depends_on_name
         AND d2.id = dep.depends_on_id
        WHERE dep.job_name = j.name
          AND dep.job_id = j.id
          AND d2.state::text NOT IN ('completed', 'skipped')
    )
)
SELECT *
FROM unblocked_jobs;
