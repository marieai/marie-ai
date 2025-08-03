CREATE OR REPLACE FUNCTION marie_scheduler.jobs_with_unmet_dependencies()
RETURNS TABLE (
    id UUID,
    name TEXT,
    state marie_scheduler.job_state,
    dependencies JSONB,
    dag_id UUID,
    job_level INTEGER
)
LANGUAGE sql
AS $$
    SELECT
        j.id,
        j.name,
        j.state,
        j.dependencies,
        j.dag_id,
        j.job_level
    FROM
        marie_scheduler.job AS j
    WHERE EXISTS (
        SELECT 1
        FROM
            marie_scheduler.job_dependencies AS jd
        JOIN
            marie_scheduler.job AS d ON jd.depends_on_id = d.id
        WHERE
            jd.job_id = j.id AND d.state != 'completed'
    );

$$;