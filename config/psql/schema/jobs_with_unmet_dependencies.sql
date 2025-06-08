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
    FROM marie_scheduler.job AS j
    WHERE j.dependencies IS NOT NULL
      AND jsonb_array_length(j.dependencies) > 0
      AND EXISTS (
          SELECT 1
          FROM marie_scheduler.job AS d
          WHERE d.id IN (
              SELECT value::uuid
              FROM jsonb_array_elements_text(j.dependencies)
          )
          AND d.state != 'completed'
      );
$$;
