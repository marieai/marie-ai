CREATE OR REPLACE FUNCTION marie_scheduler.fetch_next_job(
    job_name TEXT,
    batch_size INTEGER DEFAULT 1
)
RETURNS SETOF marie_scheduler.job
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    WITH candidate AS MATERIALIZED (
      SELECT j.id, j.job_level, j.priority
      FROM marie_scheduler.job j
      JOIN marie_scheduler.dag d ON d.id = j.dag_id
      WHERE j.name=job_name
        AND j.state IN ('created','retry')
        AND d.state NOT IN ('completed','failed','cancelled')
        AND NOT EXISTS (
          SELECT 1
          FROM marie_scheduler.job_dependencies dep
          JOIN marie_scheduler.job dj ON dj.id = dep.depends_on_id
          WHERE dep.job_id = j.id
            AND dj.state <> 'completed'
        )
      ORDER BY j.job_level ASC, j.priority DESC
      LIMIT 5000
    )
    SELECT jx.*
    FROM candidate c
    JOIN LATERAL (
      SELECT j.* FROM marie_scheduler.job j
      WHERE j.id = c.id                -- 1-row index lookup
    ) jx ON true
    ;
    --LIMIT batch_size;
END;
$$;