
--- Function to delete failed DAGS and their associated jobs
-- Usage:
-- Single DAG ID
-- SELECT marie_scheduler.delete_dags_and_jobs(ARRAY['d4f5e6b7-1234-5678-9abc-def012345678'::uuid]);
-- Multiple DAG IDs
-- SELECT marie_scheduler.delete_dags_and_jobs(ARRAY['d4f5e6b7-1234-5678-9abc-def012345678'::uuid, 'a1b2c3d4-5678-9abc-def0-1234567890ab'::uuid]);
-- SELECT marie_scheduler.delete_dags_and_jobs(
--     ARRAY(
--         SELECT DISTINCT dag_id::uuid
--         FROM marie_scheduler.job
--         WHERE name = 'extract'
--     )gb
-- );



CREATE OR REPLACE FUNCTION marie_scheduler.delete_dags_and_jobs(p_dag_ids uuid[])
RETURNS void
LANGUAGE plpgsql
AS
$$
DECLARE
    dep_count int;
    job_count int;
    dag_count int;
BEGIN
    -- Delete dependencies first
    DELETE FROM marie_scheduler.job_dependencies
    USING marie_scheduler.job j
    WHERE j.id = job_dependencies.depends_on_id
      AND j.dag_id = ANY(p_dag_ids);
    GET DIAGNOSTICS dep_count = ROW_COUNT;

    -- Delete jobs
    DELETE FROM marie_scheduler.job
    WHERE dag_id = ANY(p_dag_ids);
    GET DIAGNOSTICS job_count = ROW_COUNT;

    -- Delete DAGs
    DELETE FROM marie_scheduler.dag
    WHERE id = ANY(p_dag_ids);
    GET DIAGNOSTICS dag_count = ROW_COUNT;

    RAISE NOTICE 'Deleted % dependencies, % jobs, and % dags', dep_count, job_count, dag_count;
END;
$$;
