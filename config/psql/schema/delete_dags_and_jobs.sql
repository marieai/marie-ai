
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

create function marie_scheduler.delete_dags_and_jobs(p_dag_ids uuid[]) returns void
    language plpgsql
as
$$
DECLARE
    p_id uuid;
    job_count int := 0;
    dag_count int := 0;
    temp_count int;
BEGIN
    FOREACH p_id IN ARRAY p_dag_ids
    LOOP
        DELETE FROM marie_scheduler.job WHERE dag_id = p_id;
        GET DIAGNOSTICS temp_count = ROW_COUNT;
        job_count := job_count + temp_count;

        DELETE FROM marie_scheduler.dag WHERE id = p_id;
        GET DIAGNOSTICS temp_count = ROW_COUNT;
        dag_count := dag_count + temp_count;
    END LOOP;

    RAISE NOTICE 'Deleted % job(s) and % dag(s)', job_count, dag_count;
END;
$$;