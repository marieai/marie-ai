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


 CREATE OR replace FUNCTION marie_scheduler.delete_dags_and_jobs( p_dag_ids UUID[] ) returns void LANGUAGE plpgsql
AS
  $$
  DECLARE
    v_jobs_deleted bigint := 0;
    v_dags_deleted bigint := 0;
    v_batch        bigint;
    v_batch_rows   INTEGER := 20000;
  BEGIN
    IF p_dag_ids IS NULL
      OR
      array_length(p_dag_ids,1) IS NULL THEN
      RAISE notice 'No DAG ids provided';
      RETURN;
    END IF;
    -- Stage IDs for better plans than ANY($1)
    CREATE temp TABLE _del_dag_ids(id uuid PRIMARY KEY) ON COMMIT DROP;
    INSERT INTO _del_dag_ids
                (
                            id
                )
    SELECT DISTINCT unnest(p_dag_ids);

    -- 1) Delete JOBS referencing those DAGs, in batches
    LOOP
      WITH dels AS
      (
             SELECT j.id
             FROM   marie_scheduler.job j
             join   _del_dag_ids d
             ON     j.dag_id = d.id limit v_batch_rows )
      DELETE
      FROM   marie_scheduler.job j
      USING  dels d
      WHERE  j.id = d.id;

      GET diagnostics v_batch = row_count;
      v_jobs_deleted := v_jobs_deleted + v_batch;
      EXIT
    WHEN v_batch = 0;
    END LOOP;
    -- 2) Delete the DAGs themselves, in batches
    LOOP
      WITH dels AS
      (
             SELECT d.id
             FROM   marie_scheduler.dag d
             join   _del_dag_ids x
             ON     d.id = x.id limit v_batch_rows )
      DELETE
      FROM   marie_scheduler.dag d
      USING  dels z
      WHERE  d.id = z.id;

      GET diagnostics v_batch = row_count;
      v_dags_deleted := v_dags_deleted + v_batch;
      EXIT
    WHEN v_batch = 0;
    END LOOP;
    RAISE notice 'Deleted % jobs and % DAGs', v_jobs_deleted, v_dags_deleted;
  END;
  $$;
  ALTER FUNCTION marie_scheduler.delete_dags_and_jobs(uuid[]) owner TO postgres;