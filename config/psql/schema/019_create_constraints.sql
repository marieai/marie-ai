DO $$
BEGIN
    ALTER TABLE {schema}.job
    ADD CONSTRAINT job_dag_id_fkey
    FOREIGN KEY (dag_id) REFERENCES {schema}.dag(id) ON DELETE CASCADE;
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;
