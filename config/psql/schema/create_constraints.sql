ALTER TABLE marie_scheduler.job
ADD CONSTRAINT job_dag_id_fkey
FOREIGN KEY (dag_id) REFERENCES marie_scheduler.dag(id) ON DELETE CASCADE;