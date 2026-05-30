-- File: 017_job_dependencies.sql
-- Description: Job dependencies tracking table
-- Dependencies: 005_job.sql

CREATE TABLE IF NOT EXISTS {schema}.job_dependencies (
    job_name TEXT NOT NULL,
    job_id UUID NOT NULL,
    depends_on_name TEXT NOT NULL,
    depends_on_id UUID NOT NULL,
    PRIMARY KEY (job_name, job_id, depends_on_name, depends_on_id),
    FOREIGN KEY (job_name, job_id) REFERENCES {schema}.job(name, id) ON DELETE CASCADE,
    FOREIGN KEY (depends_on_name, depends_on_id) REFERENCES {schema}.job(name, id) ON DELETE CASCADE
);

COMMENT ON TABLE {schema}.job_dependencies IS 'Tracks job-to-job dependencies for DAG execution';
