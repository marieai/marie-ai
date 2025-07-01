CREATE TABLE marie_scheduler.job_dependencies (
    job_name  text NOT NULL,
    job_id    uuid NOT NULL,
    depends_on_name text NOT NULL,
    depends_on_id   uuid NOT NULL,
    PRIMARY KEY (job_name, job_id, depends_on_name, depends_on_id),
    FOREIGN KEY (job_name, job_id) REFERENCES marie_scheduler.job(name, id) ON DELETE CASCADE,
    FOREIGN KEY (depends_on_name, depends_on_id) REFERENCES marie_scheduler.job(name, id) ON DELETE CASCADE
);
