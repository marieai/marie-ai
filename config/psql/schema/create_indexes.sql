create index idx_job_name_state_start
    on marie_scheduler.job (name, state, start_after);

create index idx_job_id_state
    on marie_scheduler.job (id, state);

create index idx_dependencies_gin
    on marie_scheduler.job using gin (dependencies jsonb_path_ops);

create index idx_dag_id_state
    on marie_scheduler.dag (id, state);

-- Used for job prioritization
CREATE INDEX idx_job_hard_sla_due
  ON marie_scheduler.job (hard_sla)
  WHERE hard_sla IS NOT NULL;

CREATE INDEX idx_job_soft_sla_due
  ON marie_scheduler.job (soft_sla)
  WHERE soft_sla IS NOT NULL;

CREATE INDEX idx_job_not_completed
    ON marie_scheduler.job (id)
    WHERE state <> 'completed';


-- For fast filtering
CREATE INDEX ON marie_scheduler.job (state, start_after, name);
CREATE INDEX ON marie_scheduler.job (name, id, state);
CREATE INDEX  job_name_state_idx ON marie_scheduler.job (name, state);


CREATE INDEX IF NOT EXISTS idx_job_dag_id ON marie_scheduler.job (dag_id);
CREATE INDEX IF NOT EXISTS idx_job_name_dag_id ON marie_scheduler.job (name, dag_id);

-- For dependency resolution
CREATE INDEX idx_dep_job_id ON marie_scheduler.job_dependencies (job_id);
CREATE INDEX idx_dep_depends_on_id ON marie_scheduler.job_dependencies (depends_on_id);


CREATE INDEX idx_dep_job_id_dep_on_id ON marie_scheduler.job_dependencies (job_id, depends_on_id);
CREATE INDEX idx_dep_depends_on_id_state ON marie_scheduler.job_dependencies (depends_on_id) INCLUDE (job_id);

-- used by count_dag_states
CREATE INDEX  IF NOT EXISTS job_dag_id_name_idx ON marie_scheduler.job (dag_id, name);
