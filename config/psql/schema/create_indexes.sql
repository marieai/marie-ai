create index idx_job_name_state_start
    on marie_scheduler.job (name, state, start_after);

create index idx_job_id_state
    on marie_scheduler.job (id, state);

create index idx_dependencies_gin
    on marie_scheduler.job using gin (dependencies jsonb_path_ops);

create index idx_dag_id_state
    on marie_scheduler.dag (id, state);

-- Used for job proritization
CREATE INDEX idx_job_hard_sla_due
  ON marie_scheduler.job (hard_sla)
  WHERE hard_sla IS NOT NULL;

CREATE INDEX idx_job_soft_sla_due
  ON marie_scheduler.job (soft_sla)
  WHERE soft_sla IS NOT NULL;

CREATE INDEX idx_job_not_completed
    ON marie_scheduler.job (id)
    WHERE state <> 'completed';
