create index idx_job_name_state_start
    on marie_scheduler.job (name, state, start_after);

create index idx_job_id_state
    on marie_scheduler.job (id, state);

create index idx_dependencies_gin
    on marie_scheduler.job using gin (dependencies jsonb_path_ops);

create index idx_dag_id_state
    on marie_scheduler.dag (id, state);