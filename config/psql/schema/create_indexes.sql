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

-- On the parent, so every partition gets one:
CREATE INDEX IF NOT EXISTS job_state_idx ON marie_scheduler.job (state);

-- For candidate selection
CREATE INDEX IF NOT EXISTS job_name_state_start_after_idx ON marie_scheduler.job (name, state, start_after);

-- lookup of non-completed deps by depends_on_id
CREATE INDEX IF NOT EXISTS idx_job_id_not_completed ON marie_scheduler.job (id) WHERE state <> 'completed';


CREATE INDEX IF NOT EXISTS job_extract_ready_idx_partial ON marie_scheduler.job (state, start_after)
INCLUDE (id, dag_id)
WHERE state IN ('created','retry');

CREATE INDEX IF NOT EXISTS job_name_state_start_after_ready_idx ON marie_scheduler.job (name, state, start_after)
INCLUDE (id, dag_id)
WHERE state IN ('created','retry');


-- Ready jobs across all partitions (parent = partitioned index)
CREATE INDEX IF NOT EXISTS job_state_start_after_ready_idx
ON marie_scheduler.job (state, start_after, dag_id)
INCLUDE (id, name)
WHERE state IN ('created','retry');


CREATE INDEX IF NOT EXISTS idx_job_dag_id ON marie_scheduler.job (dag_id);
CREATE INDEX IF NOT EXISTS idx_job_name_dag_id ON marie_scheduler.job (name, dag_id);

-- For dependency resolution
CREATE INDEX idx_dep_job_id ON marie_scheduler.job_dependencies (job_id);
CREATE INDEX idx_dep_depends_on_id ON marie_scheduler.job_dependencies (depends_on_id);

CREATE INDEX idx_dep_job_id_dep_on_id ON marie_scheduler.job_dependencies (job_id, depends_on_id);
CREATE INDEX idx_dep_depends_on_dep_on_job_id ON marie_scheduler.job_dependencies (depends_on_id, job_id);


CREATE INDEX jobname_jobid_idx ON marie_scheduler.job_dependencies (job_name, job_id);
CREATE INDEX depname_depid_idx ON marie_scheduler.job_dependencies (depends_on_name, depends_on_id);


-- used by count_dag_states
CREATE INDEX  IF NOT EXISTS job_dag_id_name_idx ON marie_scheduler.job (dag_id, name);


-- DAG: avoid scanning tons of completed/failed/cancelled rows and kill heap fetches
CREATE INDEX IF NOT EXISTS dag_id_state_not_bad_idx
ON marie_scheduler.dag (id, state)
WHERE state NOT IN ('completed','failed','cancelled');

CREATE INDEX  IF NOT EXISTS dag_ok_idx
ON marie_scheduler.dag (id)
WHERE state NOT IN ('completed','failed','cancelled');


CREATE INDEX IF NOT EXISTS job_id_failed_idx ON marie_scheduler.job (id) WHERE state = 'failed';

WE NEED TO DO THIS PER PARTITION:
-- Speeds the initial candidate scan; covers (id, dag_id) so no heap.
CREATE INDEX IF NOT EXISTS j5294dca0cf67eba9f6066f08560c47b010e0dce4a3ef60ff128d306e_ready_cover_idx
ON marie_scheduler.j5294dca0cf67eba9f6066f08560c47b010e0dce4a3ef60ff128d306e -- for each partition
(name, start_after, id, dag_id)
INCLUDE (state)
WHERE state IN ('created','retry');




