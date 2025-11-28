-- File: create_indexes.sql
-- Description: Index definitions for scheduler tables
-- Dependencies: 02_tables/*.sql

-- Job table indexes
CREATE INDEX IF NOT EXISTS idx_job_name_state_start
    ON {schema}.job (name, state, start_after);

CREATE INDEX IF NOT EXISTS idx_job_id_state
    ON {schema}.job (id, state);

CREATE INDEX IF NOT EXISTS idx_dependencies_gin
    ON {schema}.job USING gin (dependencies jsonb_path_ops);

-- DAG table indexes
CREATE INDEX IF NOT EXISTS idx_dag_id_state
    ON {schema}.dag (id, state);

-- Used for job prioritization
CREATE INDEX IF NOT EXISTS idx_job_hard_sla_due
    ON {schema}.job (hard_sla)
    WHERE hard_sla IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_job_soft_sla_due
    ON {schema}.job (soft_sla)
    WHERE soft_sla IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_job_not_completed
    ON {schema}.job (id)
    WHERE state <> 'completed';

-- For fast filtering
CREATE INDEX IF NOT EXISTS idx_job_state_start_name
    ON {schema}.job (state, start_after, name);

CREATE INDEX IF NOT EXISTS idx_job_name_id_state
    ON {schema}.job (name, id, state);

CREATE INDEX IF NOT EXISTS job_name_state_idx
    ON {schema}.job (name, state);

-- On the parent, so every partition gets one
CREATE INDEX IF NOT EXISTS job_state_idx
    ON {schema}.job (state);

-- For candidate selection
CREATE INDEX IF NOT EXISTS job_name_state_start_after_idx
    ON {schema}.job (name, state, start_after);

-- Lookup of non-completed deps by depends_on_id
CREATE INDEX IF NOT EXISTS idx_job_id_not_completed
    ON {schema}.job (id)
    WHERE state <> 'completed';

CREATE INDEX IF NOT EXISTS job_extract_ready_idx_partial
    ON {schema}.job (state, start_after)
    INCLUDE (id, dag_id)
    WHERE state IN ('created', 'retry');

CREATE INDEX IF NOT EXISTS job_name_state_start_after_ready_idx
    ON {schema}.job (name, state, start_after)
    INCLUDE (id, dag_id)
    WHERE state IN ('created', 'retry');

-- Ready jobs across all partitions (parent = partitioned index)
CREATE INDEX IF NOT EXISTS job_state_start_after_ready_idx
    ON {schema}.job (state, start_after, dag_id)
    INCLUDE (id, name)
    WHERE state IN ('created', 'retry');

CREATE INDEX IF NOT EXISTS idx_job_dag_id
    ON {schema}.job (dag_id);

CREATE INDEX IF NOT EXISTS idx_job_name_dag_id
    ON {schema}.job (name, dag_id);

-- For dependency resolution
CREATE INDEX IF NOT EXISTS idx_dep_job_id
    ON {schema}.job_dependencies (job_id);

CREATE INDEX IF NOT EXISTS idx_dep_depends_on_id
    ON {schema}.job_dependencies (depends_on_id);

CREATE INDEX IF NOT EXISTS idx_dep_job_id_dep_on_id
    ON {schema}.job_dependencies (job_id, depends_on_id);

CREATE INDEX IF NOT EXISTS idx_dep_depends_on_dep_on_job_id
    ON {schema}.job_dependencies (depends_on_id, job_id);

CREATE INDEX IF NOT EXISTS jobname_jobid_idx
    ON {schema}.job_dependencies (job_name, job_id);

CREATE INDEX IF NOT EXISTS depname_depid_idx
    ON {schema}.job_dependencies (depends_on_name, depends_on_id);

-- Used by count_dag_states
CREATE INDEX IF NOT EXISTS job_dag_id_name_idx
    ON {schema}.job (dag_id, name);

-- DAG: avoid scanning tons of completed/failed/cancelled rows and kill heap fetches
CREATE INDEX IF NOT EXISTS dag_id_state_not_bad_idx
    ON {schema}.dag (id, state)
    WHERE state NOT IN ('completed', 'failed', 'cancelled');

CREATE INDEX IF NOT EXISTS dag_ok_idx
    ON {schema}.dag (id)
    WHERE state NOT IN ('completed', 'failed', 'cancelled');

CREATE INDEX IF NOT EXISTS job_id_failed_idx
    ON {schema}.job (id)
    WHERE state = 'failed';

-- Note: Per-partition covering indexes for ready job scans should be created
-- dynamically when queues are created via the create_queue function.
-- Example pattern for partition-specific index:
-- CREATE INDEX IF NOT EXISTS <partition_name>_ready_cover_idx
-- ON {schema}.<partition_name> (name, start_after, id, dag_id)
-- INCLUDE (state)
-- WHERE state IN ('created', 'retry');
