-- File: 018_create_indexes.sql
-- Description: Minimal scheduler indexes for the unpartitioned active job table

-- Hydrate all ready jobs for a bounded set of DAGs in frontier order.
CREATE INDEX IF NOT EXISTS job_u_hydrate_frontier_idx
    ON {schema}.job (dag_id, job_level, created_on, id)
    WHERE state IN ('created', 'retry');

-- Admission performs a DAG-correlated readiness probe.
CREATE INDEX IF NOT EXISTS job_u_admission_ready_idx
    ON {schema}.job (dag_id, start_after)
    WHERE state IN ('created', 'retry');

CREATE INDEX IF NOT EXISTS job_u_admission_blocker_idx
    ON {schema}.job (dag_id)
    WHERE state IN ('failed', 'expired', 'cancelled');

-- Queue-local scheduling remains indexed even though queues are no longer
-- physical tables. The global primary key handles ID-only lifecycle updates.
CREATE INDEX IF NOT EXISTS job_u_queue_ready_order_idx
    ON {schema}.job (name, job_level DESC, priority DESC, id)
    INCLUDE (dag_id, start_after)
    WHERE state IN ('created', 'retry');

CREATE INDEX IF NOT EXISTS job_u_dag_state_idx
    ON {schema}.job (dag_id, state);

CREATE INDEX IF NOT EXISTS job_u_hard_sla_idx
    ON {schema}.job (hard_sla)
    WHERE hard_sla IS NOT NULL;

CREATE INDEX IF NOT EXISTS job_u_soft_sla_idx
    ON {schema}.job (soft_sla)
    WHERE soft_sla IS NOT NULL;

-- Normalized dependency traversal. The primary key already covers
-- (job_name, job_id, depends_on_name, depends_on_id).
CREATE INDEX IF NOT EXISTS idx_dep_job_id_dep_on_id
    ON {schema}.job_dependencies (job_id, depends_on_id);

CREATE INDEX IF NOT EXISTS idx_dep_depends_on_dep_on_job_id
    ON {schema}.job_dependencies (depends_on_id, job_id);

CREATE INDEX IF NOT EXISTS depname_depid_idx
    ON {schema}.job_dependencies (depends_on_name, depends_on_id);

-- DAG access paths retained from the measured admission workload.
CREATE INDEX IF NOT EXISTS dag_id_state_not_bad_idx
    ON {schema}.dag (id, state)
    WHERE state NOT IN ('completed', 'failed', 'cancelled');

CREATE INDEX IF NOT EXISTS dag_ok_idx
    ON {schema}.dag (id)
    WHERE state NOT IN ('completed', 'failed', 'cancelled');

CREATE INDEX IF NOT EXISTS dag_admission_active_idx
    ON {schema}.dag (id)
    INCLUDE (soft_sla, hard_sla, created_on)
    WHERE state IN ('created', 'active');
