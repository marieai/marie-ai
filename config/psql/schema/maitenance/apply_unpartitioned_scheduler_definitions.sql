-- Reconcile scheduler functions and indexes after the partitioned-to-
-- unpartitioned job-table cutover.
--
-- Copy and execute this complete file in one PostgreSQL session while every
-- gateway, scheduler, executor-terminal, recovery, and administrative writer
-- is stopped. The script contains no psql meta-commands and can be executed
-- from DataGrip or another SQL console.
--
-- This script does not alter, drop, or finalize job_partitioned_old.

BEGIN;

SET LOCAL statement_timeout = 0;
SET LOCAL lock_timeout = '5s';
SET LOCAL idle_in_transaction_session_timeout = 0;

DO $preflight$
DECLARE
    active_kind "char";
    active_partitions bigint;
BEGIN
    SELECT relation.relkind
    INTO active_kind
    FROM pg_class AS relation
    WHERE relation.oid = 'marie_scheduler.job'::regclass;

    IF active_kind <> 'r' THEN
        RAISE EXCEPTION
            'marie_scheduler.job must be unpartitioned (relkind=r), found %',
            active_kind;
    END IF;

    SELECT count(*)
    INTO active_partitions
    FROM pg_inherits
    WHERE inhparent = 'marie_scheduler.job'::regclass;

    IF active_partitions <> 0 THEN
        RAISE EXCEPTION
            'marie_scheduler.job still has % active partitions',
            active_partitions;
    END IF;
END
$preflight$;

-- Fail instead of waiting behind an application writer. Keep these locks for
-- the definition and index transaction so no scheduler state changes midway.
LOCK TABLE
    marie_scheduler.job,
    marie_scheduler.job_attempt,
    marie_scheduler.job_dependencies,
    marie_scheduler.dag,
    marie_scheduler.job_history,
    marie_scheduler.dag_history
IN SHARE MODE NOWAIT;

-- File: 014_create_queue.sql
-- Description: Function to create logical queue metadata
-- Dependencies: 004_queue.sql, 005_job.sql

-- Create or replace the queue creation function (idempotent)
CREATE OR REPLACE FUNCTION marie_scheduler.create_queue(queue_name TEXT, options JSON)
RETURNS VOID AS
$$
BEGIN
    INSERT INTO marie_scheduler.queue (
        name,
        policy,
        retry_limit,
        retry_delay,
        retry_backoff,
        expire_seconds,
        retention_minutes,
        dead_letter
    )
    VALUES (
        queue_name,
        options->>'policy',
        (options->>'retry_limit')::INT,
        (options->>'retry_delay')::INT,
        (options->>'retry_backoff')::BOOL,
        (options->>'expire_in_seconds')::INT,
        (options->>'retention_minutes')::INT,
        options->>'dead_letter'
    )
    ON CONFLICT (name) DO NOTHING;
END;
$$
LANGUAGE plpgsql;

-- File: 015_delete_queue.sql
-- Description: Function to delete a logical queue and its retained jobs
-- Dependencies: 004_queue.sql

-- Create or replace the queue deletion function (idempotent)
CREATE OR REPLACE FUNCTION marie_scheduler.delete_queue(queue_name TEXT)
RETURNS VOID AS
$$
BEGIN
    -- Job foreign keys cascade to dependency, HITL, and search-projection rows.
    DELETE FROM marie_scheduler.job AS job
    WHERE job.name = delete_queue.queue_name;

    DELETE FROM marie_scheduler.queue AS queue
    WHERE queue.name = delete_queue.queue_name;
END;
$$
LANGUAGE plpgsql;

-- File: 018_create_indexes.sql
-- Description: Minimal scheduler indexes for the unpartitioned active job table

-- Hydrate all ready jobs for a bounded set of DAGs in frontier order.
CREATE INDEX IF NOT EXISTS job_u_hydrate_frontier_idx
    ON marie_scheduler.job (dag_id, job_level, created_on, id)
    WHERE state IN ('created', 'retry');

-- Admission performs a DAG-correlated readiness probe.
CREATE INDEX IF NOT EXISTS job_u_admission_ready_idx
    ON marie_scheduler.job (dag_id, start_after)
    WHERE state IN ('created', 'retry');

CREATE INDEX IF NOT EXISTS job_u_admission_blocker_idx
    ON marie_scheduler.job (dag_id)
    WHERE state IN ('failed', 'expired', 'cancelled');

-- Queue-local scheduling remains indexed even though queues are no longer
-- physical tables. The global primary key handles ID-only lifecycle updates.
CREATE INDEX IF NOT EXISTS job_u_queue_ready_order_idx
    ON marie_scheduler.job (name, job_level DESC, priority DESC, id)
    INCLUDE (dag_id, start_after)
    WHERE state IN ('created', 'retry');

CREATE INDEX IF NOT EXISTS job_u_dag_state_idx
    ON marie_scheduler.job (dag_id, state);

CREATE INDEX IF NOT EXISTS job_u_hard_sla_idx
    ON marie_scheduler.job (hard_sla)
    WHERE hard_sla IS NOT NULL;

CREATE INDEX IF NOT EXISTS job_u_soft_sla_idx
    ON marie_scheduler.job (soft_sla)
    WHERE soft_sla IS NOT NULL;

-- Normalized dependency traversal. The primary key already covers
-- (job_name, job_id, depends_on_name, depends_on_id).
CREATE INDEX IF NOT EXISTS idx_dep_job_id_dep_on_id
    ON marie_scheduler.job_dependencies (job_id, depends_on_id);

CREATE INDEX IF NOT EXISTS idx_dep_depends_on_dep_on_job_id
    ON marie_scheduler.job_dependencies (depends_on_id, job_id);

CREATE INDEX IF NOT EXISTS depname_depid_idx
    ON marie_scheduler.job_dependencies (depends_on_name, depends_on_id);

-- DAG access paths retained from the measured admission workload.
CREATE INDEX IF NOT EXISTS dag_id_state_not_bad_idx
    ON marie_scheduler.dag (id, state)
    WHERE state NOT IN ('completed', 'failed', 'cancelled');

CREATE INDEX IF NOT EXISTS dag_ok_idx
    ON marie_scheduler.dag (id)
    WHERE state NOT IN ('completed', 'failed', 'cancelled');

CREATE INDEX IF NOT EXISTS dag_admission_active_idx
    ON marie_scheduler.dag (id)
    INCLUDE (soft_sla, hard_sla, created_on)
    WHERE state IN ('created', 'active');

-- Reset every DAG and its jobs to a fresh schedulable state.
CREATE OR REPLACE FUNCTION marie_scheduler.reset_all()
RETURNS void
LANGUAGE plpgsql
AS $$
DECLARE
    reset_at TIMESTAMPTZ := statement_timestamp();
    dag_count BIGINT;
    job_count BIGINT;
BEGIN
    ALTER TABLE marie_scheduler.job
        DISABLE TRIGGER job_update_state_trigger;
    ALTER TABLE marie_scheduler.dag
        DISABLE TRIGGER dag_update_state_trigger;
    ALTER TABLE marie_scheduler.dag
        DISABLE TRIGGER trg_dag_state_changed;

    UPDATE marie_scheduler.job
    SET state = 'created',
        started_on = NULL,
        completed_on = NULL,
        start_after = reset_at,
        retry_count = 0,
        output = NULL,
        duration = NULL,
        sla_miss_logged = FALSE,
        branch_metadata = NULL,
        lease_owner = NULL,
        lease_expires_at = NULL,
        lease_epoch = 0,
        run_owner = NULL,
        run_attempt_id = NULL,
        run_lease_expires_at = NULL
    WHERE state IS DISTINCT FROM 'created'
       OR started_on IS NOT NULL
       OR completed_on IS NOT NULL
       OR start_after > reset_at
       OR retry_count <> 0
       OR output IS NOT NULL
       OR duration IS NOT NULL
       OR sla_miss_logged
       OR branch_metadata IS NOT NULL
       OR lease_owner IS NOT NULL
       OR lease_expires_at IS NOT NULL
       OR lease_epoch IS DISTINCT FROM 0
       OR run_owner IS NOT NULL
       OR run_attempt_id IS NOT NULL
       OR run_lease_expires_at IS NOT NULL;
    GET DIAGNOSTICS job_count = ROW_COUNT;

    UPDATE marie_scheduler.dag
    SET state = 'created',
        started_on = NULL,
        completed_on = NULL,
        updated_on = reset_at,
        duration = NULL,
        sla_miss_logged = FALSE
    WHERE state IS DISTINCT FROM 'created'
       OR started_on IS NOT NULL
       OR completed_on IS NOT NULL
       OR duration IS NOT NULL
       OR sla_miss_logged;
    GET DIAGNOSTICS dag_count = ROW_COUNT;

    ALTER TABLE marie_scheduler.job
        ENABLE TRIGGER job_update_state_trigger;
    ALTER TABLE marie_scheduler.dag
        ENABLE TRIGGER dag_update_state_trigger;
    ALTER TABLE marie_scheduler.dag
        ENABLE TRIGGER trg_dag_state_changed;

    RAISE NOTICE 'Reset % DAG(s) and % job(s) to a fresh schedulable state.',
        dag_count, job_count;
END;
$$;

COMMENT ON FUNCTION marie_scheduler.reset_all() IS
'Reset DAGs and jobs with execution residue while preserving existing dependency and audit rows. Run with scheduler writers stopped; bulk resets do not append per-row history or DAG notifications.';

-- File: 043_branch_metadata_indexes.sql
-- Migration: Add branch_metadata indexes and comments
-- Date: 2025-11-13
-- Description: Adds indexes and documentation for branch_metadata column
-- Dependencies: 005_job.sql, 006_job_history.sql

-- Add comment explaining the column
COMMENT ON COLUMN marie_scheduler.job.branch_metadata IS
'Branch execution metadata for tracking conditional execution flow.
Contains node_type, selected_path_ids, skip_reason, etc.';

COMMENT ON COLUMN marie_scheduler.job_history.branch_metadata IS
'Historical branch execution metadata for audit and debugging.';

-- Create GIN index for efficient JSON queries
CREATE INDEX IF NOT EXISTS job_u_branch_metadata_idx
ON marie_scheduler.job USING gin(branch_metadata);

-- Create partial index for quickly finding skipped jobs
CREATE INDEX IF NOT EXISTS job_u_branch_skipped_idx
ON marie_scheduler.job ((branch_metadata->>'skipped'))
WHERE branch_metadata->>'skipped' = 'true';

-- Create partial index for finding BRANCH/SWITCH nodes
CREATE INDEX IF NOT EXISTS job_u_branch_node_type_idx
ON marie_scheduler.job ((branch_metadata->>'node_type'))
WHERE branch_metadata->>'node_type' IN ('BRANCH', 'SWITCH');

-- Index maintenance scans by expiration time so empty cycles do not scan the
-- ready and active working sets.

CREATE INDEX IF NOT EXISTS job_u_expired_acquisition_lease_idx
    ON marie_scheduler.job (lease_expires_at, id)
    WHERE state IN ('created', 'retry')
      AND lease_owner IS NOT NULL
      AND lease_expires_at IS NOT NULL;

CREATE INDEX IF NOT EXISTS job_u_expired_run_lease_idx
    ON marie_scheduler.job (run_lease_expires_at, id)
    WHERE state = 'active'
      AND run_owner IS NOT NULL
      AND run_attempt_id IS NOT NULL
      AND run_lease_expires_at IS NOT NULL;

CREATE INDEX IF NOT EXISTS dag_admission_order_idx
    ON marie_scheduler.dag (
        priority DESC,
        (COALESCE(soft_sla, hard_sla)),
        created_on,
        id
    )
    WHERE state IN ('created', 'active');

-- File: 075_list_operational_jobs.sql
-- Description: Payload-free operational job page
-- Dependencies: 004_queue.sql, 005_job.sql, 007_dag.sql, 065_job_attempt.sql

CREATE OR REPLACE FUNCTION marie_scheduler.list_operational_jobs(
    p_limit INTEGER DEFAULT 25,
    p_offset INTEGER DEFAULT 0,
    p_states TEXT[] DEFAULT NULL,
    p_attention TEXT DEFAULT 'any',
    p_queue TEXT DEFAULT NULL,
    p_search TEXT DEFAULT NULL,
    p_sort TEXT DEFAULT 'attention',
    p_dag_id UUID DEFAULT NULL,
    p_queued_too_long_seconds INTEGER DEFAULT 300,
    p_running_too_long_seconds INTEGER DEFAULT 900,
    p_stale_update_seconds INTEGER DEFAULT 600
)
RETURNS TABLE (
    total_count BIGINT,
    queue_facets TEXT[],
    job_id UUID,
    queue_name TEXT,
    job_state TEXT,
    dag_id UUID,
    dag_name TEXT,
    planner TEXT,
    priority INTEGER,
    job_level INTEGER,
    retry_count INTEGER,
    retry_limit INTEGER,
    created_on TIMESTAMPTZ,
    started_on TIMESTAMPTZ,
    completed_on TIMESTAMPTZ,
    last_updated_on TIMESTAMPTZ,
    age_seconds DOUBLE PRECISION,
    last_update_age_seconds DOUBLE PRECISION,
    run_owner TEXT,
    run_attempt_id UUID,
    executor TEXT,
    attempt_activated_at TIMESTAMPTZ,
    attempt_terminal_at TIMESTAMPTZ,
    terminal_status TEXT,
    terminal_work_state TEXT,
    terminal_source TEXT,
    terminal_accepted BOOLEAN
)
LANGUAGE SQL
STABLE
PARALLEL SAFE
AS $function$
WITH page_parameters AS (
    SELECT GREATEST(p_limit, 0) + GREATEST(p_offset, 0) AS page_size
), eligible_jobs AS NOT MATERIALIZED (
    SELECT
        j.id AS job_id,
        j.name AS queue_name,
        j.state AS job_state,
        j.dag_id,
        j.priority,
        j.job_level,
        j.retry_count,
        j.retry_limit,
        j.created_on,
        j.started_on,
        j.completed_on,
        j.run_owner,
        j.run_attempt_id
    FROM marie_scheduler.job AS j
    WHERE (
            p_states IS NULL
            OR CARDINALITY(p_states) = 0
            OR j.state::TEXT = ANY(p_states)
        )
      AND (COALESCE(p_queue, '') = '' OR j.name = p_queue)
      AND (p_dag_id IS NULL OR j.dag_id = p_dag_id)
      AND (
            COALESCE(p_search, '') = ''
            OR j.id::TEXT ILIKE '%' || p_search || '%'
            OR j.name ILIKE '%' || p_search || '%'
            OR j.dag_id::TEXT ILIKE '%' || p_search || '%'
            OR COALESCE(j.run_owner, '') ILIKE '%' || p_search || '%'
            OR EXISTS (
                SELECT 1
                FROM marie_scheduler.dag AS searched_dag
                WHERE searched_dag.id = j.dag_id
                  AND (
                        COALESCE(searched_dag.name, '')
                            ILIKE '%' || p_search || '%'
                        OR COALESCE(searched_dag.planner, '')
                            ILIKE '%' || p_search || '%'
                    )
            )
            OR EXISTS (
                SELECT 1
                FROM marie_scheduler.job_attempt AS searched_attempt
                WHERE searched_attempt.run_attempt_id = j.run_attempt_id
                  AND COALESCE(searched_attempt.executor, '')
                        ILIKE '%' || p_search || '%'
            )
        )
), page_metadata AS (
    SELECT COUNT(*) AS total_count
    FROM eligible_jobs AS eligible
    LEFT JOIN marie_scheduler.job_attempt AS attempt
      ON attempt.run_attempt_id = eligible.run_attempt_id
    WHERE CASE p_attention
        WHEN 'any' THEN TRUE
        WHEN 'queued_too_long' THEN
            eligible.job_state IN ('created', 'retry')
            AND COALESCE(
                attempt.updated_on,
                eligible.completed_on,
                eligible.started_on,
                eligible.created_on
            ) < NOW() - MAKE_INTERVAL(secs => p_queued_too_long_seconds)
        WHEN 'running_too_long' THEN
            eligible.job_state = 'active'
            AND eligible.started_on IS NOT NULL
            AND eligible.started_on
                < NOW() - MAKE_INTERVAL(secs => p_running_too_long_seconds)
        WHEN 'stale_update' THEN
            eligible.job_state IN ('active', 'retry')
            AND COALESCE(
                attempt.updated_on,
                eligible.completed_on,
                eligible.started_on,
                eligible.created_on
            ) < NOW() - MAKE_INTERVAL(secs => p_stale_update_seconds)
        WHEN 'retrying' THEN eligible.job_state = 'retry'
        WHEN 'failed' THEN
            eligible.job_state IN ('failed', 'expired', 'cancelled')
        WHEN 'terminal_mismatch' THEN
            eligible.run_attempt_id IS NOT NULL
            AND eligible.job_state IN (
                'completed',
                'skipped',
                'failed',
                'expired',
                'cancelled'
            )
            AND (
                attempt.terminal_accepted IS FALSE
                OR (
                    attempt.terminal_work_state IS NOT NULL
                    AND attempt.terminal_work_state
                        <> eligible.job_state::TEXT
                )
            )
        ELSE FALSE
    END
), facets AS (
    SELECT CASE
        WHEN p_dag_id IS NULL THEN COALESCE(
            (SELECT ARRAY_AGG(name ORDER BY name) FROM marie_scheduler.queue),
            ARRAY[]::TEXT[]
        )
        ELSE ARRAY[]::TEXT[]
    END AS queue_facets
), terminal_mismatch_page AS MATERIALIZED (
    SELECT
        eligible.job_id,
        eligible.created_on,
        0 AS attention_rank
    FROM eligible_jobs AS eligible
    JOIN marie_scheduler.job_attempt AS attempt
      ON attempt.run_attempt_id = eligible.run_attempt_id
    WHERE p_attention = 'any'
      AND p_sort = 'attention'
      AND eligible.run_attempt_id IS NOT NULL
      AND eligible.job_state IN (
            'completed',
            'skipped',
            'failed',
            'expired',
            'cancelled'
        )
      AND (
            attempt.terminal_accepted IS FALSE
            OR (
                attempt.terminal_work_state IS NOT NULL
                AND attempt.terminal_work_state <> eligible.job_state::TEXT
            )
        )
    ORDER BY eligible.created_on DESC, eligible.job_id
    LIMIT (SELECT page_size FROM page_parameters)
), failed_page AS MATERIALIZED (
    SELECT
        eligible.job_id,
        eligible.created_on,
        1 AS attention_rank
    FROM eligible_jobs AS eligible
    WHERE p_attention = 'any'
      AND p_sort = 'attention'
      AND eligible.job_state IN ('failed', 'expired', 'cancelled')
      AND NOT EXISTS (
            SELECT 1
            FROM terminal_mismatch_page AS prior
            WHERE prior.job_id = eligible.job_id
        )
    ORDER BY eligible.created_on DESC, eligible.job_id
    LIMIT GREATEST(
        (SELECT page_size FROM page_parameters)
            - (SELECT COUNT(*) FROM terminal_mismatch_page),
        0
    )
), stale_page AS MATERIALIZED (
    SELECT
        eligible.job_id,
        eligible.created_on,
        2 AS attention_rank
    FROM eligible_jobs AS eligible
    WHERE p_attention = 'any'
      AND p_sort = 'attention'
      AND eligible.job_state IN ('active', 'retry')
      AND COALESCE(
            (
                SELECT attempt.updated_on
                FROM marie_scheduler.job_attempt AS attempt
                WHERE attempt.run_attempt_id = eligible.run_attempt_id
            ),
            eligible.completed_on,
            eligible.started_on,
            eligible.created_on
        ) < NOW() - MAKE_INTERVAL(secs => p_stale_update_seconds)
    ORDER BY eligible.created_on DESC, eligible.job_id
    LIMIT GREATEST(
        (SELECT page_size FROM page_parameters)
            - (SELECT COUNT(*) FROM terminal_mismatch_page)
            - (SELECT COUNT(*) FROM failed_page),
        0
    )
), running_page AS MATERIALIZED (
    SELECT
        eligible.job_id,
        eligible.created_on,
        3 AS attention_rank
    FROM eligible_jobs AS eligible
    WHERE p_attention = 'any'
      AND p_sort = 'attention'
      AND eligible.job_state = 'active'
      AND eligible.started_on IS NOT NULL
      AND eligible.started_on
            < NOW() - MAKE_INTERVAL(secs => p_running_too_long_seconds)
      AND NOT EXISTS (
            SELECT 1
            FROM stale_page AS prior
            WHERE prior.job_id = eligible.job_id
        )
    ORDER BY eligible.created_on DESC, eligible.job_id
    LIMIT GREATEST(
        (SELECT page_size FROM page_parameters)
            - (SELECT COUNT(*) FROM terminal_mismatch_page)
            - (SELECT COUNT(*) FROM failed_page)
            - (SELECT COUNT(*) FROM stale_page),
        0
    )
), queued_page AS MATERIALIZED (
    SELECT
        eligible.job_id,
        eligible.created_on,
        4 AS attention_rank
    FROM eligible_jobs AS eligible
    WHERE p_attention = 'any'
      AND p_sort = 'attention'
      AND eligible.job_state IN ('created', 'retry')
      AND COALESCE(
            (
                SELECT attempt.updated_on
                FROM marie_scheduler.job_attempt AS attempt
                WHERE attempt.run_attempt_id = eligible.run_attempt_id
            ),
            eligible.completed_on,
            eligible.started_on,
            eligible.created_on
        ) < NOW() - MAKE_INTERVAL(secs => p_queued_too_long_seconds)
      AND NOT EXISTS (
            SELECT 1
            FROM stale_page AS prior
            WHERE prior.job_id = eligible.job_id
        )
    ORDER BY eligible.created_on DESC, eligible.job_id
    LIMIT GREATEST(
        (SELECT page_size FROM page_parameters)
            - (SELECT COUNT(*) FROM terminal_mismatch_page)
            - (SELECT COUNT(*) FROM failed_page)
            - (SELECT COUNT(*) FROM stale_page)
            - (SELECT COUNT(*) FROM running_page),
        0
    )
), retry_page AS MATERIALIZED (
    SELECT
        eligible.job_id,
        eligible.created_on,
        5 AS attention_rank
    FROM eligible_jobs AS eligible
    WHERE p_attention = 'any'
      AND p_sort = 'attention'
      AND eligible.job_state = 'retry'
      AND NOT EXISTS (
            SELECT 1
            FROM stale_page AS prior
            WHERE prior.job_id = eligible.job_id
        )
      AND NOT EXISTS (
            SELECT 1
            FROM queued_page AS prior
            WHERE prior.job_id = eligible.job_id
        )
    ORDER BY eligible.created_on DESC, eligible.job_id
    LIMIT GREATEST(
        (SELECT page_size FROM page_parameters)
            - (SELECT COUNT(*) FROM terminal_mismatch_page)
            - (SELECT COUNT(*) FROM failed_page)
            - (SELECT COUNT(*) FROM stale_page)
            - (SELECT COUNT(*) FROM running_page)
            - (SELECT COUNT(*) FROM queued_page),
        0
    )
), fallback_page AS MATERIALIZED (
    SELECT
        eligible.job_id,
        eligible.created_on,
        6 AS attention_rank
    FROM eligible_jobs AS eligible
    WHERE p_attention = 'any'
      AND p_sort = 'attention'
      AND NOT EXISTS (
            SELECT 1
            FROM terminal_mismatch_page AS prior
            WHERE prior.job_id = eligible.job_id
        )
      AND NOT EXISTS (
            SELECT 1
            FROM failed_page AS prior
            WHERE prior.job_id = eligible.job_id
        )
      AND NOT EXISTS (
            SELECT 1
            FROM stale_page AS prior
            WHERE prior.job_id = eligible.job_id
        )
      AND NOT EXISTS (
            SELECT 1
            FROM running_page AS prior
            WHERE prior.job_id = eligible.job_id
        )
      AND NOT EXISTS (
            SELECT 1
            FROM queued_page AS prior
            WHERE prior.job_id = eligible.job_id
        )
      AND NOT EXISTS (
            SELECT 1
            FROM retry_page AS prior
            WHERE prior.job_id = eligible.job_id
        )
    ORDER BY eligible.created_on DESC, eligible.job_id
    LIMIT GREATEST(
        (SELECT page_size FROM page_parameters)
            - (SELECT COUNT(*) FROM terminal_mismatch_page)
            - (SELECT COUNT(*) FROM failed_page)
            - (SELECT COUNT(*) FROM stale_page)
            - (SELECT COUNT(*) FROM running_page)
            - (SELECT COUNT(*) FROM queued_page)
            - (SELECT COUNT(*) FROM retry_page),
        0
    )
), priority_candidates AS (
    SELECT * FROM terminal_mismatch_page
    UNION ALL
    SELECT * FROM failed_page
    UNION ALL
    SELECT * FROM stale_page
    UNION ALL
    SELECT * FROM running_page
    UNION ALL
    SELECT * FROM queued_page
    UNION ALL
    SELECT * FROM retry_page
    UNION ALL
    SELECT * FROM fallback_page
), priority_page_slice AS (
    SELECT job_id, attention_rank, created_on
    FROM priority_candidates
    ORDER BY attention_rank, created_on DESC, job_id
    LIMIT p_limit OFFSET p_offset
), priority_page AS (
    SELECT
        job_id,
        ROW_NUMBER() OVER (
            ORDER BY attention_rank, created_on DESC, job_id
        ) AS ordinal
    FROM priority_page_slice
), generic_candidates AS NOT MATERIALIZED (
    SELECT
        eligible.*,
        COALESCE(
            attempt.updated_on,
            eligible.completed_on,
            eligible.started_on,
            eligible.created_on
        ) AS last_updated_on,
        eligible.job_state IN ('created', 'retry')
            AND COALESCE(
                attempt.updated_on,
                eligible.completed_on,
                eligible.started_on,
                eligible.created_on
            ) < NOW() - MAKE_INTERVAL(secs => p_queued_too_long_seconds)
            AS queued_too_long,
        eligible.job_state = 'active'
            AND eligible.started_on IS NOT NULL
            AND eligible.started_on
                < NOW() - MAKE_INTERVAL(secs => p_running_too_long_seconds)
            AS running_too_long,
        eligible.job_state IN ('active', 'retry')
            AND COALESCE(
                attempt.updated_on,
                eligible.completed_on,
                eligible.started_on,
                eligible.created_on
            ) < NOW() - MAKE_INTERVAL(secs => p_stale_update_seconds)
            AS stale_update,
        eligible.job_state = 'retry' AS retrying,
        eligible.job_state IN ('failed', 'expired', 'cancelled')
            AS failed_attention,
        eligible.run_attempt_id IS NOT NULL
            AND eligible.job_state IN (
                'completed',
                'skipped',
                'failed',
                'expired',
                'cancelled'
            )
            AND (
                attempt.terminal_accepted IS FALSE
                OR (
                    attempt.terminal_work_state IS NOT NULL
                    AND attempt.terminal_work_state
                        <> eligible.job_state::TEXT
                )
            ) AS terminal_mismatch
    FROM eligible_jobs AS eligible
    LEFT JOIN marie_scheduler.job_attempt AS attempt
      ON attempt.run_attempt_id = eligible.run_attempt_id
    WHERE NOT (p_attention = 'any' AND p_sort = 'attention')
), generic_ranked AS NOT MATERIALIZED (
    SELECT
        generic.*,
        CASE
            WHEN terminal_mismatch THEN 0
            WHEN failed_attention THEN 1
            WHEN stale_update THEN 2
            WHEN running_too_long THEN 3
            WHEN queued_too_long THEN 4
            WHEN retrying THEN 5
            ELSE 6
        END AS attention_rank
    FROM generic_candidates AS generic
    WHERE CASE p_attention
        WHEN 'any' THEN TRUE
        WHEN 'queued_too_long' THEN queued_too_long
        WHEN 'running_too_long' THEN running_too_long
        WHEN 'stale_update' THEN stale_update
        WHEN 'retrying' THEN retrying
        WHEN 'failed' THEN failed_attention
        WHEN 'terminal_mismatch' THEN terminal_mismatch
        ELSE FALSE
    END
), generic_page_slice AS (
    SELECT *
    FROM generic_ranked
    ORDER BY
        CASE WHEN p_sort = 'timeline' THEN job_level END DESC,
        CASE WHEN p_sort = 'timeline' THEN
            CASE
                WHEN started_on IS NULL AND completed_on IS NULL THEN 1
                ELSE 0
            END
        END,
        CASE WHEN p_sort = 'timeline' THEN
            COALESCE(started_on, completed_on, created_on)
        END,
        CASE WHEN p_sort = 'newest' THEN created_on END DESC,
        CASE WHEN p_sort = 'oldest' THEN created_on END,
        CASE WHEN p_sort = 'updated' THEN last_updated_on END DESC,
        CASE WHEN p_sort = 'attention' THEN attention_rank END,
        CASE WHEN p_sort = 'attention' THEN created_on END DESC,
        job_id
    LIMIT p_limit OFFSET p_offset
), generic_page AS (
    SELECT
        job_id,
        ROW_NUMBER() OVER (
            ORDER BY
                CASE WHEN p_sort = 'timeline' THEN job_level END DESC,
                CASE WHEN p_sort = 'timeline' THEN
                    CASE
                        WHEN started_on IS NULL AND completed_on IS NULL
                            THEN 1
                        ELSE 0
                    END
                END,
                CASE WHEN p_sort = 'timeline' THEN
                    COALESCE(started_on, completed_on, created_on)
                END,
                CASE WHEN p_sort = 'newest' THEN created_on END DESC,
                CASE WHEN p_sort = 'oldest' THEN created_on END,
                CASE WHEN p_sort = 'updated' THEN last_updated_on END DESC,
                CASE WHEN p_sort = 'attention' THEN attention_rank END,
                CASE WHEN p_sort = 'attention' THEN created_on END DESC,
                job_id
        ) AS ordinal
    FROM generic_page_slice
), selected_jobs AS (
    SELECT job_id, ordinal FROM priority_page
    UNION ALL
    SELECT job_id, ordinal FROM generic_page
), paged AS (
    SELECT
        selected.ordinal,
        job.id AS job_id,
        job.name AS queue_name,
        job.state::TEXT AS job_state,
        job.dag_id,
        dag.name::TEXT AS dag_name,
        dag.planner::TEXT AS planner,
        job.priority,
        job.job_level,
        job.retry_count,
        job.retry_limit,
        job.created_on,
        job.started_on,
        job.completed_on,
        COALESCE(
            attempt.updated_on,
            job.completed_on,
            job.started_on,
            job.created_on
        ) AS last_updated_on,
        EXTRACT(EPOCH FROM (NOW() - job.created_on))::DOUBLE PRECISION
            AS age_seconds,
        EXTRACT(EPOCH FROM (
            NOW() - COALESCE(
                attempt.updated_on,
                job.completed_on,
                job.started_on,
                job.created_on
            )
        ))::DOUBLE PRECISION AS last_update_age_seconds,
        job.run_owner,
        job.run_attempt_id,
        attempt.executor,
        attempt.activated_at AS attempt_activated_at,
        attempt.terminal_at AS attempt_terminal_at,
        attempt.terminal_status,
        attempt.terminal_work_state,
        attempt.terminal_source,
        attempt.terminal_accepted
    FROM selected_jobs AS selected
    JOIN marie_scheduler.job AS job ON job.id = selected.job_id
    LEFT JOIN marie_scheduler.dag AS dag ON dag.id = job.dag_id
    LEFT JOIN marie_scheduler.job_attempt AS attempt
      ON attempt.run_attempt_id = job.run_attempt_id
)
SELECT
    page_metadata.total_count,
    facets.queue_facets,
    paged.job_id,
    paged.queue_name,
    paged.job_state,
    paged.dag_id,
    paged.dag_name,
    paged.planner,
    paged.priority,
    paged.job_level,
    paged.retry_count,
    paged.retry_limit,
    paged.created_on,
    paged.started_on,
    paged.completed_on,
    paged.last_updated_on,
    paged.age_seconds,
    paged.last_update_age_seconds,
    paged.run_owner,
    paged.run_attempt_id,
    paged.executor,
    paged.attempt_activated_at,
    paged.attempt_terminal_at,
    paged.terminal_status,
    paged.terminal_work_state,
    paged.terminal_source,
    paged.terminal_accepted
FROM page_metadata
CROSS JOIN facets
LEFT JOIN paged ON TRUE
ORDER BY paged.ordinal;
$function$;

COMMENT ON FUNCTION marie_scheduler.list_operational_jobs(
    INTEGER,
    INTEGER,
    TEXT[],
    TEXT,
    TEXT,
    TEXT,
    TEXT,
    UUID,
    INTEGER,
    INTEGER,
    INTEGER
)
IS 'Returns one bounded payload-free operational job page with total count and queue facets.';

-- File: 076_operational_observability_indexes.sql
-- Description: Bounded operational history and flow query indexes

CREATE INDEX IF NOT EXISTS job_history_operational_event_idx
    ON marie_scheduler.job_history (history_created_on DESC, history_id DESC);

CREATE INDEX IF NOT EXISTS dag_history_operational_event_idx
    ON marie_scheduler.dag_history (history_created_on DESC, history_id DESC);

CREATE INDEX IF NOT EXISTS job_attempt_operational_updated_idx
    ON marie_scheduler.job_attempt (updated_on DESC, run_attempt_id DESC);

CREATE INDEX IF NOT EXISTS job_attempt_operational_activated_idx
    ON marie_scheduler.job_attempt (activated_at DESC, run_attempt_id DESC);

CREATE INDEX IF NOT EXISTS job_u_operational_created_idx
    ON marie_scheduler.job (created_on DESC);

CREATE INDEX IF NOT EXISTS job_u_operational_started_idx
    ON marie_scheduler.job (started_on DESC)
    WHERE started_on IS NOT NULL;

CREATE INDEX IF NOT EXISTS job_u_operational_completed_idx
    ON marie_scheduler.job (completed_on DESC)
    WHERE completed_on IS NOT NULL;

CREATE INDEX IF NOT EXISTS job_u_operational_terminal_attempt_idx
    ON marie_scheduler.job (created_on DESC, id)
    INCLUDE (state, run_attempt_id)
    WHERE run_attempt_id IS NOT NULL
      AND state IN ('completed', 'skipped', 'failed', 'expired', 'cancelled');

-- File: 086_job_attempt_terminal_lookup_index.sql
-- Description: Cover terminal-attempt checks in operational job pages

CREATE INDEX IF NOT EXISTS job_attempt_terminal_lookup_idx
    ON marie_scheduler.job_attempt (run_attempt_id)
    INCLUDE (terminal_accepted, terminal_work_state)
    WHERE terminal_accepted IS FALSE
       OR terminal_work_state IS NOT NULL;

CREATE OR REPLACE FUNCTION marie_scheduler.hydrate_frontier_jobs(dag_ids uuid[])
RETURNS TABLE (
  dag_id uuid,
  job json
)
LANGUAGE plpgsql
STABLE
SET jit = off
SET work_mem = '16MB'
SET plan_cache_mode = force_custom_plan
AS $$
BEGIN
  RETURN QUERY
  SELECT
    j.dag_id,
    json_build_object(
      'id',                j.id,
      'name',              j.name,
      'priority',          j.priority,
      'state',             j.state,
      'retry_limit',       j.retry_limit,
      'start_after',       j.start_after,
      'expire_in_seconds', EXTRACT(EPOCH FROM j.expire_in)::integer,
      'data',              j.data,
      'retry_delay',       j.retry_delay,
      'retry_backoff',     j.retry_backoff,
      'keep_until',        j.keep_until,
      'job_level',         j.job_level,
      'soft_sla',          j.soft_sla,
      'hard_sla',          j.hard_sla,
      'dependencies',      COALESCE(dep.deps, '[]'::json)
    ) AS job
  FROM (
    SELECT DISTINCT requested.dag_id
    FROM unnest(COALESCE(dag_ids, ARRAY[]::uuid[])) AS requested(dag_id)
  ) requested
  JOIN marie_scheduler.job j
    ON j.dag_id = requested.dag_id
  LEFT JOIN LATERAL (
    SELECT json_agg(jd.depends_on_id) FILTER (
             WHERE p.id IS NOT NULL
               AND p.state NOT IN ('completed','failed','cancelled','skipped')
           ) AS deps
    FROM marie_scheduler.job_dependencies jd
    LEFT JOIN marie_scheduler.job p
           ON p.name = jd.depends_on_name
          AND p.id = jd.depends_on_id
    WHERE jd.job_name = j.name
      AND jd.job_id = j.id
  ) dep ON TRUE
  WHERE j.state IN ('created','retry')
  ORDER BY j.dag_id, j.job_level, j.created_on;
END;
$$;

-- Current definition. Earlier definitions remain immutable deployment history.
CREATE OR REPLACE FUNCTION marie_scheduler.admission_candidate_dags(
    p_limit integer,
    p_sla_interval_seconds integer,
    p_excluded_dag_ids uuid[] DEFAULT ARRAY[]::uuid[]
)
RETURNS TABLE
        (
            dag_id         uuid,
            serialized_dag jsonb
        )
LANGUAGE plpgsql
STABLE
SET jit = off
SET plan_cache_mode = force_custom_plan
AS
$$
BEGIN
    RETURN QUERY
    SELECT
        d.id,
        d.serialized_dag
    FROM marie_scheduler.dag d
    WHERE d.state IN ('created', 'active')
      AND NOT EXISTS (
          SELECT 1
          FROM unnest(
              COALESCE(p_excluded_dag_ids, ARRAY[]::uuid[])
          ) AS excluded(dag_id)
          WHERE excluded.dag_id = d.id
      )
      AND EXISTS (
          SELECT 1
          FROM marie_scheduler.job ready
          WHERE ready.dag_id = d.id
            AND ready.state IN ('created', 'retry')
            AND ready.start_after <= CURRENT_TIMESTAMP
      )
      AND NOT EXISTS (
          SELECT 1
          FROM marie_scheduler.job blocker
          WHERE blocker.dag_id = d.id
            AND blocker.state IN ('failed', 'expired', 'cancelled')
      )
    ORDER BY
        d.priority DESC,
        COALESCE(d.soft_sla, d.hard_sla) ASC NULLS LAST,
        d.created_on,
        d.id
    LIMIT GREATEST(0, p_limit);
END;
$$;

COMMENT ON FUNCTION marie_scheduler.admission_candidate_dags(integer, integer, uuid[])
IS 'Selects durable DAG admission candidates by operator priority, earliest SLA, then FIFO.';

ANALYZE marie_scheduler.job;
ANALYZE marie_scheduler.job_attempt;
ANALYZE marie_scheduler.job_dependencies;
ANALYZE marie_scheduler.dag;

DO $verify$
DECLARE
    operational_parallel "char";
    operational_definition text;
    queue_definition text;
    missing_indexes text;
BEGIN
    SELECT procedure.proparallel, pg_get_functiondef(procedure.oid)
    INTO operational_parallel, operational_definition
    FROM pg_proc AS procedure
    WHERE procedure.oid = (
        'marie_scheduler.list_operational_jobs('
        'integer,integer,text[],text,text,text,text,uuid,integer,integer,integer'
        ')'::regprocedure
    );

    IF operational_parallel IS DISTINCT FROM 's'
       OR position(
            'terminal_mismatch_page AS MATERIALIZED'
            IN operational_definition
       ) = 0 THEN
        RAISE EXCEPTION
            'list_operational_jobs verification failed: parallel=%, optimized=%',
            operational_parallel,
            position(
                'terminal_mismatch_page AS MATERIALIZED'
                IN operational_definition
            ) > 0;
    END IF;

    SELECT pg_get_functiondef(procedure.oid)
    INTO queue_definition
    FROM pg_proc AS procedure
    WHERE procedure.oid =
        'marie_scheduler.create_queue(text,json)'::regprocedure;

    IF position('ATTACH PARTITION' IN queue_definition) > 0 THEN
        RAISE EXCEPTION
            'create_queue still contains partition DDL';
    END IF;

    SELECT string_agg(required.index_name, ', ' ORDER BY required.index_name)
    INTO missing_indexes
    FROM unnest(ARRAY[
        'job_u_hydrate_frontier_idx',
        'job_u_admission_ready_idx',
        'job_u_admission_blocker_idx',
        'job_u_queue_ready_order_idx',
        'job_u_dag_state_idx',
        'job_u_expired_acquisition_lease_idx',
        'job_u_expired_run_lease_idx',
        'job_attempt_terminal_lookup_idx',
        'job_u_operational_created_idx',
        'job_u_operational_started_idx',
        'job_u_operational_completed_idx',
        'job_u_operational_terminal_attempt_idx'
    ]) AS required(index_name)
    WHERE to_regclass(
        'marie_scheduler.' || required.index_name
    ) IS NULL;

    IF missing_indexes IS NOT NULL THEN
        RAISE EXCEPTION
            'required unpartitioned indexes are missing: %',
            missing_indexes;
    END IF;
END
$verify$;

COMMIT;

SELECT
    relation.relname AS active_relation,
    relation.relkind AS active_relkind,
    (
        SELECT count(*)
        FROM pg_inherits
        WHERE inhparent = relation.oid
    ) AS active_partitions,
    pg_size_pretty(pg_total_relation_size(relation.oid)) AS active_total_size
FROM pg_class AS relation
WHERE relation.oid = 'marie_scheduler.job'::regclass;

SELECT
    procedure.proparallel,
    position(
        'terminal_mismatch_page AS MATERIALIZED'
        IN pg_get_functiondef(procedure.oid)
    ) > 0 AS optimized_definition
FROM pg_proc AS procedure
WHERE procedure.oid = (
    'marie_scheduler.list_operational_jobs('
    'integer,integer,text[],text,text,text,text,uuid,integer,integer,integer'
    ')'::regprocedure
);
