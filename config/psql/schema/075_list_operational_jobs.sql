-- File: 075_list_operational_jobs.sql
-- Description: Payload-free operational job page
-- Dependencies: 004_queue.sql, 005_job.sql, 007_dag.sql, 065_job_attempt.sql

CREATE OR REPLACE FUNCTION {schema}.list_operational_jobs(
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
AS $function$
WITH source AS (
    SELECT
        j.id AS job_id,
        j.name AS queue_name,
        j.state::TEXT AS job_state,
        j.dag_id,
        d.name::TEXT AS dag_name,
        d.planner::TEXT AS planner,
        j.priority,
        j.job_level,
        j.retry_count,
        j.retry_limit,
        j.created_on,
        j.started_on,
        j.completed_on,
        COALESCE(
            ja.updated_on,
            j.completed_on,
            j.started_on,
            j.created_on
        ) AS last_updated_on,
        EXTRACT(EPOCH FROM (NOW() - j.created_on))::DOUBLE PRECISION
            AS age_seconds,
        EXTRACT(EPOCH FROM (
            NOW() - COALESCE(
                ja.updated_on,
                j.completed_on,
                j.started_on,
                j.created_on
            )
        ))::DOUBLE PRECISION AS last_update_age_seconds,
        j.run_owner,
        j.run_attempt_id,
        ja.executor,
        ja.activated_at AS attempt_activated_at,
        ja.terminal_at AS attempt_terminal_at,
        ja.terminal_status,
        ja.terminal_work_state,
        ja.terminal_source,
        ja.terminal_accepted,
        j.state::TEXT IN ('created', 'retry')
            AND EXTRACT(EPOCH FROM (
                NOW() - COALESCE(
                    ja.updated_on,
                    j.completed_on,
                    j.started_on,
                    j.created_on
                )
            )) > p_queued_too_long_seconds AS queued_too_long,
        j.state::TEXT = 'active'
            AND j.started_on IS NOT NULL
            AND EXTRACT(EPOCH FROM (NOW() - j.started_on))
                > p_running_too_long_seconds AS running_too_long,
        j.state::TEXT IN ('active', 'retry')
            AND EXTRACT(EPOCH FROM (
                NOW() - COALESCE(
                    ja.updated_on,
                    j.completed_on,
                    j.started_on,
                    j.created_on
                )
            )) > p_stale_update_seconds AS stale_update,
        j.state::TEXT = 'retry' AS retrying,
        j.state::TEXT IN ('failed', 'expired', 'cancelled') AS failed_attention,
        j.run_attempt_id IS NOT NULL
            AND j.state::TEXT IN (
                'completed',
                'skipped',
                'failed',
                'expired',
                'cancelled'
            )
            AND (
                ja.terminal_accepted IS FALSE
                OR (
                    ja.terminal_work_state IS NOT NULL
                    AND ja.terminal_work_state <> j.state::TEXT
                )
            ) AS terminal_mismatch
    FROM {schema}.job AS j
    LEFT JOIN {schema}.dag AS d ON d.id = j.dag_id
    LEFT JOIN {schema}.job_attempt AS ja
      ON ja.run_attempt_id = j.run_attempt_id
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
            OR COALESCE(d.name, '') ILIKE '%' || p_search || '%'
            OR COALESCE(d.planner, '') ILIKE '%' || p_search || '%'
            OR COALESCE(ja.executor, '') ILIKE '%' || p_search || '%'
            OR COALESCE(j.run_owner, '') ILIKE '%' || p_search || '%'
        )
), filtered AS NOT MATERIALIZED (
    SELECT
        source.*,
        CASE
            WHEN terminal_mismatch THEN 0
            WHEN failed_attention THEN 1
            WHEN stale_update THEN 2
            WHEN running_too_long THEN 3
            WHEN queued_too_long THEN 4
            WHEN retrying THEN 5
            ELSE 6
        END AS attention_rank
    FROM source
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
), page_metadata AS (
    SELECT COUNT(*) AS total_count
    FROM filtered
), facets AS (
    SELECT CASE
        WHEN p_dag_id IS NULL THEN COALESCE(
            (SELECT ARRAY_AGG(name ORDER BY name) FROM {schema}.queue),
            ARRAY[]::TEXT[]
        )
        ELSE ARRAY[]::TEXT[]
    END AS queue_facets
), paged AS (
    SELECT *
    FROM filtered
    ORDER BY
        CASE WHEN p_sort = 'timeline' THEN job_level END DESC,
        CASE WHEN p_sort = 'timeline' THEN
            CASE WHEN started_on IS NULL AND completed_on IS NULL THEN 1 ELSE 0 END
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
ORDER BY
    CASE WHEN p_sort = 'timeline' THEN paged.job_level END DESC,
    CASE WHEN p_sort = 'timeline' THEN
        CASE
            WHEN paged.started_on IS NULL AND paged.completed_on IS NULL THEN 1
            ELSE 0
        END
    END,
    CASE WHEN p_sort = 'timeline' THEN
        COALESCE(paged.started_on, paged.completed_on, paged.created_on)
    END,
    CASE WHEN p_sort = 'newest' THEN paged.created_on END DESC,
    CASE WHEN p_sort = 'oldest' THEN paged.created_on END,
    CASE WHEN p_sort = 'updated' THEN paged.last_updated_on END DESC,
    CASE WHEN p_sort = 'attention' THEN paged.attention_rank END,
    CASE WHEN p_sort = 'attention' THEN paged.created_on END DESC,
    paged.job_id;
$function$;

COMMENT ON FUNCTION {schema}.list_operational_jobs(
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
