-- File: 077_list_operational_attempts.sql
-- Description: Payload-free operational attempt page
-- Dependencies: 065_job_attempt.sql

CREATE OR REPLACE FUNCTION {schema}.list_operational_attempts(
    p_limit INTEGER DEFAULT 25,
    p_offset INTEGER DEFAULT 0,
    p_states TEXT[] DEFAULT NULL,
    p_attention TEXT DEFAULT 'any',
    p_gateway TEXT DEFAULT NULL,
    p_executor TEXT DEFAULT NULL,
    p_search TEXT DEFAULT NULL,
    p_sort TEXT DEFAULT 'attention',
    p_active_too_long_seconds INTEGER DEFAULT 900,
    p_stale_update_seconds INTEGER DEFAULT 600
)
RETURNS TABLE (
    total_count BIGINT,
    gateway_facets TEXT[],
    executor_facets TEXT[],
    run_attempt_id UUID,
    job_id UUID,
    queue_name TEXT,
    dag_id UUID,
    run_owner TEXT,
    scheduler_lease_owner TEXT,
    gateway_instance_id TEXT,
    executor TEXT,
    attempt_state TEXT,
    activated_at TIMESTAMPTZ,
    terminal_at TIMESTAMPTZ,
    terminal_status TEXT,
    terminal_work_state TEXT,
    terminal_source TEXT,
    terminal_gateway_instance_id TEXT,
    terminal_scheduler_lease_owner TEXT,
    terminal_accepted BOOLEAN,
    recovery_at TIMESTAMPTZ,
    recovery_state TEXT,
    created_on TIMESTAMPTZ,
    updated_on TIMESTAMPTZ,
    age_seconds DOUBLE PRECISION,
    last_update_age_seconds DOUBLE PRECISION,
    attention_codes TEXT[]
)
LANGUAGE SQL
STABLE
AS $function$
WITH source AS NOT MATERIALIZED (
    SELECT
        ja.*,
        EXTRACT(EPOCH FROM (NOW() - ja.activated_at))::DOUBLE PRECISION AS age_seconds,
        EXTRACT(EPOCH FROM (NOW() - ja.updated_on))::DOUBLE PRECISION AS last_update_age_seconds,
        ja.terminal_at IS NULL AND ja.recovery_at IS NULL AS is_active,
        ja.terminal_at IS NOT NULL AND (
            ja.terminal_accepted IS FALSE
            OR (
                ja.terminal_status IS NOT NULL
                AND ja.terminal_work_state IS NOT NULL
                AND LOWER(ja.terminal_status) <> LOWER(ja.terminal_work_state)
            )
        ) AS terminal_mismatch,
        ja.terminal_at IS NOT NULL AND (
            (
                ja.terminal_gateway_instance_id IS NOT NULL
                AND ja.gateway_instance_id IS NOT NULL
                AND ja.terminal_gateway_instance_id <> ja.gateway_instance_id
            )
            OR (
                ja.terminal_scheduler_lease_owner IS NOT NULL
                AND ja.terminal_scheduler_lease_owner <> ja.scheduler_lease_owner
            )
        ) AS owner_mismatch
    FROM {schema}.job_attempt AS ja
    WHERE (p_states IS NULL OR LOWER(ja.attempt_state) = ANY(p_states))
      AND (p_gateway IS NULL OR ja.gateway_instance_id = p_gateway)
      AND (p_executor IS NULL OR ja.executor = p_executor)
      AND (
          p_search IS NULL
          OR ja.run_attempt_id::TEXT ILIKE '%' || p_search || '%'
          OR ja.job_id::TEXT ILIKE '%' || p_search || '%'
          OR ja.dag_id::TEXT ILIKE '%' || p_search || '%'
          OR ja.job_name ILIKE '%' || p_search || '%'
          OR ja.run_owner ILIKE '%' || p_search || '%'
          OR COALESCE(ja.gateway_instance_id, '') ILIKE '%' || p_search || '%'
          OR COALESCE(ja.executor, '') ILIKE '%' || p_search || '%'
      )
), filtered AS NOT MATERIALIZED (
    SELECT
        source.*,
        ARRAY_REMOVE(ARRAY[
            CASE WHEN terminal_accepted IS FALSE THEN 'TERMINAL_REJECTED' END,
            CASE WHEN terminal_mismatch THEN 'TERMINAL_MISMATCH' END,
            CASE WHEN owner_mismatch THEN 'OWNER_MISMATCH' END,
            CASE WHEN recovery_at IS NOT NULL THEN 'RECOVERED' END,
            CASE
                WHEN is_active AND age_seconds > p_active_too_long_seconds
                THEN 'ACTIVE_TOO_LONG'
            END,
            CASE
                WHEN is_active AND last_update_age_seconds > p_stale_update_seconds
                THEN 'STALE_UPDATE'
            END
        ]::TEXT[], NULL) AS attention_codes,
        CASE
            WHEN terminal_accepted IS FALSE OR terminal_mismatch THEN 0
            WHEN owner_mismatch THEN 1
            WHEN recovery_at IS NOT NULL THEN 2
            WHEN is_active AND age_seconds > p_active_too_long_seconds THEN 3
            WHEN is_active AND last_update_age_seconds > p_stale_update_seconds THEN 4
            ELSE 5
        END AS attention_rank
    FROM source
    WHERE p_attention = 'any'
       OR (p_attention = 'active_too_long' AND is_active AND age_seconds > p_active_too_long_seconds)
       OR (p_attention = 'stale_update' AND is_active AND last_update_age_seconds > p_stale_update_seconds)
       OR (p_attention = 'recovered' AND recovery_at IS NOT NULL)
       OR (p_attention = 'terminal_rejected' AND terminal_accepted IS FALSE)
       OR (p_attention = 'terminal_mismatch' AND terminal_mismatch)
       OR (p_attention = 'owner_mismatch' AND owner_mismatch)
), metadata AS (
    SELECT
        COUNT(*) AS total_count,
        COALESCE(
            ARRAY_AGG(DISTINCT gateway_instance_id ORDER BY gateway_instance_id)
                FILTER (WHERE gateway_instance_id IS NOT NULL),
            ARRAY[]::TEXT[]
        ) AS gateway_facets,
        COALESCE(
            ARRAY_AGG(DISTINCT executor ORDER BY executor)
                FILTER (WHERE executor IS NOT NULL),
            ARRAY[]::TEXT[]
        ) AS executor_facets
    FROM filtered
), paged AS (
    SELECT *
    FROM filtered
    ORDER BY
        CASE WHEN p_sort = 'attention' THEN attention_rank END,
        CASE WHEN p_sort = 'attention' THEN updated_on END DESC,
        CASE WHEN p_sort = 'newest' THEN activated_at END DESC,
        CASE WHEN p_sort = 'oldest' THEN activated_at END,
        CASE WHEN p_sort = 'updated' THEN updated_on END DESC,
        run_attempt_id DESC
    LIMIT LEAST(GREATEST(p_limit, 1), 100)
    OFFSET GREATEST(p_offset, 0)
)
SELECT
    metadata.total_count,
    metadata.gateway_facets,
    metadata.executor_facets,
    paged.run_attempt_id,
    paged.job_id,
    paged.job_name,
    paged.dag_id,
    paged.run_owner,
    paged.scheduler_lease_owner,
    paged.gateway_instance_id,
    paged.executor,
    paged.attempt_state,
    paged.activated_at,
    paged.terminal_at,
    paged.terminal_status,
    paged.terminal_work_state,
    paged.terminal_source,
    paged.terminal_gateway_instance_id,
    paged.terminal_scheduler_lease_owner,
    paged.terminal_accepted,
    paged.recovery_at,
    paged.recovery_state,
    paged.created_on,
    paged.updated_on,
    paged.age_seconds,
    paged.last_update_age_seconds,
    paged.attention_codes
FROM metadata
LEFT JOIN paged ON TRUE
ORDER BY
    CASE WHEN p_sort = 'attention' THEN paged.attention_rank END,
    CASE WHEN p_sort = 'attention' THEN paged.updated_on END DESC,
    CASE WHEN p_sort = 'newest' THEN paged.activated_at END DESC,
    CASE WHEN p_sort = 'oldest' THEN paged.activated_at END,
    CASE WHEN p_sort = 'updated' THEN paged.updated_on END DESC,
    paged.run_attempt_id DESC;
$function$;

COMMENT ON FUNCTION {schema}.list_operational_attempts(
    INTEGER, INTEGER, TEXT[], TEXT, TEXT, TEXT, TEXT, TEXT, INTEGER, INTEGER
)
IS 'Returns a bounded, payload-free job-attempt audit page with safe attention signals.';
