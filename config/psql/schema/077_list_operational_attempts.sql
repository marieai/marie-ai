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
LANGUAGE plpgsql
STABLE
PARALLEL SAFE
AS $function$
BEGIN
    -- Optional filters need a fresh plan; retain index scans for selective pages.
    RETURN QUERY EXECUTE $query$
WITH signals AS NOT MATERIALIZED (
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
    WHERE ($3 IS NULL OR LOWER(ja.attempt_state) = ANY($3))
      AND ($5 IS NULL OR ja.gateway_instance_id = $5)
      AND ($6 IS NULL OR ja.executor = $6)
      AND (
          $7 IS NULL
          OR ja.run_attempt_id::TEXT ILIKE '%' || $7 || '%'
          OR ja.job_id::TEXT ILIKE '%' || $7 || '%'
          OR ja.dag_id::TEXT ILIKE '%' || $7 || '%'
          OR ja.job_name ILIKE '%' || $7 || '%'
          OR ja.run_owner ILIKE '%' || $7 || '%'
          OR COALESCE(ja.gateway_instance_id, '') ILIKE '%' || $7 || '%'
          OR COALESCE(ja.executor, '') ILIKE '%' || $7 || '%'
      )
), source AS NOT MATERIALIZED (
    SELECT
        signals.*,
        CASE
            WHEN terminal_accepted IS FALSE OR terminal_mismatch THEN 0
            WHEN owner_mismatch THEN 1
            WHEN recovery_at IS NOT NULL THEN 2
            WHEN is_active THEN 3
            ELSE 4
        END AS attention_group
    FROM signals
), filtered AS NOT MATERIALIZED (
    SELECT
        source.*,
        ARRAY_REMOVE(ARRAY[
            CASE WHEN terminal_accepted IS FALSE THEN 'TERMINAL_REJECTED' END,
            CASE WHEN terminal_mismatch THEN 'TERMINAL_MISMATCH' END,
            CASE WHEN owner_mismatch THEN 'OWNER_MISMATCH' END,
            CASE WHEN recovery_at IS NOT NULL THEN 'RECOVERED' END,
            CASE
                WHEN is_active AND age_seconds > $9
                THEN 'ACTIVE_TOO_LONG'
            END,
            CASE
                WHEN is_active AND last_update_age_seconds > $10
                THEN 'STALE_UPDATE'
            END
        ]::TEXT[], NULL) AS attention_codes,
        CASE
            WHEN terminal_accepted IS FALSE OR terminal_mismatch THEN 0
            WHEN owner_mismatch THEN 1
            WHEN recovery_at IS NOT NULL THEN 2
            WHEN is_active AND age_seconds > $9 THEN 3
            WHEN is_active AND last_update_age_seconds > $10 THEN 4
            ELSE 5
        END AS attention_rank
    FROM source
    WHERE $4 = 'any'
       OR ($4 = 'active_too_long' AND is_active AND age_seconds > $9)
       OR ($4 = 'stale_update' AND is_active AND last_update_age_seconds > $10)
       OR ($4 = 'recovered' AND recovery_at IS NOT NULL)
       OR ($4 = 'terminal_rejected' AND terminal_accepted IS FALSE)
       OR ($4 = 'terminal_mismatch' AND terminal_mismatch)
       OR ($4 = 'owner_mismatch' AND owner_mismatch)
), metadata_pairs AS (
    SELECT
        gateway_instance_id,
        executor,
        COUNT(*) AS attempt_count
    FROM filtered
    GROUP BY gateway_instance_id, executor
), metadata AS (
    SELECT
        COALESCE(SUM(attempt_count), 0)::BIGINT AS total_count,
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
    FROM metadata_pairs
), page_budget AS (
    -- Lower-priority scans stop once the offset and page have been filled.
    SELECT CASE
        WHEN $8 = 'attention'
         AND GREATEST($2, 0) < (SELECT total_count FROM metadata)
        THEN GREATEST($2, 0)::BIGINT + LEAST(GREATEST($1, 1), 100)
        ELSE 0::BIGINT
    END AS candidate_limit
), rejected_page AS MATERIALIZED (
    SELECT run_attempt_id, updated_on, activated_at, 0 AS attention_rank
    FROM filtered
    WHERE attention_group = 0
    ORDER BY updated_on DESC, run_attempt_id DESC
    LIMIT (SELECT candidate_limit FROM page_budget)
), owner_page AS MATERIALIZED (
    SELECT run_attempt_id, updated_on, activated_at, 1 AS attention_rank
    FROM filtered
    WHERE attention_group = 1
    ORDER BY updated_on DESC, run_attempt_id DESC
    LIMIT (SELECT candidate_limit - (SELECT COUNT(*) FROM rejected_page) FROM page_budget)
), recovered_page AS MATERIALIZED (
    SELECT run_attempt_id, updated_on, activated_at, 2 AS attention_rank
    FROM filtered
    WHERE attention_group = 2
    ORDER BY updated_on DESC, run_attempt_id DESC
    LIMIT (
        SELECT candidate_limit
            - (SELECT COUNT(*) FROM rejected_page)
            - (SELECT COUNT(*) FROM owner_page)
        FROM page_budget
    )
), long_running_page AS MATERIALIZED (
    SELECT run_attempt_id, updated_on, activated_at, 3 AS attention_rank
    FROM filtered
    WHERE attention_group = 3
      AND age_seconds > $9
    ORDER BY updated_on DESC, run_attempt_id DESC
    LIMIT (
        SELECT candidate_limit
            - (SELECT COUNT(*) FROM rejected_page)
            - (SELECT COUNT(*) FROM owner_page)
            - (SELECT COUNT(*) FROM recovered_page)
        FROM page_budget
    )
), stale_page AS MATERIALIZED (
    SELECT run_attempt_id, updated_on, activated_at, 4 AS attention_rank
    FROM filtered
    WHERE attention_group = 3
      AND (age_seconds > $9) IS NOT TRUE
      AND last_update_age_seconds > $10
    ORDER BY updated_on DESC, run_attempt_id DESC
    LIMIT (
        SELECT candidate_limit
            - (SELECT COUNT(*) FROM rejected_page)
            - (SELECT COUNT(*) FROM owner_page)
            - (SELECT COUNT(*) FROM recovered_page)
            - (SELECT COUNT(*) FROM long_running_page)
        FROM page_budget
    )
), remaining_budget AS (
    SELECT candidate_limit
        - (SELECT COUNT(*) FROM rejected_page)
        - (SELECT COUNT(*) FROM owner_page)
        - (SELECT COUNT(*) FROM recovered_page)
        - (SELECT COUNT(*) FROM long_running_page)
        - (SELECT COUNT(*) FROM stale_page) AS candidate_limit
    FROM page_budget
), ordinary_page AS (
    (
        SELECT run_attempt_id, updated_on, activated_at, 5 AS attention_rank
        FROM filtered
        WHERE attention_group = 3
          AND (age_seconds > $9) IS NOT TRUE
          AND (last_update_age_seconds > $10) IS NOT TRUE
        ORDER BY updated_on DESC, run_attempt_id DESC
        LIMIT (SELECT candidate_limit FROM remaining_budget)
    )
    UNION ALL
    (
        SELECT run_attempt_id, updated_on, activated_at, 5 AS attention_rank
        FROM filtered
        WHERE attention_group = 4
        ORDER BY updated_on DESC, run_attempt_id DESC
        LIMIT (SELECT candidate_limit FROM remaining_budget)
    )
    ORDER BY updated_on DESC, run_attempt_id DESC
    LIMIT (SELECT candidate_limit FROM remaining_budget)
), attention_page AS (
    SELECT * FROM rejected_page
    UNION ALL SELECT * FROM owner_page
    UNION ALL SELECT * FROM recovered_page
    UNION ALL SELECT * FROM long_running_page
    UNION ALL SELECT * FROM stale_page
    UNION ALL SELECT * FROM ordinary_page
    ORDER BY attention_rank, updated_on DESC, run_attempt_id DESC
    LIMIT LEAST(GREATEST($1, 1), 100)
    OFFSET GREATEST($2, 0)
), other_page AS (
    SELECT run_attempt_id, updated_on, activated_at, attention_rank
    FROM filtered
    WHERE $8 IS DISTINCT FROM 'attention'
    ORDER BY
        CASE WHEN $8 = 'attention' THEN attention_rank END,
        CASE WHEN $8 = 'attention' THEN updated_on END DESC,
        CASE WHEN $8 = 'newest' THEN activated_at END DESC,
        CASE WHEN $8 = 'oldest' THEN activated_at END,
        CASE WHEN $8 = 'updated' THEN updated_on END DESC,
        run_attempt_id DESC
    LIMIT LEAST(GREATEST($1, 1), 100)
    OFFSET GREATEST($2, 0)
), page_ids AS (
    SELECT run_attempt_id FROM attention_page
    UNION ALL SELECT run_attempt_id FROM other_page
), paged AS (
    SELECT filtered.*
    FROM page_ids
    JOIN filtered USING (run_attempt_id)
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
    CASE WHEN $8 = 'attention' THEN paged.attention_rank END,
    CASE WHEN $8 = 'attention' THEN paged.updated_on END DESC,
    CASE WHEN $8 = 'newest' THEN paged.activated_at END DESC,
    CASE WHEN $8 = 'oldest' THEN paged.activated_at END,
    CASE WHEN $8 = 'updated' THEN paged.updated_on END DESC,
    paged.run_attempt_id DESC;
    $query$ USING
        p_limit, p_offset, p_states, p_attention, p_gateway,
        p_executor, p_search, p_sort, p_active_too_long_seconds, p_stale_update_seconds;
END;

$function$;

COMMENT ON FUNCTION {schema}.list_operational_attempts(
    INTEGER, INTEGER, TEXT[], TEXT, TEXT, TEXT, TEXT, TEXT, INTEGER, INTEGER
)
IS 'Returns a bounded, payload-free job-attempt audit page with safe attention signals.';
