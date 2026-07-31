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
    FROM {schema}.job AS j
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
                FROM {schema}.dag AS searched_dag
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
                FROM {schema}.job_attempt AS searched_attempt
                WHERE searched_attempt.run_attempt_id = j.run_attempt_id
                  AND COALESCE(searched_attempt.executor, '')
                        ILIKE '%' || p_search || '%'
            )
        )
), page_metadata AS (
    SELECT COUNT(*) AS total_count
    FROM eligible_jobs AS eligible
    LEFT JOIN {schema}.job_attempt AS attempt
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
            (SELECT ARRAY_AGG(name ORDER BY name) FROM {schema}.queue),
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
    JOIN {schema}.job_attempt AS attempt
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
                FROM {schema}.job_attempt AS attempt
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
                FROM {schema}.job_attempt AS attempt
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
    LEFT JOIN {schema}.job_attempt AS attempt
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
    JOIN {schema}.job AS job ON job.id = selected.job_id
    LEFT JOIN {schema}.dag AS dag ON dag.id = job.dag_id
    LEFT JOIN {schema}.job_attempt AS attempt
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
