-- File: 082_list_operational_execution_history.sql
-- Description: Bounded worker execution history for one job or DAG
-- Dependencies: 005_job.sql, 007_dag.sql, 050_kv_store_worker.sql

CREATE OR REPLACE FUNCTION {schema}.list_operational_execution_history(
    p_job_id UUID DEFAULT NULL,
    p_dag_id UUID DEFAULT NULL,
    p_limit INTEGER DEFAULT 50,
    p_offset INTEGER DEFAULT 0
)
RETURNS TABLE (
    total_count BIGINT,
    scope_dag_id UUID,
    history_id BIGINT,
    task_job_id UUID,
    queue_name TEXT,
    change_time TIMESTAMPTZ,
    operation TEXT,
    worker_status TEXT,
    worker_message TEXT,
    run_attempt_id TEXT,
    executor TEXT,
    runtime_name TEXT,
    executor_host TEXT,
    endpoint TEXT,
    error_type TEXT,
    error_message TEXT,
    error_file TEXT,
    error_function TEXT,
    error_line TEXT
)
LANGUAGE SQL
STABLE
AS $function$
WITH scope AS (
    SELECT
        COALESCE(p_dag_id, seed.dag_id) AS dag_id,
        p_job_id AS seed_job_id
    FROM (SELECT 1) AS singleton
    LEFT JOIN {schema}.job AS seed ON seed.id = p_job_id
    WHERE (
            p_dag_id IS NOT NULL
            AND EXISTS (
                SELECT 1 FROM {schema}.dag AS d WHERE d.id = p_dag_id
            )
        )
       OR (p_job_id IS NOT NULL AND seed.id IS NOT NULL)
), tasks AS (
    SELECT j.id, j.name::TEXT
    FROM {schema}.job AS j
    CROSS JOIN scope AS s
    WHERE (s.dag_id IS NOT NULL AND j.dag_id = s.dag_id)
       OR (s.dag_id IS NULL AND j.id = s.seed_job_id)
), history_source AS (
    SELECT
        kh.history_id::BIGINT,
        t.id AS task_job_id,
        t.name AS queue_name,
        kh.change_time,
        kh.operation::TEXT,
        NULLIF(LEFT(REGEXP_REPLACE(
            COALESCE(kh.value->>'status', ''),
            '[[:cntrl:]]+',
            ' ',
            'g'
        ), 128), '') AS worker_status,
        NULLIF(LEFT(REGEXP_REPLACE(
            COALESCE(kh.value->>'message', ''),
            '[[:cntrl:]]+',
            ' ',
            'g'
        ), 1024), '') AS worker_message,
        NULLIF(LEFT(kh.value->>'run_attempt_id', 128), '') AS run_attempt_id,
        NULLIF(LEFT(REGEXP_REPLACE(
            COALESCE(env #>> '{attributes,executor}', ''),
            '[[:cntrl:]]+',
            ' ',
            'g'
        ), 256), '') AS executor,
        NULLIF(LEFT(REGEXP_REPLACE(
            COALESCE(env #>> '{attributes,runtime_name}', ''),
            '[[:cntrl:]]+',
            ' ',
            'g'
        ), 256), '') AS runtime_name,
        NULLIF(LEFT(REGEXP_REPLACE(
            COALESCE(env #>> '{attributes,host}', ''),
            '[[:cntrl:]]+',
            ' ',
            'g'
        ), 256), '') AS executor_host,
        NULLIF(LEFT(REGEXP_REPLACE(
            COALESCE(env #>> '{attributes,executor_endpoint}', ''),
            '[[:cntrl:]]+',
            ' ',
            'g'
        ), 256), '') AS endpoint,
        NULLIF(LEFT(REGEXP_REPLACE(
            COALESCE(env #>> '{error,type}', ''),
            '[[:cntrl:]]+',
            ' ',
            'g'
        ), 256), '') AS error_type,
        NULLIF(LEFT(REGEXP_REPLACE(
            COALESCE(env #>> '{error,message}', ''),
            '[[:cntrl:]]+',
            ' ',
            'g'
        ), 2048), '') AS error_message,
        NULLIF(LEFT(REGEXP_REPLACE(
            COALESCE(env #>> '{error,filename}', ''),
            '[[:cntrl:]]+',
            ' ',
            'g'
        ), 512), '') AS error_file,
        NULLIF(LEFT(REGEXP_REPLACE(
            COALESCE(env #>> '{error,name}', ''),
            '[[:cntrl:]]+',
            ' ',
            'g'
        ), 256), '') AS error_function,
        NULLIF(LEFT(REGEXP_REPLACE(
            COALESCE(env #>> '{error,line_no}', ''),
            '[[:cntrl:]]+',
            ' ',
            'g'
        ), 32), '') AS error_line
    FROM tasks AS t
    JOIN {schema}.kv_store_worker_history AS kh
      ON kh.namespace = 'job'
     AND kh.key = 'marie_internal/job_info_' || t.id::TEXT
    CROSS JOIN LATERAL (
        SELECT COALESCE(
            NULLIF(kh.value->>'runtime_env_json', '')::JSONB,
            '{}'::JSONB
        ) AS env
    ) AS parsed
), page_metadata AS (
    SELECT COUNT(*) AS total_count FROM history_source
), page AS (
    SELECT *
    FROM history_source
    ORDER BY change_time DESC, history_id DESC
    LIMIT p_limit OFFSET p_offset
)
SELECT
    metadata.total_count,
    scope.dag_id,
    page.history_id,
    page.task_job_id,
    page.queue_name,
    page.change_time,
    page.operation,
    page.worker_status,
    page.worker_message,
    page.run_attempt_id,
    page.executor,
    page.runtime_name,
    page.executor_host,
    page.endpoint,
    page.error_type,
    page.error_message,
    page.error_file,
    page.error_function,
    page.error_line
FROM scope
CROSS JOIN page_metadata AS metadata
LEFT JOIN page ON TRUE
ORDER BY page.change_time DESC NULLS LAST, page.history_id DESC NULLS LAST;
$function$;

COMMENT ON FUNCTION {schema}.list_operational_execution_history(
    UUID,
    UUID,
    INTEGER,
    INTEGER
) IS 'Returns server-paged worker status history and bounded structured errors for one job or DAG without raw runtime environment data or tracebacks.';
