-- Locate a gateway submission across scheduler and worker persistence layers.
-- Replace the UUID in target.
-- On legacy partitioned schemas, run with SET LOCAL jit = off; see QUERY_GUIDE.md.
-- EXPLAIN (ANALYZE, BUFFERS, SETTINGS)

WITH target(input_id) AS (
    VALUES ('00000000-0000-0000-0000-000000000000'::uuid)
),
resolved_dags(dag_id) AS (
    SELECT d.id
    FROM marie_scheduler.dag d
    JOIN target t ON t.input_id = d.id

    UNION

    SELECT dh.id
    FROM marie_scheduler.dag_history dh
    JOIN target t ON t.input_id = dh.id

    UNION

    SELECT j.dag_id
    FROM marie_scheduler.job j
    JOIN target t ON t.input_id = j.id

    UNION

    SELECT jh.dag_id
    FROM marie_scheduler.job_history jh
    JOIN target t ON t.input_id = jh.id

    UNION

    SELECT ja.dag_id
    FROM marie_scheduler.job_attempt ja
    JOIN target t
      ON t.input_id = ja.job_id
      OR t.input_id = ja.dag_id
),
task_ids(job_id) AS (
    SELECT j.id
    FROM marie_scheduler.job j
    WHERE j.dag_id IN (SELECT dag_id FROM resolved_dags)

    UNION

    SELECT jh.id
    FROM marie_scheduler.job_history jh
    WHERE jh.dag_id IN (SELECT dag_id FROM resolved_dags)

    UNION

    SELECT t.input_id
    FROM target t
    WHERE EXISTS (
        SELECT 1
        FROM marie_scheduler.job j
        WHERE j.id = t.input_id
    )
),
-- Keep downstream lookups indexable when legacy job partitions inflate the
-- estimated cardinality of task_ids.
task_keys(job_id, worker_key) AS (
    SELECT
        ti.job_id,
        ('marie_internal/job_info_' || ti.job_id::text)::varchar(1024)
    FROM task_ids ti
),
scope_ids(dag_ids, job_ids, worker_keys) AS (
    SELECT
        ARRAY(SELECT rd.dag_id FROM resolved_dags rd),
        ARRAY(SELECT ti.job_id FROM task_ids ti),
        ARRAY(SELECT tk.worker_key FROM task_keys tk)
),
matched_attempt_ids(run_attempt_id) AS (
    SELECT ja.run_attempt_id
    FROM scope_ids s
    JOIN marie_scheduler.job_attempt ja
      ON ja.job_id = ANY(s.job_ids)

    UNION

    SELECT ja.run_attempt_id
    FROM scope_ids s
    JOIN marie_scheduler.job_attempt ja
      ON ja.dag_id = ANY(s.dag_ids)
),
records AS (
    SELECT
        'dag.current'::text AS record_type,
        COALESCE(d.updated_on, d.created_on) AS event_time,
        d.id AS submission_id,
        NULL::uuid AS job_id,
        NULL::uuid AS run_attempt_id,
        d.state::text AS state,
        jsonb_strip_nulls(jsonb_build_object(
            'name', d.name,
            'submission_name', d.submission_name,
            'planner', d.planner,
            'project_id', d.project_id,
            'ref_type', d.ref_type,
            'ref_id', d.ref_id,
            'policy', d.policy,
            'priority', d.priority,
            'task_count', d.task_count,
            'created_on', d.created_on,
            'started_on', d.started_on,
            'completed_on', d.completed_on
        )) AS details
    FROM marie_scheduler.dag d
    WHERE d.id IN (SELECT dag_id FROM resolved_dags)

    UNION ALL

    SELECT
        'dag.history',
        dh.history_created_on,
        dh.id,
        NULL::uuid,
        NULL::uuid,
        dh.state::text,
        jsonb_strip_nulls(jsonb_build_object(
            'name', dh.name,
            'submission_name', dh.submission_name,
            'planner', dh.planner,
            'project_id', dh.project_id,
            'ref_type', dh.ref_type,
            'ref_id', dh.ref_id,
            'policy', dh.policy,
            'priority', dh.priority,
            'task_count', dh.task_count,
            'created_on', dh.created_on,
            'started_on', dh.started_on,
            'completed_on', dh.completed_on
        ))
    FROM marie_scheduler.dag_history dh
    WHERE dh.id IN (SELECT dag_id FROM resolved_dags)

    UNION ALL

    SELECT
        'job.current',
        COALESCE(j.completed_on, j.started_on, j.created_on),
        j.dag_id,
        j.id,
        j.run_attempt_id,
        j.state::text,
        jsonb_strip_nulls(jsonb_build_object(
            'queue_name', j.name,
            'job_level', j.job_level,
            'endpoint', j.data #>> '{metadata,on}',
            'planner', j.data #>> '{metadata,planner}',
            'dependencies', j.dependencies,
            'retry_count', j.retry_count,
            'retry_limit', j.retry_limit,
            'start_after', j.start_after,
            'lease_owner', j.lease_owner,
            'lease_expires_at', j.lease_expires_at,
            'run_owner', j.run_owner,
            'run_lease_expires_at', j.run_lease_expires_at,
            'output', j.output
        ))
    FROM marie_scheduler.job j
    WHERE j.id IN (SELECT job_id FROM task_ids)

    UNION ALL

    SELECT
        'job.history',
        jh.history_created_on,
        jh.dag_id,
        jh.id,
        jh.run_attempt_id,
        jh.state::text,
        jsonb_strip_nulls(jsonb_build_object(
            'queue_name', jh.name,
            'job_level', jh.job_level,
            'endpoint', jh.data #>> '{metadata,on}',
            'dependencies', jh.dependencies,
            'retry_count', jh.retry_count,
            'retry_limit', jh.retry_limit,
            'start_after', jh.start_after,
            'started_on', jh.started_on,
            'completed_on', jh.completed_on,
            'output', jh.output
        ))
    FROM marie_scheduler.job_history jh
    WHERE jh.id IN (SELECT job_id FROM task_ids)

    UNION ALL

    SELECT
        'attempt',
        COALESCE(ja.updated_on, ja.created_on),
        ja.dag_id,
        ja.job_id,
        ja.run_attempt_id,
        ja.attempt_state,
        jsonb_strip_nulls(jsonb_build_object(
            'queue_name', ja.job_name,
            'executor', ja.executor,
            'run_owner', ja.run_owner,
            'gateway_instance_id', ja.gateway_instance_id,
            'activated_at', ja.activated_at,
            'terminal_at', ja.terminal_at,
            'terminal_status', ja.terminal_status,
            'terminal_work_state', ja.terminal_work_state,
            'terminal_accepted', ja.terminal_accepted,
            'terminal_reject_reason', ja.terminal_reject_reason,
            'recovery_at', ja.recovery_at,
            'recovery_state', ja.recovery_state,
            'recovery_reason', ja.recovery_reason
        ))
    FROM matched_attempt_ids mai
    JOIN marie_scheduler.job_attempt ja
      ON ja.run_attempt_id = mai.run_attempt_id

    UNION ALL

    SELECT
        'worker.current',
        COALESCE(kv.updated_at, kv.created_at),
        COALESCE(
            j.dag_id,
            (SELECT dag_id FROM resolved_dags LIMIT 1)
        ),
        tk.job_id,
        NULL::uuid,
        kv.value->>'status',
        jsonb_strip_nulls(jsonb_build_object(
            'message', kv.value->>'message',
            'run_owner', kv.value->>'run_owner',
            'run_attempt_id', kv.value->>'run_attempt_id',
            'executor', env.runtime_env #>> '{attributes,executor}',
            'runtime_name', env.runtime_env #>> '{attributes,runtime_name}',
            'host', env.runtime_env #>> '{attributes,host}',
            'endpoint', env.runtime_env #>> '{attributes,executor_endpoint}',
            'error_type', env.runtime_env #>> '{error,type}',
            'error_message', env.runtime_env #>> '{error,message}'
        ))
    FROM scope_ids s
    JOIN marie_scheduler.kv_store_worker kv
      ON kv.namespace = 'job'
     AND kv.key = ANY(s.worker_keys)
     AND kv.is_deleted = false
    JOIN task_keys tk ON tk.worker_key = kv.key
    LEFT JOIN marie_scheduler.job j ON j.id = tk.job_id
    LEFT JOIN LATERAL (
        SELECT COALESCE(
            NULLIF(kv.value->>'runtime_env_json', '')::jsonb,
            '{}'::jsonb
        ) AS runtime_env
    ) env ON true

    UNION ALL

    SELECT
        'worker.history',
        kh.change_time,
        COALESCE(
            j.dag_id,
            (SELECT dag_id FROM resolved_dags LIMIT 1)
        ),
        tk.job_id,
        NULL::uuid,
        kh.value->>'status',
        jsonb_strip_nulls(jsonb_build_object(
            'operation', kh.operation,
            'message', kh.value->>'message',
            'run_owner', kh.value->>'run_owner',
            'run_attempt_id', kh.value->>'run_attempt_id',
            'executor', env.runtime_env #>> '{attributes,executor}',
            'runtime_name', env.runtime_env #>> '{attributes,runtime_name}',
            'host', env.runtime_env #>> '{attributes,host}',
            'endpoint', env.runtime_env #>> '{attributes,executor_endpoint}',
            'error_type', env.runtime_env #>> '{error,type}',
            'error_message', env.runtime_env #>> '{error,message}',
            'error_file', env.runtime_env #>> '{error,filename}',
            'error_function', env.runtime_env #>> '{error,name}',
            'error_line', env.runtime_env #>> '{error,line_no}'
        ))
    FROM scope_ids s
    JOIN marie_scheduler.kv_store_worker_history kh
      ON kh.namespace = 'job'
     AND kh.key = ANY(s.worker_keys)
    JOIN task_keys tk ON tk.worker_key = kh.key
    LEFT JOIN marie_scheduler.job j ON j.id = tk.job_id
    LEFT JOIN LATERAL (
        SELECT COALESCE(
            NULLIF(kh.value->>'runtime_env_json', '')::jsonb,
            '{}'::jsonb
        ) AS runtime_env
    ) env ON true
)
SELECT
    record_type,
    event_time,
    submission_id,
    job_id,
    run_attempt_id,
    state,
    details
FROM records

UNION ALL

SELECT
    'not_found',
    now(),
    t.input_id,
    NULL::uuid,
    NULL::uuid,
    NULL::text,
    jsonb_build_object(
        'message',
        'No durable scheduler, attempt, or worker record contains this UUID'
    )
FROM target t
WHERE NOT EXISTS (SELECT 1 FROM records)

ORDER BY event_time, record_type, job_id NULLS FIRST;
