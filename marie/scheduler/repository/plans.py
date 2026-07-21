from datetime import datetime, timezone
from typing import Any, Dict

from psycopg import sql
from psycopg.types.json import Jsonb

from marie.scheduler.state import WorkState


def to_timestamp_with_tz(dt: datetime):
    """
    Convert a datetime object to a timestamp with timezone.
    If the datetime is naive (no timezone), it is assumed to be in UTC.
    :param dt: datetime object (timezone-aware or naive), or None
    :return: ISO format string with timezone (Z suffix), or None if input is None
    """
    if dt is None:
        return None

    # If naive, assume UTC; if aware, convert to UTC
    if dt.tzinfo is None:
        dt_utc = dt.replace(tzinfo=timezone.utc)
    else:
        dt_utc = dt.astimezone(timezone.utc)

    return dt_utc.isoformat().replace('+00:00', 'Z')


def try_set_maintenance_time(schema: str, maintenance_state_interval_seconds: int):
    return try_set_timestamp(
        schema, "maintained_on", maintenance_state_interval_seconds
    )


def try_set_monitor_time(schema: str, monitor_state_interval_seconds: int):
    return try_set_timestamp(schema, "monitored_on", monitor_state_interval_seconds)


def try_set_cron_time(schema: str, cron_state_interval_seconds: int):
    return try_set_timestamp(schema, "cron_on", cron_state_interval_seconds)


def try_set_timestamp(schema: str, column: str, interval: int) -> str:
    return f"""
    UPDATE {schema}.version SET {column} = now()
    WHERE EXTRACT(EPOCH FROM (now() - COALESCE({column}, now() - interval '1 week'))) > {interval}
    RETURNING true
    """


def _literal(value: Any) -> str:
    return sql.Literal(value).as_string()


def _jsonb_literal(value: Any) -> str:
    if value is None:
        return "NULL"
    return sql.Literal(Jsonb(value)).as_string()


def insert_jobs(schema: str) -> str:
    return f"""
        WITH jobs AS (
            SELECT *
            FROM jsonb_to_recordset(%s::jsonb) AS job (
                id uuid,
                dag_id uuid,
                name text,
                priority integer,
                data jsonb,
                retry_limit integer,
                retry_delay integer,
                retry_backoff boolean,
                start_after timestamptz,
                expire_in_seconds integer,
                keep_until timestamptz,
                dependencies jsonb,
                job_level integer,
                soft_sla timestamptz,
                hard_sla timestamptz
            )
        )
        INSERT INTO {schema}.job (
            id,
            dag_id,
            name,
            priority,
            state,
            data,
            start_after,
            expire_in,
            keep_until,
            retry_limit,
            retry_delay,
            retry_backoff,
            policy,
            dependencies,
            job_level,
            soft_sla,
            hard_sla
        )
        SELECT
            job.id,
            job.dag_id,
            job.name,
            job.priority,
            '{WorkState.CREATED.value}'::{schema}.job_state,
            job.data,
            job.start_after,
            CASE
                WHEN job.expire_in_seconds IS NOT NULL
                    THEN make_interval(secs => job.expire_in_seconds)
                WHEN queue.expire_seconds IS NOT NULL
                    THEN queue.expire_seconds * interval '1 second'
                ELSE interval '60 seconds'
            END,
            job.keep_until,
            COALESCE(job.retry_limit, queue.retry_limit, 2),
            CASE
                WHEN COALESCE(job.retry_backoff, queue.retry_backoff, false)
                    THEN GREATEST(
                        COALESCE(job.retry_delay, queue.retry_delay, 2),
                        1
                    )
                ELSE COALESCE(job.retry_delay, queue.retry_delay, 2)
            END,
            COALESCE(job.retry_backoff, queue.retry_backoff, false),
            queue.policy,
            COALESCE(job.dependencies, '[]'::jsonb),
            job.job_level,
            job.soft_sla,
            job.hard_sla
        FROM jobs job
        JOIN {schema}.queue queue ON job.name = queue.name
        ON CONFLICT DO NOTHING
        RETURNING id
    """


def insert_dag(
    schema: str,
    dag_id: str,
    dag_name: str,
    serialized_dag: dict,
    soft_sla: datetime = None,
    hard_sla: datetime = None,
    planner: str = None,
) -> str:
    soft_sla_str = (
        f"CAST('{to_timestamp_with_tz(soft_sla)}' as timestamptz)"
        if soft_sla
        else "NULL"
    )
    hard_sla_str = (
        f"CAST('{to_timestamp_with_tz(hard_sla)}' as timestamptz)"
        if hard_sla
        else "NULL"
    )
    planner_str = f"'{planner}'" if planner else "NULL"

    return f"""
        INSERT INTO {schema}.dag (
            id,
            name,
            state,
            serialized_dag,
            soft_sla,
            hard_sla,
            planner
            )
        VALUES (
            '{dag_id}'::uuid,
            '{dag_name}'::text,
            '{WorkState.CREATED.value}',
            {_jsonb_literal(serialized_dag)},
            {soft_sla_str},
            {hard_sla_str},
            {planner_str}
            )
        ON CONFLICT DO NOTHING
    RETURNING id
    """


def insert_job_search_documents(schema: str) -> str:
    return f"""
        WITH documents AS (
            SELECT *
            FROM jsonb_to_recordset(%s::jsonb) AS document (
                job_id uuid,
                queue_name text,
                dag_id uuid,
                planner text,
                job_name text,
                node_label text,
                ref_id text,
                ref_type text,
                asset_uri text,
                metadata_queue_id text,
                layout text,
                mode text,
                policy text,
                method text,
                endpoint text,
                executor text,
                model_name text,
                search_text text
            )
        )
        INSERT INTO {schema}.job_search_document (
            job_id,
            queue_name,
            dag_id,
            planner,
            job_name,
            node_label,
            ref_id,
            ref_type,
            asset_uri,
            metadata_queue_id,
            layout,
            mode,
            policy,
            method,
            endpoint,
            executor,
            model_name,
            search_text
        )
        SELECT
            document.job_id,
            document.queue_name,
            document.dag_id,
            document.planner,
            document.job_name,
            document.node_label,
            document.ref_id,
            document.ref_type,
            document.asset_uri,
            document.metadata_queue_id,
            document.layout,
            document.mode,
            document.policy,
            document.method,
            document.endpoint,
            document.executor,
            document.model_name,
            document.search_text
        FROM documents document
        ON CONFLICT (queue_name, job_id) DO UPDATE
        SET
            dag_id = EXCLUDED.dag_id,
            planner = EXCLUDED.planner,
            job_name = EXCLUDED.job_name,
            node_label = EXCLUDED.node_label,
            ref_id = EXCLUDED.ref_id,
            ref_type = EXCLUDED.ref_type,
            asset_uri = EXCLUDED.asset_uri,
            metadata_queue_id = EXCLUDED.metadata_queue_id,
            layout = EXCLUDED.layout,
            mode = EXCLUDED.mode,
            policy = EXCLUDED.policy,
            method = EXCLUDED.method,
            endpoint = EXCLUDED.endpoint,
            executor = EXCLUDED.executor,
            model_name = EXCLUDED.model_name,
            search_text = EXCLUDED.search_text,
            updated_on = now()
        RETURNING job_id
    """


def load_dag(schema: str, dag_id: str) -> str:
    return f"""
        SELECT serialized_dag FROM {schema}.dag WHERE id = '{dag_id}'::uuid
    """


def create_queue(schema: str, queue_name: str, options: Dict[str, str]) -> str:
    return f"""
            SELECT {schema}.create_queue('{queue_name}', '{{"retry_limit":2}}'::json)
           """


def delete_queue(schema: str, queue_name: str) -> str:
    return f"SELECT {schema}.delete_queue({queue_name})"


def version_table_exists(schema: str) -> str:
    return f"SELECT to_regclass('{schema}.version') as name"


def insert_version(schema: str, version: str) -> str:
    query = (
        f"INSERT INTO {schema}.version(version) VALUES ('{version}') "
        "ON CONFLICT DO NOTHING"
    )
    return query


def count_job_states(schema: str):
    """
    Count the number of jobs in each state.
    """
    return f"SELECT * FROM {schema}.count_job_states()"


def count_dag_states(schema: str):
    """
    Count the number of dags in each state.
    """
    return f"SELECT * FROM {schema}.count_dag_states()"


def cancel_jobs(schema: str, name: str, ids: list):
    ids_string = "ARRAY[" + ",".join(f"'{str(_id)}'" for _id in ids) + "]"
    nonterminal = "', '".join(
        [
            WorkState.CREATED.value,
            WorkState.RETRY.value,
            WorkState.ACTIVE.value,
        ]
    )

    return f"""
    WITH results AS (
      UPDATE {schema}.job
      SET completed_on = now(),
          state = '{WorkState.CANCELLED.value}',
          lease_owner = NULL,
          lease_expires_at = NULL,
          run_owner = NULL,
          run_attempt_id = NULL,
          run_lease_expires_at = NULL
      WHERE name = '{name}'
        AND id IN (SELECT UNNEST({ids_string}::uuid[]))
        AND state::text IN ('{nonterminal}')
      RETURNING 1
    )
    SELECT COUNT(*) FROM results
    """


def cancel_pending_jobs_for_dag(schema: str, dag_id: str, output: dict):
    return f"""
    WITH results AS (
      UPDATE {schema}.job
      SET completed_on = now(),
          state = '{WorkState.CANCELLED.value}',
          output = COALESCE(output, '{{}}'::jsonb) || {_jsonb_literal(output)},
          lease_owner = NULL,
          lease_expires_at = NULL,
          run_owner = NULL,
          run_attempt_id = NULL,
          run_lease_expires_at = NULL
      WHERE dag_id = '{dag_id}'::uuid
        AND state IN ('{WorkState.CREATED.value}', '{WorkState.RETRY.value}')
      RETURNING id
    )
    SELECT COUNT(*) FROM results
    """


def resume_jobs(schema: str, name: str, ids: list):
    ids_string = "ARRAY[" + ",".join(f"'{str(_id)}'" for _id in ids) + "]"

    return f"""
    WITH results AS (
      UPDATE {schema}.job
      SET completed_on = NULL,
          state = '{WorkState.CREATED.value}'
      WHERE name = '{name}'
        AND id IN (SELECT UNNEST({ids_string}::uuid[]))
        AND state = '{WorkState.CANCELLED.value}'
      RETURNING 1
    )
    SELECT COUNT(*) FROM results
    """


def fetch_next_job(schema: str):
    def query(
        name: str,
        batch_size: int = 1,
        include_metadata: bool = False,
        priority: bool = True,
    ) -> str:
        """
        Constructs a SQL query that calls the stored function to fetch the next job(s),
        using the standardized DAG-aware dependency logic and state transitions.
        """
        function_call = f"{schema}.fetch_next_job('{name}', {batch_size})"

        # Select only relevant columns if include_metadata is False
        if include_metadata:
            return f"SELECT * FROM {function_call};"
        else:
            return (
                "SELECT id, name, priority, state, retry_limit, start_after, expire_in, "
                "data, retry_delay, retry_backoff, keep_until, dag_id, job_level "
                f"FROM {function_call};"
            )

    return query


def mark_as_active_jobs(
    schema: str, name: str, ids: list, include_metadata: bool = False
):
    ids_string = "ARRAY[" + ",".join(f"'{str(_id)}'" for _id in ids) + "]"

    return f"""
    WITH next AS (
        SELECT id
        FROM {schema}.job
        WHERE name = '{name}' AND id IN (SELECT UNNEST({ids_string}::uuid[]))
        --FOR UPDATE SKIP LOCKED -- We don't need this because we are using a single worker
    )
    UPDATE {schema}.job j SET
        state = '{WorkState.ACTIVE.value}',
        started_on = now(),
        retry_count = CASE WHEN started_on IS NOT NULL THEN retry_count + 1 ELSE retry_count END
    FROM next
    WHERE name = '{name}' AND j.id = next.id
    RETURNING j.{'*' if include_metadata else 'id,name, priority,state,retry_limit,start_after,expire_in,data,retry_delay,retry_backoff,keep_until'}
    """


def mark_as_active_dags(schema: str, ids: list, include_metadata: bool = False):
    ids_string = "ARRAY[" + ",".join(f"'{str(_id)}'" for _id in ids) + "]"

    return f"""
    WITH next AS (
        SELECT id
        FROM {schema}.dag
        WHERE id IN (SELECT UNNEST({ids_string}::uuid[]))
          AND state IN ('{WorkState.CREATED.value}', '{WorkState.ACTIVE.value}')
        FOR UPDATE
    )
    UPDATE {schema}.dag j SET
        started_on = COALESCE(j.started_on, now()),
        state = '{WorkState.ACTIVE.value}'
    FROM next
    WHERE j.id = next.id
      AND j.state IN ('{WorkState.CREATED.value}', '{WorkState.ACTIVE.value}')
    RETURNING j.{'*' if include_metadata else 'id, name, state '}
    """


def _complete_jobs_query(
    schema: str, name: str, ids: list, output: dict, state_condition: str
):
    ids_string = "ARRAY[" + ",".join(f"'{str(_id)}'" for _id in ids) + "]"
    return f"""
    WITH results AS (
      UPDATE {schema}.job
      SET completed_on = now(),
          state = '{WorkState.COMPLETED.value}',
          output = {_jsonb_literal(output)},
          -- clear leases / run ownership 
          lease_owner          = NULL,
          lease_expires_at     = NULL,
          run_owner            = NULL,
          run_lease_expires_at = NULL  
      WHERE name = '{name}'
        AND id IN (SELECT UNNEST({ids_string}::uuid[]))
        AND {state_condition}
      RETURNING *
    )
    SELECT COUNT(*) FROM results
    """


def complete_jobs(schema: str, name: str, ids: list, output: dict):
    state_condition = f"state = '{WorkState.ACTIVE.value}'"
    return _complete_jobs_query(schema, name, ids, output, state_condition)


def complete_jobs_by_attempt(
    schema: str,
    name: str,
    ids: list,
    output: dict,
    run_owner: str,
    run_attempt_id: str,
):
    state_condition = (
        f"state = '{WorkState.ACTIVE.value}' "
        f"AND run_owner = {_literal(run_owner)} "
        f"AND run_attempt_id = {_literal(run_attempt_id)}::uuid"
    )
    return _complete_jobs_query(schema, name, ids, output, state_condition)


def complete_jobs_by_id(schema: str, name: str, ids: list, output: dict):
    state_condition = "TRUE"  # No state condition for complete_jobs_by_id
    return _complete_jobs_query(schema, name, ids, output, state_condition)


def fail_jobs_by_id(schema: str, name: str, ids: list, output: dict):
    ids_string = "ARRAY[" + ",".join(f"'{str(_id)}'" for _id in ids) + "]"
    where = (
        f"name = '{name}' "
        f"AND id IN (SELECT UNNEST({ids_string}::uuid[])) "
        "AND state::text IN ('created', 'retry', 'active')"
    )
    return fail_jobs(schema, where, output)


def fail_jobs_by_attempt(
    schema: str,
    name: str,
    ids: list,
    output: dict,
    run_owner: str,
    run_attempt_id: str,
):
    ids_string = "ARRAY[" + ",".join(f"'{str(_id)}'" for _id in ids) + "]"
    where = (
        f"name = '{name}' "
        f"AND id IN (SELECT UNNEST({ids_string}::uuid[])) "
        f"AND state = '{WorkState.ACTIVE.value}' "
        f"AND run_owner = {_literal(run_owner)} "
        f"AND run_attempt_id = {_literal(run_attempt_id)}::uuid"
    )
    return fail_jobs(schema, where, output)


def fail_jobs_by_timeout(schema: str):
    where = f"state = '{WorkState.ACTIVE.value}' AND (started_on + expire_in) < now()"
    return fail_jobs(
        schema, where, {"value": {"message": "job failed by timeout in active state"}}
    )


def fail_jobs(schema: str, where: str, output: dict):
    query = f"""
    WITH results AS (
      UPDATE {schema}.job SET
        state = CASE
          WHEN retry_count < retry_limit THEN '{WorkState.RETRY.value}'::{schema}.job_state
          ELSE '{WorkState.FAILED.value}'::{schema}.job_state
          END,
        completed_on = CASE
          WHEN retry_count < retry_limit THEN NULL
          ELSE now()
          END,
        start_after = CASE
          WHEN retry_count = retry_limit THEN start_after
          WHEN NOT retry_backoff THEN now() + retry_delay * interval '1'
          ELSE {schema}.exponential_backoff(retry_delay, retry_count)
          END,
        output = {_jsonb_literal(output)},
        -- clear leases / run ownership
        lease_owner          = NULL,
        lease_expires_at     = NULL,
        run_owner            = NULL,
        run_attempt_id       = CASE
          WHEN retry_count < retry_limit THEN NULL
          ELSE run_attempt_id
          END,
        run_lease_expires_at = NULL
      WHERE {where}
      RETURNING *
    ), dlq_jobs AS (
      INSERT INTO {schema}.job (name, data, output, retry_limit, keep_until)
      SELECT
        dead_letter,
        data,
        output,
        retry_limit,
        keep_until + (keep_until - start_after)
      FROM results
      WHERE state = '{WorkState.FAILED.value}'
        AND dead_letter IS NOT NULL
        AND NOT name = dead_letter
    )
    SELECT COUNT(*), (SELECT state::text FROM results LIMIT 1) as final_state FROM results
    """
    return query


def get_active_jobs(schema: str) -> str:
    """
    Get all items in the active state.
    :param schema: The schema name.
    """
    return f"""
    SELECT *
    FROM {schema}.job
    WHERE state = '{WorkState.ACTIVE.value}'
    """
