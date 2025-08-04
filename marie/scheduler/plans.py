from datetime import datetime, timezone
from typing import Any, Dict

from psycopg2.extras import Json

from marie.scheduler.models import WorkInfo
from marie.scheduler.state import WorkState


def to_timestamp_with_tz(dt: datetime):
    """
    Convert a datetime object to a timestamp with timezone.
    :param dt:
    :return:
    """
    timestamp = dt.replace(tzinfo=timezone.utc).timestamp()
    return datetime.utcfromtimestamp(timestamp).isoformat() + "Z"


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


def insert_job(schema: str, work_info: WorkInfo) -> str:
    dependencies_json = (
        Json(work_info.dependencies) if work_info.dependencies else "'[]'::jsonb"
    )

    return f"""
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
          id,
          j.dag_id,
          j.name,
          priority,
          state,         
          data,
          start_after,
          CASE
            WHEN expire_in IS NOT NULL THEN CAST(expire_in as interval)
            WHEN q.expire_seconds IS NOT NULL THEN q.expire_seconds * interval '1s'
            WHEN expire_in_default IS NOT NULL THEN CAST(expire_in_default as interval)
            ELSE interval '15 minutes'
          END as expire_in,
          CASE
            WHEN right(keep_until, 1) = 'Z' THEN CAST(keep_until as timestamp with time zone)
          END as keep_until,

          COALESCE(j.retry_limit, q.retry_limit, retry_limit_default, 2) as retry_limit,
          CASE
            WHEN COALESCE(j.retry_backoff, q.retry_backoff, retry_backoff_default, false)
            THEN GREATEST(COALESCE(j.retry_delay, q.retry_delay, retry_delay_default), 1)
            ELSE COALESCE(j.retry_delay, q.retry_delay, retry_delay_default, 0)
          END as retry_delay,

          COALESCE(j.retry_backoff, q.retry_backoff, retry_backoff_default, false) as retry_backoff,
          q.policy,          
          {dependencies_json} as dependencies,
          j.job_level,
          j.soft_sla,
          j.hard_sla
        FROM
        ( SELECT
                '{work_info.id}'::uuid as id,
                '{work_info.dag_id}'::uuid as dag_id,
                '{work_info.name}'::text as name,
                {work_info.priority}::int as priority,
                {work_info.job_level}::int as job_level,                
                CAST('{to_timestamp_with_tz(work_info.soft_sla)}' as timestamp with time zone) as soft_sla,
                CAST('{to_timestamp_with_tz(work_info.hard_sla)}' as timestamp with time zone) as hard_sla,
                '{WorkState.CREATED.value}'::{schema}.job_state as state,
                {work_info.retry_limit}::int as retry_limit,
                CASE
                  WHEN right('{to_timestamp_with_tz(work_info.start_after)}', 1) = 'Z' THEN CAST('{to_timestamp_with_tz(work_info.start_after)}' as timestamp with time zone)
                  ELSE now() + CAST(COALESCE('{to_timestamp_with_tz(work_info.start_after)}','0') as interval)
                END as start_after,

                CAST('{work_info.expire_in_seconds}' as interval) as expire_in,
                {Json(work_info.data)}::jsonb as data,
                {work_info.retry_delay}::int as retry_delay,
                {work_info.retry_backoff}::bool as retry_backoff,
                '{to_timestamp_with_tz(work_info.keep_until)}'::text as keep_until,

                2::int as retry_limit_default,
                2::int as retry_delay_default,
                False::boolean as retry_backoff_default,
                interval '60s'::interval as expire_in_default,
                now() + interval '14 days'::interval as keep_until_default
        )  j JOIN {schema}.queue q ON j.name = q.name
    ON CONFLICT DO NOTHING
    RETURNING id
    """


def insert_dag(schema: str, dag_id: str, dag_name: str, serialized_dag: dict) -> str:
    return f"""
        INSERT INTO {schema}.dag (
            id,
            name,
            state,
            serialized_dag
            )
        VALUES (
            '{dag_id}'::uuid,
            '{dag_name}'::text,
            '{WorkState.CREATED.value}',
            {Json(serialized_dag)}::jsonb
            ) 
        ON CONFLICT DO NOTHING
        RETURNING id
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
    query = f"INSERT INTO {schema}.version(version) VALUES ('{version}')"
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

    return f"""
    WITH results AS (
      UPDATE {schema}.job
      SET completed_on = now(),
          state = '{WorkState.CANCELLED.value}'
      WHERE name = {name}
        AND id IN (SELECT UNNEST({ids_string}::uuid[]))
        AND state < '{WorkState.COMPLETED.value}'
      RETURNING 1
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
      WHERE name = {name}
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
        --FOR UPDATE SKIP LOCKED -- We don't need this because we are using a single worker
    )
    UPDATE {schema}.dag j SET
        started_on = now()
    FROM next
    WHERE  j.id = next.id
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
          output = {Json(output)}::jsonb
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


def complete_jobs_by_id(schema: str, name: str, ids: list, output: dict):
    state_condition = "TRUE"  # No state condition for complete_jobs_by_id
    return _complete_jobs_query(schema, name, ids, output, state_condition)


def complete_jobs_by_id(schema: str, name: str, ids: list, output: dict):
    ids_string = "ARRAY[" + ",".join(f"'{str(_id)}'" for _id in ids) + "]"
    query = f"""
    WITH results AS (
      UPDATE {schema}.job
      SET completed_on = now(),
          state = '{WorkState.COMPLETED.value}',
          output = {Json(output)}::jsonb
      WHERE name = '{name}'
        AND id IN (SELECT UNNEST({ids_string}::uuid[]))
      RETURNING *
    )
    SELECT COUNT(*) FROM results
    """
    return query


def fail_jobs_by_id(schema: str, name: str, ids: list, output: dict):
    ids_string = "ARRAY[" + ",".join(f"'{str(_id)}'" for _id in ids) + "]"
    where = f"name = '{name}' AND id IN (SELECT UNNEST({ids_string}::uuid[])) AND state < '{WorkState.COMPLETED.value}'"
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
        output = {Json(output)}::jsonb
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
    SELECT COUNT(*) FROM results
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
