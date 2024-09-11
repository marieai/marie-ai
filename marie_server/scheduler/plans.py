from datetime import datetime, timezone
from typing import Dict

from marie.utils.json import to_json
from marie_server.scheduler.models import WorkInfo
from marie_server.scheduler.state import WorkState


def to_timestamp_with_tz(dt: datetime):
    """
    Convert a datetime object to a timestamp with timezone.
    :param dt:
    :return:
    """
    timestamp = dt.replace(tzinfo=timezone.utc).timestamp()
    return datetime.utcfromtimestamp(timestamp).isoformat() + "Z"


def insert_job(schema: str, work_info: WorkInfo) -> str:
    return f"""
        INSERT INTO {schema}.job (
          id,
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
          policy          
        )
        SELECT
          id,
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
            --ELSE start_after + CAST(COALESCE(keep_until, (q.retention_minutes * 60)::text, keep_until_default, '14 days') as interval)
            -- ELSE start_after + COALESCE(keep_until::interval, (q.retention_minutes * 60) * interval '1 second', keep_until_default, interval '14 days')
          END as keep_until,
          
          COALESCE(j.retry_limit, q.retry_limit, retry_limit_default, 2) as retry_limit,
          CASE
            WHEN COALESCE(j.retry_backoff, q.retry_backoff, retry_backoff_default, false)
            THEN GREATEST(COALESCE(j.retry_delay, q.retry_delay, retry_delay_default), 1)
            ELSE COALESCE(j.retry_delay, q.retry_delay, retry_delay_default, 0)
          END as retry_delay,
          
          COALESCE(j.retry_backoff, q.retry_backoff, retry_backoff_default, false) as retry_backoff,
          q.policy
        FROM
        ( SELECT
                '{work_info.id}'::uuid as id,
                '{work_info.name}'::text as name,
                {work_info.priority}::int as priority,
                '{WorkState.CREATED.value}'::{schema}.job_state as state,
                {work_info.retry_limit}::int as retry_limit,
                --'{to_timestamp_with_tz(work_info.start_after)}'::text as start_after,
                CASE
                  WHEN right('{to_timestamp_with_tz(work_info.start_after)}', 1) = 'Z' THEN CAST('{to_timestamp_with_tz(work_info.start_after)}' as timestamp with time zone)
                  ELSE now() + CAST(COALESCE('{to_timestamp_with_tz(work_info.start_after)}','0') as interval)
                END as start_after,
            
                CAST('{work_info.expire_in_seconds}' as interval) as expire_in,
                '{work_info.data}'::jsonb as data,
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


def create_queue(schema: str, queue_name: str, options: Dict[str, str]) -> str:
    # return f"SELECT {schema}.create_queue('{queue_name}', {to_json(options)})"
    return f"""
            SELECT {schema}.create_queue('{queue_name}', '{{"retry_limit":2}}'::json)
           """


def delete_queue(schema: str, queue_name: str) -> str:
    return f"SELECT {schema}.delete_queue({queue_name})"


def version_table_exists(schema: str) -> str:
    return f"SELECT to_regclass('{schema}.version') as name"


def count_states(schema: str):
    return f"""
    SELECT name, state, count(*) size
    FROM {schema}.job
    GROUP BY ROLLUP(name), ROLLUP(state)
    """


# Example usage:
# schema = 'public'
# print(count_states(schema))


def fetch_next_job(schema: str):
    def query(
        name: str,
        batch_size: int = 1,
        include_metadata: bool = False,
        priority: bool = True,
    ) -> str:
        return f"""
        WITH next AS (
            SELECT id
            FROM {schema}.job
            WHERE name = '{name}'
              AND state < '{WorkState.ACTIVE.value}'
              AND start_after < now()
            ORDER BY {'priority DESC, ' if priority else ''}created_on, id
            LIMIT {batch_size}
            
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

    return query
