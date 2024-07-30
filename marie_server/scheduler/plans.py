from datetime import datetime, timezone

from marie_server.scheduler.models import WorkInfo
from marie_server.scheduler.state import States


def to_timestamp_with_tz(dt: datetime):
    timestamp = dt.replace(tzinfo=timezone.utc).timestamp()
    return datetime.utcfromtimestamp(timestamp).isoformat() + "Z"


def insert_job(schema: str, work_info: WorkInfo) -> str:
    return f"""
        INSERT INTO {schema}.job (
          id,
          name,
          priority,
          state,
          retry_limit,
          start_after,
          expire_in,
          data,
          retry_delay,
          retry_backoff,
          keep_until,
          on_complete
        )
        SELECT
          id,
          name,
          priority,
          state,
          retry_limit,
          start_after,
          expire_in,
          data,
          retry_delay,
          retry_backoff,
          keep_until,
          on_complete
        FROM
        ( SELECT *,
            CASE
              WHEN right(keepUntilValue, 1) = 'Z' THEN CAST(keepUntilValue as timestamp with time zone)
              ELSE start_after + CAST(COALESCE(keepUntilValue,'0') as interval)
              END as keep_until
          FROM
          ( SELECT *,
              CASE
                WHEN right(startAfterValue, 1) = 'Z' THEN CAST(startAfterValue as timestamp with time zone)
                ELSE now() + CAST(COALESCE(startAfterValue,'0') as interval)
                END as start_after
            FROM
            ( SELECT
                '{work_info.id}'::uuid as id,
                '{work_info.name}'::text as name,
                {work_info.priority}::int as priority,
                '{States.CREATED.value}'::{schema}.job_state as state,
                {work_info.retry_limit}::int as retry_limit,
                '{to_timestamp_with_tz(work_info.start_after)}'::text as startAfterValue,
                CAST('{work_info.expire_in_seconds}' as interval) as expire_in,
                '{work_info.data}'::jsonb as data,
                {work_info.retry_delay}::int as retry_delay,
                {work_info.retry_backoff}::bool as retry_backoff,
                '{to_timestamp_with_tz(work_info.keep_until)}'::text as keepUntilValue,
                {work_info.on_complete}::boolean as on_complete
        ) j1
      ) j2
    ) j3
    ON CONFLICT DO NOTHING
    RETURNING id
    """
