"""On-demand job diagnostics using the sources in job_failure_analysis.sql."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from marie.storage.database.postgres_pool import AsyncPostgresConnectionPool

HISTORY_LIMIT = 250
TEXT_LIMIT = 65_536


def _error_json(expression: str) -> str:
    fields = (
        'type',
        'message',
        'filename',
        'name',
        'line_no',
        'traceback',
        'traceback_truncated',
        'category',
        'state',
        'confirmed',
        'request_id',
        'primary_task_id',
        'total',
        'failed_count',
        'failed_tasks',
        'failed_tasks_truncated',
        'cause',
        'batch_error_type',
    )
    pairs = ', '.join(f"'{field}', {expression}->'{field}'" for field in fields)
    return f'jsonb_strip_nulls(jsonb_build_object({pairs}))'


_WORKER_EVENT = f"""jsonb_build_object(
    'history_id', kh.history_id, 'changed_at', kh.change_time,
    'operation', kh.operation, 'job_id', j.id, 'queue', j.name,
    'status', kh.value->>'status', 'worker_message', kh.value->>'message',
    'run_attempt_id', kh.value->>'run_attempt_id',
    'run_owner', kh.value->>'run_owner',
    'start_time_ms', kh.value->'start_time', 'end_time_ms', kh.value->'end_time',
    'executor', env #>> '{{attributes,executor}}',
    'runtime_name', env #>> '{{attributes,runtime_name}}',
    'executor_host', env #>> '{{attributes,host}}',
    'endpoint', env #>> '{{attributes,executor_endpoint}}',
    'error', {_error_json("(env->'error')")}
)"""

_QUERY = f"""
WITH target AS (
    SELECT j.* FROM marie_scheduler.job j WHERE j.id = %s::uuid
), worker_events AS NOT MATERIALIZED (
    SELECT kh.history_id, kh.change_time, {_WORKER_EVENT} AS event
    FROM target j
    JOIN marie_scheduler.kv_store_worker_history kh
      ON kh.namespace = 'job'
     AND kh.key = 'marie_internal/job_info_' || j.id::text
    CROSS JOIN LATERAL (
        SELECT COALESCE(NULLIF(kh.value->>'runtime_env_json', '')::jsonb,
                        '{{}}'::jsonb) AS env
    ) parsed
), selected AS (
    SELECT event FROM worker_events WHERE history_id = %s
), worker_page AS (
    SELECT event, change_time, history_id FROM worker_events
    ORDER BY change_time DESC, history_id DESC LIMIT {HISTORY_LIMIT + 1}
), scheduler_page AS (
    SELECT jsonb_build_object(
        'history_id', h.history_id, 'changed_at', h.history_created_on,
        'status', h.state::text, 'run_attempt_id', h.run_attempt_id,
        'retry_count', h.retry_count, 'retry_limit', h.retry_limit,
        'started_at', h.started_on, 'completed_at', h.completed_on,
        'failure_source', h.output->>'failure_source',
        'message', h.output->>'error_message',
        'error', {_error_json("(h.output->'error')")}
    ) AS event, h.history_created_on, h.history_id
    FROM marie_scheduler.job_history h JOIN target j ON j.id = h.id
    ORDER BY h.history_created_on DESC, h.history_id DESC LIMIT {HISTORY_LIMIT + 1}
), attempt_events AS NOT MATERIALIZED (
    SELECT jsonb_build_object(
        'run_attempt_id', a.run_attempt_id, 'run_owner', a.run_owner,
        'scheduler_lease_owner', a.scheduler_lease_owner,
        'gateway_instance_id', a.gateway_instance_id, 'executor', a.executor,
        'attempt_state', a.attempt_state, 'activated_at', a.activated_at,
        'dispatch_started_at', a.dispatch_started_at,
        'dispatch_confirmed_at', a.dispatch_confirmed_at,
        'dispatch_error', a.dispatch_error, 'terminal_at', a.terminal_at,
        'terminal_status', a.terminal_status, 'terminal_work_state', a.terminal_work_state,
        'terminal_source', a.terminal_source, 'terminal_accepted', a.terminal_accepted,
        'terminal_reject_reason', a.terminal_reject_reason,
        'terminal_gateway_instance_id', a.terminal_gateway_instance_id,
        'terminal_scheduler_lease_owner', a.terminal_scheduler_lease_owner,
        'recovery_at', a.recovery_at, 'recovery_state', a.recovery_state,
        'recovery_reason', a.recovery_reason, 'updated_at', a.updated_on
    ) AS event, a.activated_at, a.run_attempt_id
    FROM marie_scheduler.job_attempt a JOIN target j ON j.id = a.job_id
), attempt_page AS (
    SELECT * FROM attempt_events
    ORDER BY activated_at DESC, run_attempt_id DESC LIMIT {HISTORY_LIMIT + 1}
)
SELECT jsonb_build_object(
    'job', jsonb_build_object(
        'job_id', j.id, 'dag_id', j.dag_id, 'queue', j.name,
        'submission_name', d.submission_name, 'planner', d.planner,
        'project_id', d.project_id, 'ref_type', COALESCE(d.ref_type, j.data #>> '{{metadata,ref_type}}'),
        'ref_id', COALESCE(d.ref_id, j.data #>> '{{metadata,ref_id}}'),
        'priority', d.priority, 'task_count', d.task_count,
        'scheduler_state', j.state::text, 'retry_count', j.retry_count,
        'retry_limit', j.retry_limit, 'run_attempt_id', j.run_attempt_id,
        'created_at', j.created_on, 'started_at', j.started_on, 'completed_at', j.completed_on,
        'failure_source', j.output->>'failure_source',
        'scheduler_error_message', j.output->>'error_message',
        'scheduler_error', {_error_json("(j.output->'error')")},
        'worker_status', kv.value->>'status', 'worker_message', kv.value->>'message',
        'worker_end_time_ms', kv.value->'end_time'
    ),
    'selected_event', (SELECT event FROM selected),
    'selected_attempt', (SELECT event FROM attempt_events
                         WHERE run_attempt_id::text = (SELECT event->>'run_attempt_id' FROM selected)),
    'worker_history', COALESCE((SELECT jsonb_agg(event ORDER BY change_time DESC, history_id DESC)
                              FROM worker_page), '[]'::jsonb),
    'scheduler_history', COALESCE((SELECT jsonb_agg(event ORDER BY history_created_on DESC, history_id DESC)
                                 FROM scheduler_page), '[]'::jsonb),
    'attempts', COALESCE((SELECT jsonb_agg(event ORDER BY activated_at DESC, run_attempt_id DESC)
                        FROM attempt_page), '[]'::jsonb)
)
FROM target j
LEFT JOIN marie_scheduler.dag d ON d.id = j.dag_id
LEFT JOIN marie_scheduler.kv_store_worker kv
  ON kv.namespace = 'job' AND kv.key = 'marie_internal/job_info_' || j.id::text
 AND kv.is_deleted = FALSE
"""

_CREDENTIALS = re.compile(
    r'''(?i)(\b(?:api[_-]?key|access[_-]?token|password|secret)["']?\s*[=:]\s*["']?)([^\s,;"'}]+)'''
)
_BEARER = re.compile(r'(?i)(\bBearer\s+)[A-Za-z0-9._~+/=-]+')
_URL_PASSWORD = re.compile(r'(https?://)[^\s/@]+:[^\s/@]+@', re.IGNORECASE)


async def read_failure_report(
    pool: AsyncPostgresConnectionPool, job_id: str, history_id: int
) -> dict[str, Any] | None:
    """Read selected-event diagnostics and retained timelines in one statement."""
    async with pool.acquire() as connection:
        row = await connection.fetchrow(_QUERY, job_id, history_id)
    if not row or row[0] is None:
        return None
    report = row[0]
    if report.get('selected_event') is None:
        raise LookupError('Execution event not found for this job')
    truncated: dict[str, bool] = {}
    for section in ('worker_history', 'scheduler_history', 'attempts'):
        entries = report.get(section, [])
        truncated[section] = len(entries) > HISTORY_LIMIT
        report[section] = entries[:HISTORY_LIMIT]
    truncated['text'] = False

    def scrub(value: Any) -> Any:
        if isinstance(value, str):
            value = _CREDENTIALS.sub(r'\1[redacted]', value)
            value = _BEARER.sub(r'\1[redacted]', value)
            value = _URL_PASSWORD.sub(r'\1[redacted]@', value)
            encoded = value.encode('utf-8')
            if len(encoded) > TEXT_LIMIT:
                truncated['text'] = True
                value = (
                    encoded[:TEXT_LIMIT].decode('utf-8', errors='ignore')
                    + '\n[truncated]'
                )
            return ''.join(
                c for c in value if c in '\n\t' or ord(c) >= 32 and ord(c) != 127
            )
        if isinstance(value, dict):
            if value.get('traceback_truncated') is True:
                truncated['text'] = True
            return {key: scrub(item) for key, item in value.items()}
        if isinstance(value, list):
            return [scrub(item) for item in value]
        return value

    report = scrub(report)
    report.update(
        schema_version='1.0',
        generated_at=datetime.now(timezone.utc).isoformat(),
        history_limit=HISTORY_LIMIT,
        text_limit=TEXT_LIMIT,
        truncated=truncated,
        source='config/psql/schema/monitoring/job_failure_analysis.sql',
        raw_runtime_environment_suppressed=True,
        task_payload_suppressed=True,
    )
    return report
