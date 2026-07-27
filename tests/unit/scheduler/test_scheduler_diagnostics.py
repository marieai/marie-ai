import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from marie.scheduler.services.scheduler_diagnostics import SchedulerDiagnostics


@pytest.mark.asyncio
async def test_diagnostics_collects_component_owned_runtime_state() -> None:
    repository = SimpleNamespace(
        count_job_states=AsyncMock(return_value={'created': 2}),
        count_dag_states=AsyncMock(return_value={'active': 1}),
    )
    frontier = SimpleNamespace(summary=MagicMock(return_value={'totals': {'jobs': 2}}))
    submission = SimpleNamespace(
        submission_count=7,
        pending_count=1,
        queue_size=1,
        status=MagicMock(return_value={'queue_size': 1}),
    )
    runtime = SimpleNamespace(tasks=MagicMock(return_value=[]))
    diagnostics = SchedulerDiagnostics(
        repository=repository,
        frontier=frontier,
        submission_service=submission,
        runtime=runtime,
        event_queue=asyncio.Queue(),
        active_dags={'dag-1': SimpleNamespace(status='active')},
        known_queues={'extract'},
        execution_planner=object(),
        gateway_instance_id='gateway-1',
        lease_owner='scheduler-1',
        max_concurrent_dags=16,
        start_time=datetime.now(timezone.utc),
        sla_warning_top_n=5,
        frontier_batch_size=100,
        lease_ttl_seconds=5,
    )

    snapshot = await diagnostics.snapshot(
        running=True,
        paused=False,
        fetch_counter=3,
    )

    assert snapshot['counters'] == {
        'fetch_counter': 3,
        'submission_count': 7,
        'pending_requests': 1,
    }
    assert snapshot['queues']['request_queue_size'] == 1
    assert snapshot['active_dags']['dag-1']['status'] == 'active'
    assert snapshot['sla_monitoring'] == {'warning_top_n': 5}
    assert 'scheduler_mode' not in snapshot['scheduler_info']
    assert snapshot['job_state_counts'] == {'created': 2}
    assert snapshot['dag_state_counts'] == {'active': 1}
    runtime.tasks.assert_called_once_with(prefix='scheduler-submission-')
