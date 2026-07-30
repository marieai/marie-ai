import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from marie.scheduler.psql import PostgreSQLJobScheduler
from marie.scheduler.services.scheduler_diagnostics import SchedulerDiagnostics


@pytest.mark.asyncio
async def test_execution_history_passes_bounded_scope_to_repository() -> None:
    diagnostics = object.__new__(SchedulerDiagnostics)
    diagnostics.repository = SimpleNamespace(
        list_operational_execution_history=AsyncMock(return_value={"items": []})
    )

    result = await diagnostics.execution_history(
        job_id="job-1",
        limit=50,
        offset=100,
    )

    assert result == {"items": []}
    diagnostics.repository.list_operational_execution_history.assert_awaited_once_with(
        job_id="job-1",
        dag_id=None,
        limit=50,
        offset=100,
    )


@pytest.mark.asyncio
async def test_diagnostics_collects_component_owned_runtime_state() -> None:
    repository = SimpleNamespace(
        count_job_states=AsyncMock(return_value={'created': 2}),
        count_dag_states=AsyncMock(return_value={'active': 1}),
    )
    frontier = SimpleNamespace(summary=MagicMock(return_value={'totals': {'jobs': 2}}))
    submission = SimpleNamespace(submission_count=7)
    diagnostics = SchedulerDiagnostics(
        repository=repository,
        frontier=frontier,
        submission_service=submission,
        event_queue=asyncio.Queue(),
        active_dags={'dag-1': SimpleNamespace(status='active')},
        known_queues={'extract'},
        scheduling_engine=SimpleNamespace(
            diagnostics=MagicMock(
                return_value={
                    'sample_count': 3,
                    'latency_ms': {'p95': 4.2},
                }
            )
        ),
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
    }
    assert snapshot['queues']['event_queue_size'] == 0
    assert snapshot['active_dags']['dag-1']['status'] == 'active'
    assert snapshot['sla_monitoring'] == {'warning_top_n': 5}
    assert snapshot['execution_planning'] == {'scheduling_engine_available': True}
    assert snapshot['selection'] == {
        'sample_count': 3,
        'latency_ms': {'p95': 4.2},
    }
    assert 'scheduler_mode' not in snapshot['scheduler_info']
    assert snapshot['job_state_counts'] == {'created': 2}
    assert snapshot['dag_state_counts'] == {'active': 1}


@pytest.mark.asyncio
async def test_scheduler_debug_info_reports_dispatch_pressure() -> None:
    scheduler = object.__new__(PostgreSQLJobScheduler)
    scheduler.running = True
    scheduler._paused = False
    scheduler._fetch_counter = 9
    scheduler._pending_dispatches = {'attempt-1': object()}
    scheduler.dispatch_confirmation_max_in_flight = 4
    scheduler._scheduler_counters = {'run_lease_recovered_retry_total': 2}
    scheduler.diagnostics = SimpleNamespace(
        snapshot=AsyncMock(return_value={'scheduler_info': {}})
    )

    snapshot = await scheduler.debug_info()

    assert snapshot['dispatch'] == {
        'pending_confirmations': 1,
        'confirmation_limit': 4,
        'available_confirmations': 3,
        'utilization_pct': 25.0,
        'counters': {'run_lease_recovered_retry_total': 2},
    }
