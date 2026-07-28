import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from marie.scheduler.services.dag_management_service import DAGManagementService


@pytest.mark.asyncio
async def test_hydrated_dag_activation_failure_logs_database_reason() -> None:
    service = DAGManagementService.__new__(DAGManagementService)
    service._admission_lock = asyncio.Lock()
    service.active_dags = {}
    service.max_active_dags = 10
    service._admission_gate = MagicMock(return_value=(True, set()))
    service.repository = MagicMock()
    service.repository.mark_dag_as_active = AsyncMock(return_value=False)
    service.repository.diagnose_dag_activation_failure = AsyncMock(
        return_value={
            "dag_id": "dag-1",
            "reason": "dag_state_not_activatable",
            "dag_state": "completed",
            "hydratable_jobs": 0,
            "active_jobs": 0,
            "total_jobs": 4,
            "job_states": {"completed": 3, "failed": 1},
            "blocking_jobs": [
                {
                    "job_id": "job-failed",
                    "state": "failed",
                    "output": {"error_message": "processor crashed"},
                }
            ],
            "dag_state_history": [{"state": "failed", "changed_on": "now"}],
        }
    )
    service.frontier = MagicMock()
    service.logger = MagicMock()

    admitted, reason = await service._admit_hydrated_dag(
        "dag-1", MagicMock(), [MagicMock()], source="hydrate_single_dag"
    )

    assert admitted is False
    assert reason == "stale_terminal_state"
    message = service.logger.info.call_args.args[0]
    assert "Discarded stale hydration candidate dag-1" in message
    assert "dag_state=completed" in message
    assert "processor crashed" in message
    service.logger.warning.assert_not_called()


@pytest.mark.asyncio
async def test_terminal_dag_resolution_traces_lock_wait(monkeypatch) -> None:
    repository = SimpleNamespace(resolve_dag_state=AsyncMock(return_value="active"))
    service = DAGManagementService(
        repository=repository,
        frontier=MagicMock(),
        active_dags={},
    )
    trace = MagicMock()
    monkeypatch.setattr(
        "marie.scheduler.services.dag_management_service.scheduler_trace",
        trace,
    )

    resolved = await service.resolve_dag_status(
        "job-1",
        SimpleNamespace(dag_id="dag-1"),
    )

    assert resolved is False
    event = trace.call_args_list[0]
    assert event.args[0] == "terminal_dag_lock_acquired"
    assert event.kwargs["job_id"] == "job-1"
    assert event.kwargs["dag_id"] == "dag-1"
    assert event.kwargs["contended"] is False
    assert event.kwargs["wait_ms"] >= 0
