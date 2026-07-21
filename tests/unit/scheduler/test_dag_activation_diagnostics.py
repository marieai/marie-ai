import asyncio
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
