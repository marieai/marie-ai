from unittest.mock import AsyncMock

import pytest

from marie.query_planner.base import Query, QueryPlan, QueryType
from marie.query_planner.guardrail import GuardrailPath, GuardrailQueryDefinition
from marie.scheduler.job_lock import AsyncJobLock
from marie.scheduler.models import WorkInfo
from marie.scheduler.psql import PostgreSQLJobScheduler
from marie.scheduler.state import WorkState


@pytest.mark.asyncio
async def test_scheduler_builds_branch_metadata_from_guardrail_report() -> None:
    guardrail_id = "00000000-0000-0000-0000-000000000101"
    pass_id = "00000000-0000-0000-0000-000000000102"
    fail_id = "00000000-0000-0000-0000-000000000103"
    dag_id = "00000000-0000-0000-0000-000000000201"
    attempt_id = "00000000-0000-0000-0000-000000000301"
    plan = QueryPlan(
        nodes=[
            Query(
                task_id=guardrail_id,
                query_str="guardrail",
                node_type=QueryType.GUARDRAIL,
                definition=GuardrailQueryDefinition(
                    paths=[
                        GuardrailPath(path_id="pass", target_node_ids=[pass_id]),
                        GuardrailPath(path_id="fail", target_node_ids=[fail_id]),
                    ]
                ),
            ),
            Query(task_id=pass_id, query_str="pass", dependencies=[guardrail_id]),
            Query(task_id=fail_id, query_str="fail", dependencies=[guardrail_id]),
        ]
    )
    report_asset = {
        "asset_key": f"guardrail/report/{guardrail_id}",
        "asset_version": "v:sha256:abc",
        "partition_key": None,
    }

    scheduler = object.__new__(PostgreSQLJobScheduler)
    scheduler.get_dag_by_id = AsyncMock(return_value=plan)
    scheduler.repository = AsyncMock()
    scheduler.repository.get_guardrail_report_decision.return_value = {
        "outcome": "VALID",
        "evaluated_at": "2026-07-16T12:00:00+00:00",
        "report_asset": report_asset,
    }
    scheduler.repository.commit_guardrail_route.return_value = (
        True,
        {fail_id},
        None,
    )
    scheduler.frontier = AsyncMock()
    scheduler._status_update_lock = AsyncJobLock()
    scheduler._job_cache = {}
    work_item = WorkInfo.model_construct(
        id=guardrail_id,
        dag_id=dag_id,
        name="extract",
        data={},
        state=WorkState.ACTIVE,
        branch_metadata=None,
    )

    committed, skipped, reject_reason = (
        await scheduler._commit_guardrail_route_if_needed(
            guardrail_id,
            work_item,
            run_owner="scheduler-1",
            run_attempt_id=attempt_id,
        )
    )

    assert committed is True
    assert skipped == {fail_id}
    assert reject_reason is None
    branch_metadata = scheduler.repository.commit_guardrail_route.await_args.kwargs[
        "branch_metadata"
    ]
    assert branch_metadata["node_type"] == "GUARDRAIL"
    assert branch_metadata["outcome"] == "VALID"
    assert branch_metadata["selected_path_ids"] == ["pass"]
    assert branch_metadata["report_asset"] == report_asset
    assert work_item.branch_metadata == branch_metadata
    scheduler.frontier.on_job_completed_with_skips.assert_awaited_once_with(
        guardrail_id, {fail_id}
    )
