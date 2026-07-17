import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from docarray import DocList

from marie.api.docs import AssetKeyDoc
from marie.executor.guardrail.evaluation_executor import GuardrailEvaluationExecutor
from marie.executor.guardrail.runtime import GuardrailRuntime
from marie.query_planner.guardrail import (
    GuardrailAggregationMode,
    GuardrailExecutionSpec,
    GuardrailMetric,
    GuardrailMetricType,
)


@pytest.mark.asyncio
async def test_runtime_uses_configured_weights() -> None:
    runtime = GuardrailRuntime({})
    spec = GuardrailExecutionSpec(
        metrics=[
            GuardrailMetric(
                type=GuardrailMetricType.LENGTH_CHECK,
                name="length",
                threshold=1.0,
                weight=1.0,
                params={"min": 1},
            ),
            GuardrailMetric(
                type=GuardrailMetricType.REGEX_MATCH,
                name="forbidden",
                threshold=1.0,
                weight=3.0,
                params={"pattern": "forbidden"},
            ),
        ],
        aggregation_mode=GuardrailAggregationMode.WEIGHTED_AVERAGE,
        pass_threshold=0.5,
    )

    report = await runtime.evaluate(spec, "safe output")

    assert report.overall_score == pytest.approx(0.25)
    assert report.overall_passed is False
    assert report.outcome == "INVALID"


@pytest.mark.asyncio
async def test_runtime_rejects_unimplemented_metric() -> None:
    runtime = GuardrailRuntime({})
    spec = GuardrailExecutionSpec(
        metrics=[
            GuardrailMetric(
                type=GuardrailMetricType.LLM_JUDGE,
                name="judge",
                params={"prompt": "Evaluate"},
            )
        ]
    )

    with pytest.raises(NotImplementedError, match="llm_judge"):
        await runtime.evaluate(spec, "output")


@pytest.mark.asyncio
async def test_executor_materializes_report_and_returns_only_report_reference() -> None:
    executor = object.__new__(GuardrailEvaluationExecutor)
    executor.evaluation_registry = {}
    executor.asset_tracking_enabled = True
    executor.asset_tracker = AsyncMock()
    parameters = {
        "job_id": "00000000-0000-0000-0000-000000000101",
        "dag_id": "00000000-0000-0000-0000-000000000201",
        "node_task_id": "00000000-0000-0000-0000-000000000101",
        "run_attempt_id": "00000000-0000-0000-0000-000000000301",
        "payload": {
            "op_params": {
                "input_data": "valid output",
                "guardrail": {
                    "metrics": [
                        {
                            "type": "length_check",
                            "name": "length",
                            "threshold": 1.0,
                            "params": {"min": 1},
                        }
                    ]
                },
            }
        },
    }

    with patch(
        "marie.executor.guardrail.evaluation_executor.StorageManager.write",
        return_value=True,
    ) as write:
        result = await executor._evaluate_guardrail(
            [],
            parameters,
            parameters["payload"]["op_params"]["guardrail"],
        )

    assert set(result) == {"guardrail_report_asset"}
    assert result["guardrail_report_asset"]["asset_key"].startswith("guardrail/report/")

    report = json.loads(write.call_args.args[0].getvalue())
    assert write.call_args.args[1].startswith("s3://")
    assert write.call_args.args[2] is True
    assert report["overall_passed"] is True
    assert report["outcome"] == "VALID"
    assert report["individual_results"][0]["metric_name"] == "length"
    executor.asset_tracker.record_materializations.assert_awaited_once()
    materialization = executor.asset_tracker.record_materializations.await_args.kwargs[
        "assets"
    ][0]
    assert materialization["metadata"]["outcome"] == "VALID"
    assert materialization["metadata"]["run_attempt_id"] == parameters["run_attempt_id"]


@pytest.mark.asyncio
async def test_executor_resolves_upstream_output_from_materialized_asset() -> None:
    node_id = "00000000-0000-0000-0000-000000000101"

    class Cursor:
        def execute(self, _query, _params) -> None:
            pass

        def fetchall(self):
            return [
                (
                    node_id,
                    "annotation/result",
                    "v:sha256:abc",
                    None,
                    "s3://marie/result.json",
                )
            ]

        def close(self) -> None:
            pass

    class Connection:
        def cursor(self):
            return Cursor()

        def commit(self) -> None:
            pass

    executor = object.__new__(GuardrailEvaluationExecutor)
    executor.storage_enabled = True
    executor.storage_handler = SimpleNamespace(
        _get_connection=lambda: Connection(),
        _close_connection=lambda _connection: None,
    )

    with patch(
        "marie.executor.guardrail.evaluation_executor.StorageManager.read",
        return_value=b'{"status": "valid"}',
    ):
        nodes, lineage = await executor._load_upstream_outputs(
            "00000000-0000-0000-0000-000000000201", [node_id]
        )

    assert nodes[node_id]["output"] == {"status": "valid"}
    assert lineage == [("annotation/result", "v:sha256:abc", None)]


@pytest.mark.asyncio
async def test_executor_resolves_declared_document_output() -> None:
    executor = object.__new__(GuardrailEvaluationExecutor)
    spec = GuardrailExecutionSpec(input_source="$.document.output")
    docs = DocList[AssetKeyDoc]([AssetKeyDoc(asset_key="s3://marie/source.json")])

    with patch(
        "marie.executor.guardrail.evaluation_executor.StorageManager.read",
        return_value=b'{"status": "valid"}',
    ) as read:
        context, lineage = await executor._build_guardrail_context(
            docs,
            {"payload": {"op_params": {}}},
            spec,
        )

    assert context["document"]["output"] == {"status": "valid"}
    assert lineage == []
    read.assert_called_once_with("s3://marie/source.json")


@pytest.mark.asyncio
async def test_executor_fails_before_materialization_when_report_write_fails() -> None:
    executor = object.__new__(GuardrailEvaluationExecutor)
    executor.asset_tracking_enabled = True
    executor.asset_tracker = AsyncMock()
    job_id = "00000000-0000-0000-0000-000000000101"
    parameters = {
        "job_id": job_id,
        "dag_id": "00000000-0000-0000-0000-000000000201",
        "node_task_id": job_id,
        "run_attempt_id": "00000000-0000-0000-0000-000000000301",
    }

    with (
        patch(
            "marie.executor.guardrail.evaluation_executor.StorageManager.write",
            return_value=False,
        ),
        pytest.raises(RuntimeError, match="Failed to write guardrail report"),
    ):
        await executor._materialize_report(
            b'{"outcome":"VALID","evaluated_at":"2026-07-16T12:00:00Z"}',
            parameters,
            [],
        )

    executor.asset_tracker.record_materializations.assert_not_awaited()
