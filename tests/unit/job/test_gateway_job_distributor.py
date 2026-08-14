from __future__ import annotations

from marie.job.common import JobInfo, JobStatus
from marie.job.gateway_job_distributor import GatewayJobDistributor


async def test_build_payload_passes_mock_failure_controls_to_executor_parameters() -> (
    None
):
    distributor = GatewayJobDistributor()
    job_info = JobInfo(
        status=JobStatus.PENDING,
        entrypoint="mock_executor:///document/process",
        metadata={
            "metadata": {
                "uri": "s3://marie/extract/sample.tif",
                "ref_id": "sample",
                "ref_type": "extract",
                "failure_rate": 0.5,
                "failure_mode": "timeout",
                "force_fail": True,
                "randomize_time": True,
            }
        },
    )

    parameters, asset_doc = await distributor._build_payload("job-a", job_info)

    assert asset_doc.asset_key == "s3://marie/extract/sample.tif"
    assert parameters["job_id"] == "job-a"
    assert parameters["failure_rate"] == 0.5
    assert parameters["failure_mode"] == "timeout"
    assert parameters["force_fail"] is True
    assert parameters["randomize_time"] is True


async def test_build_payload_preserves_datasource_for_retry() -> None:
    distributor = GatewayJobDistributor()
    payload = {
        "uri": "s3://marie/extract/sample.tif",
        "ref_id": "sample",
        "ref_type": "extract",
    }
    job_info = JobInfo(
        status=JobStatus.PENDING,
        entrypoint="mock_executor:///document/process",
        metadata={"metadata": payload},
    )

    first_parameters, first_asset = await distributor._build_payload("job-a", job_info)
    second_parameters, second_asset = await distributor._build_payload(
        "job-a", job_info
    )

    assert payload["uri"] == "s3://marie/extract/sample.tif"
    assert first_asset.asset_key == second_asset.asset_key == payload["uri"]
    assert "uri" not in first_parameters["payload"]
    assert "uri" not in second_parameters["payload"]


def test_resolve_endpoint_accepts_mock_annotator_llm_route() -> None:
    distributor = GatewayJobDistributor(
        deployment_nodes={
            "annotator_llm": [
                {"endpoint": "/annotator/llm"},
            ],
        }
    )

    assert distributor._resolve_endpoint(
        "job-a",
        "annotator_llm://annotator/llm",
    ) == ("annotator_llm", "/annotator/llm")


async def test_build_payload_passes_run_attempt_identity() -> None:
    distributor = GatewayJobDistributor()
    job_info = JobInfo(
        status=JobStatus.PENDING,
        entrypoint="guardrail_executor://evaluate",
        metadata={
            "metadata": {
                "uri": "s3://marie/extract/sample.json",
                "ref_id": "sample",
                "ref_type": "extract",
            },
            "dag_id": "00000000-0000-0000-0000-000000000201",
            "node_task_id": "00000000-0000-0000-0000-000000000101",
            "run_owner": "scheduler-1",
            "run_attempt_id": "00000000-0000-0000-0000-000000000301",
        },
    )

    parameters, _ = await distributor._build_payload("job-a", job_info)

    assert parameters["run_owner"] == "scheduler-1"
    assert parameters["run_attempt_id"] == job_info.metadata["run_attempt_id"]


def test_resolve_endpoint_routes_connector_to_plugin_daemon_execute() -> None:
    # W2: a CONNECTOR node serializes to the FIXED endpoint
    # plugin_daemon_executor://execute (identity rides in params, not the path).
    distributor = GatewayJobDistributor(
        deployment_nodes={
            "plugin_daemon_executor": [
                {"endpoint": "/execute"},
            ],
        }
    )

    assert distributor._resolve_endpoint(
        "job-a",
        "plugin_daemon_executor://execute",
    ) == ("plugin_daemon_executor", "/execute")


def test_resolve_endpoint_routes_agent_application() -> None:
    distributor = GatewayJobDistributor(
        deployment_nodes={
            "agent_executor": [
                {"endpoint": "/agent/run"},
            ],
        }
    )

    assert distributor._resolve_endpoint(
        "job-a",
        "agent_executor://agent/run",
    ) == ("agent_executor", "/agent/run")
