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
