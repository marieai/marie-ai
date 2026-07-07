from __future__ import annotations

from marie.job.common import JobInfo, JobStatus
from marie.job.gateway_job_distributor import GatewayJobDistributor


async def test_identity_keys_promoted_to_parameters() -> None:
    distributor = GatewayJobDistributor()
    metadata = {
        "uri": "s3://b/k",
        "ref_id": "k",
        "ref_type": "rag_document",
        "source_id": "s1",
        "index_name": "i1",
        "tenant_id": "t1",
        "run_params": {"multimodal": True},
    }
    # snapshot before _build_payload runs: parse_payload_to_docs mutates
    # `metadata` in place (e.g. pops "uri"), and job_info.metadata["metadata"]
    # is the same dict object, not a copy.
    expected = dict(metadata)
    job_info = JobInfo(
        status=JobStatus.PENDING,
        entrypoint="rag_indexing:///embed",
        metadata={"metadata": metadata},
    )

    parameters, _asset_doc = await distributor._build_payload("sub-1", job_info)

    for key in ("source_id", "index_name", "tenant_id", "ref_id", "ref_type", "uri"):
        assert parameters[key] == expected[key]
