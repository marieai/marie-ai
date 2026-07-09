from __future__ import annotations

from datetime import datetime, timezone

from marie.executor.kb.vector_store_executor import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    _run_param,
)
from marie.job.common import JobInfo, JobStatus
from marie.job.gateway_job_distributor import GatewayJobDistributor
from marie.query_planner.base import QueryPlanRegistry
from marie.query_planner.kb_indexing_planner import PLAN_ID as KB_INDEXING_PLAN_ID
from marie.query_planner.kb_indexing_planner import query_planner_kb_indexing
from marie.scheduler.planner_util import query_plan_work_items
from marie.sensors.daemon.worker import build_work_info
from marie.sensors.definitions.data_sink.base import FileObject
from marie.sensors.definitions.kb_document_sensor import KB_KEY_RE, KbDocumentSensor


async def test_identity_keys_promoted_to_parameters() -> None:
    distributor = GatewayJobDistributor()
    metadata = {
        "uri": "s3://b/k",
        "ref_id": "k",
        "ref_type": "kb_document",
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
        entrypoint="kb_indexing:///embed",
        metadata={"metadata": metadata},
    )

    parameters, _asset_doc = await distributor._build_payload("sub-1", job_info)

    for key in ("source_id", "index_name", "tenant_id", "ref_id", "ref_type", "uri"):
        assert parameters[key] == expected[key]


async def test_run_params_promoted_to_parameters() -> None:
    """run_params (chunk_size/chunk_overlap/segmentation_mode/multimodal) must
    reach the top level of `parameters`, not just parameters["payload"]["run_params"]."""
    distributor = GatewayJobDistributor()
    metadata = {
        "uri": "s3://b/k",
        "ref_id": "k",
        "ref_type": "kb_document",
        "source_id": "s1",
        "index_name": "i1",
        "run_params": {
            "chunk_size": 256,
            "chunk_overlap": 32,
            "segmentation_mode": "character",
            "multimodal": True,
        },
    }
    job_info = JobInfo(
        status=JobStatus.PENDING,
        entrypoint="kb_indexing:///embed",
        metadata={"metadata": metadata},
    )

    parameters, _asset_doc = await distributor._build_payload("sub-1", job_info)

    assert parameters["run_params"] == {
        "chunk_size": 256,
        "chunk_overlap": 32,
        "segmentation_mode": "character",
        "multimodal": True,
    }
    # still reachable via the old nested location for anything reading the
    # full envelope directly
    assert parameters["payload"]["run_params"]["chunk_size"] == 256


async def test_kb_indexing_job_chunk_params_reach_embed_and_store_parameters() -> None:
    """End-to-end: KB document sensor binding -> WorkInfo -> DAG nodes via the
    real kb_indexing planner -> GatewayJobDistributor._build_payload for the
    EMBED_AND_STORE node. Reproduces the sensor -> scheduler -> distributor
    path a real KB indexing job takes, without hand-building `parameters`.
    """
    QueryPlanRegistry.register(KB_INDEXING_PLAN_ID, query_planner_kb_indexing)

    key = (
        "tenants/11111111-1111-4111-8111-111111111111"
        "/kb-indexes/22222222-2222-4222-8222-222222222222"
        "/sources/33333333-3333-4333-8333-333333333333/report.pdf"
    )
    sensor = KbDocumentSensor(
        {
            "id": "sid",
            "name": "kb-document-sensor",
            "config": {"subtype": "kb_document", "provider": "s3", "prefix": "tenants/"},
        }
    )
    obj = FileObject(key=key, size=10, last_modified=datetime.now(timezone.utc), etag="x")
    binding = {
        "workflow_name": "kb_indexing",
        "run_params": {
            "chunk_size": 512,
            "chunk_overlap": 64,
            "segmentation_mode": "character",
            "multimodal": True,
        },
    }
    run_request = sensor._build_run_request(
        obj, bucket="marie", m=KB_KEY_RE.match(key), binding=binding
    )

    root_work_info = build_work_info(
        run_request,
        sensor_id="sid",
        sensor_name="kb-document-sensor",
        job_name=run_request.job_name,
        dag_id=None,
    )

    _plan, dag_nodes = query_plan_work_items(root_work_info)
    embed_wi = next(wi for wi in dag_nodes if "EMBED" in wi.data["metadata"]["name"])

    job_metadata = embed_wi.data.copy()
    job_info = JobInfo(
        status=JobStatus.PENDING,
        entrypoint=embed_wi.data["metadata"]["on"],
        metadata=job_metadata,
    )

    distributor = GatewayJobDistributor()
    parameters, _asset_doc = await distributor._build_payload("sub-embed-1", job_info)

    assert _run_param(parameters, "chunk_size", DEFAULT_CHUNK_SIZE) == 512
    assert _run_param(parameters, "chunk_overlap", DEFAULT_CHUNK_OVERLAP) == 64
    assert _run_param(parameters, "segmentation_mode", "character") == "character"
    # identity fields still flow through unaffected by the run_params fix
    assert parameters["source_id"] == "33333333-3333-4333-8333-333333333333"
    assert parameters["index_name"] == "22222222-2222-4222-8222-222222222222"
