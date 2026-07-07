from datetime import datetime, timedelta, timezone

from marie.sensors.daemon.worker import build_job_data, build_work_info
from marie.sensors.types import RunRequest


def test_metadata_envelope_built_from_run_config():
    rr = RunRequest(
        run_key="k",
        job_name="rag_indexing",
        run_config={
            "uri": "s3://b/tenants/t1/rag-indexes/i1/sources/s1/a.pdf",
            "ref_id": "tenants/t1/rag-indexes/i1/sources/s1/a.pdf",
            "ref_type": "rag_document",
            "tenant_id": "t1",
            "index_id": "i1",
            "index_name": "i1",
            "source_id": "s1",
            "run_params": {"parse_mode": "agent", "multimodal": True},
        },
        tags={"trigger": "kb_document"},
    )
    data = build_job_data(rr, sensor_id="sid", sensor_name="kb-document-sensor")
    assert data["run_config"] == rr.run_config          # legacy field intact
    md = data["metadata"]
    assert md["planner"] == "rag_indexing"
    assert md["project_id"] == "t1"
    assert md["ref_id"] == rr.run_config["ref_id"]
    assert md["ref_type"] == "rag_document"
    assert md["run_params"] == {"parse_mode": "agent", "multimodal": True}
    assert data["run_key"] == "k"
    assert data["trigger"] == "kb_document"             # tags still spread


def test_envelope_omits_missing_keys():
    rr = RunRequest(run_key="k2", job_name="extract", run_config={"uri": "s3://x/y"})
    data = build_job_data(rr, sensor_id="s", sensor_name="n")
    assert "index_id" not in data["metadata"]
    assert data["metadata"]["planner"] == "extract"


def test_build_work_info_sets_required_scheduler_fields():
    rr = RunRequest(
        run_key="k3", job_name="rag_indexing", priority=5, run_config={"uri": "s3://x/y"}
    )
    before = datetime.now(timezone.utc)
    work_info = build_work_info(
        rr, sensor_id="sid", sensor_name="n", job_name="rag_indexing", dag_id="dag-1"
    )
    after = datetime.now(timezone.utc)

    assert before <= work_info.start_after <= after
    assert work_info.expire_in_seconds == 0
    assert work_info.keep_until - work_info.start_after == timedelta(days=2)
    assert work_info.soft_sla == work_info.start_after
    assert work_info.hard_sla == work_info.start_after + timedelta(hours=4)
    assert work_info.name == "rag_indexing"
    assert work_info.dag_id == "dag-1"
    assert work_info.priority == 5
    assert work_info.data == build_job_data(rr, sensor_id="sid", sensor_name="n")
