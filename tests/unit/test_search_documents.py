from datetime import UTC, datetime, timedelta

from marie.query_planner.base import LlmQueryDefinition, Query, QueryPlan
from marie.scheduler.models import WorkInfo
from marie.scheduler.search_documents import build_job_search_documents
from marie.scheduler.state import WorkState


def test_build_job_search_documents_extracts_submission_and_node_fields() -> None:
    now = datetime.now(UTC)
    plan = QueryPlan(
        nodes=[
            Query(
                task_id="018f3c07-31d2-7d7a-8000-000000000001",
                query_str="11: Claims verifier",
                dependencies=[],
                node_type="COMPUTE",
                definition=LlmQueryDefinition(
                    endpoint="/annotator/llm",
                    model_name="qwen_3_instruct",
                    params={"layout": "122421", "key": "claims"},
                ),
            )
        ]
    )
    work_info = WorkInfo(
        id="018f3c07-31d2-7d7a-8000-000000000001",
        dag_id="018f3c07-31d2-7d7a-8000-0000000000aa",
        name="gen5_extract",
        priority=0,
        data={
            "metadata": {
                "on": "annotator_executor://annotator/llm",
                "uri": "s3://marie/incoming/PID_2504_9946_0_266161783.tif",
                "mode": "multiline",
                "name": "11: Claims verifier",
                "policy": "allow_all",
                "ref_id": "PID_2504_9946_0_266161783.tif",
                "planner": "122421",
                "queue_id": "22f11f58-e1ab-4337-92c0-b3873ea8c023",
                "ref_type": "lbxid",
                "op_params": {"layout": "122421"},
            }
        },
        state=WorkState.CREATED,
        retry_limit=2,
        retry_delay=2,
        retry_backoff=False,
        start_after=now,
        expire_in_seconds=900,
        keep_until=now + timedelta(days=14),
    )

    documents = build_job_search_documents(
        plan=plan,
        dag_nodes=[work_info],
        planner="122421",
    )

    assert len(documents) == 1
    document = documents[0]
    assert document.job_id == work_info.id
    assert document.queue_name == "gen5_extract"
    assert document.dag_id == work_info.dag_id
    assert document.planner == "122421"
    assert document.node_label == "11: Claims verifier"
    assert document.ref_id == "PID_2504_9946_0_266161783.tif"
    assert document.ref_type == "lbxid"
    assert document.asset_uri == "s3://marie/incoming/PID_2504_9946_0_266161783.tif"
    assert document.metadata_queue_id == "22f11f58-e1ab-4337-92c0-b3873ea8c023"
    assert document.layout == "122421"
    assert document.mode == "multiline"
    assert document.policy == "allow_all"
    assert document.method == "LLM"
    assert document.endpoint == "/annotator/llm"
    assert document.executor == "annotator_executor"
    assert document.model_name == "qwen_3_instruct"
    assert "pid_2504_9946_0_266161783.tif" in document.search_text
    assert "annotator_executor://annotator/llm" in document.search_text
