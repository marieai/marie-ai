import pytest

from marie.query_planner.base import Query, QueryType
from marie.query_planner.guardrail import GuardrailQueryDefinition
from marie.query_planner.mapper import JobMetadata


def test_guardrail_mapper_dispatches_typed_execution_spec() -> None:
    upstream_id = "00000000-0000-0000-0000-000000000101"
    task = Query(
        task_id="00000000-0000-0000-0000-000000000102",
        query_str="validate",
        dependencies=[upstream_id],
        node_type=QueryType.GUARDRAIL,
        definition=GuardrailQueryDefinition(
            input_source=f"$.nodes['{upstream_id}'].output"
        ),
    )

    metadata = JobMetadata.from_task(task, "base").metadata

    assert metadata.on == "guardrail_executor://evaluate"
    assert metadata.op_params["guardrail"]["upstream_node_ids"] == [upstream_id]
    assert metadata.op_params["guardrail"]["input_source"] == (
        f"$.nodes['{upstream_id}'].output"
    )


def test_guardrail_mapper_rejects_non_guardrail_executor() -> None:
    task = Query(
        task_id="00000000-0000-0000-0000-000000000102",
        query_str="validate",
        node_type=QueryType.GUARDRAIL,
        definition=GuardrailQueryDefinition(endpoint="annotator_executor://validate"),
    )

    with pytest.raises(ValueError, match="must execute on guardrail_executor"):
        JobMetadata.from_task(task, "base")
