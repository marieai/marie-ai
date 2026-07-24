from pathlib import Path


def test_admission_candidate_function_owns_only_dag_level_ordering() -> None:
    project_root = Path(__file__).parents[3]
    sql = (
        project_root
        / "config/psql/schema/lease/010_admission_candidate_dags.sql"
    ).read_text()
    normalized = " ".join(sql.lower().split())

    assert (
        "create or replace function {schema}.admission_candidate_dags(" in normalized
    )
    assert "max(j.priority) as priority" in normalized
    assert "candidate.priority desc" in normalized
    assert "coalesce(d.soft_sla, d.hard_sla) as sla_at" in normalized
    assert "candidate.sla_at asc nulls last" in normalized
    assert "candidate.created_on, candidate.id" in normalized
    assert "job_dependencies" not in normalized
    assert "extract(epoch" not in normalized

    final_order_by = normalized.rsplit("order by", maxsplit=1)[1]
    assert final_order_by.index("candidate.priority desc") < final_order_by.index(
        "candidate.sla_at asc nulls last"
    )
    assert final_order_by.index(
        "candidate.sla_at asc nulls last"
    ) < final_order_by.index("candidate.created_on, candidate.id")

    for planner_signal in (
        "job_level",
        "dag_remaining",
        "estimated_runtime",
        "free_slots",
        "control_flow",
        "_added_at",
    ):
        assert planner_signal not in normalized
