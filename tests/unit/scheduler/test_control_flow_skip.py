from pathlib import Path

import pytest

from marie.query_planner.base import Query, QueryPlan
from marie.scheduler.services.control_flow_execution_service import (
    exclusive_skip_closure,
)


def branch_plan() -> QueryPlan:
    return QueryPlan(
        nodes=[
            Query(task_id="branch", query_str="branch"),
            Query(task_id="path-a", query_str="path a", dependencies=["branch"]),
            Query(task_id="path-b", query_str="path b", dependencies=["branch"]),
            Query(task_id="path-c", query_str="path c", dependencies=["branch"]),
            Query(task_id="leaf-a", query_str="leaf a", dependencies=["path-a"]),
            Query(task_id="leaf-b", query_str="leaf b", dependencies=["path-b"]),
            Query(task_id="leaf-c", query_str="leaf c", dependencies=["path-c"]),
            Query(
                task_id="merger",
                query_str="merger",
                dependencies=["leaf-a", "leaf-b", "leaf-c"],
            ),
            Query(task_id="end", query_str="end", dependencies=["merger"]),
        ]
    )


@pytest.mark.parametrize(
    ("skipped_targets", "expected"),
    [
        (
            ["path-b", "path-c"],
            ["path-b", "path-c", "leaf-b", "leaf-c"],
        ),
        (["path-c"], ["path-c", "leaf-c"]),
    ],
)
def test_exclusive_skip_closure_preserves_shared_merger(
    skipped_targets: list[str], expected: list[str]
) -> None:
    assert exclusive_skip_closure(branch_plan(), skipped_targets) == expected


def test_exclusive_skip_closure_handles_nested_branch() -> None:
    plan = QueryPlan(
        nodes=[
            Query(task_id="outer-branch", query_str="outer branch"),
            Query(
                task_id="active-leaf",
                query_str="active leaf",
                dependencies=["outer-branch"],
            ),
            Query(
                task_id="inactive-root",
                query_str="inactive root",
                dependencies=["outer-branch"],
            ),
            Query(
                task_id="inner-branch",
                query_str="inner branch",
                dependencies=["inactive-root"],
            ),
            Query(
                task_id="inner-a",
                query_str="inner a",
                dependencies=["inner-branch"],
            ),
            Query(
                task_id="inner-b",
                query_str="inner b",
                dependencies=["inner-branch"],
            ),
            Query(
                task_id="merger",
                query_str="merger",
                dependencies=["active-leaf", "inner-a", "inner-b"],
            ),
            Query(task_id="end", query_str="end", dependencies=["merger"]),
        ]
    )

    assert exclusive_skip_closure(plan, ["inactive-root"]) == [
        "inactive-root",
        "inner-branch",
        "inner-a",
        "inner-b",
    ]


def test_frontier_hydration_treats_skipped_parents_as_satisfied() -> None:
    project_root = Path(__file__).parents[3]
    sql = (
        project_root / "config/psql/schema/lease/016_hydrate_frontier_jobs.sql"
    ).read_text()

    assert "p.state NOT IN ('completed','failed','cancelled','skipped')" in sql
    assert "'expire_in_seconds', EXTRACT(EPOCH FROM j.expire_in)::integer" in sql
