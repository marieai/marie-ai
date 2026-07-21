from pathlib import Path

from marie.scheduler.repository.plans import mark_as_active_dags


def test_hydration_discovers_only_recoverable_dag_states() -> None:
    project_root = Path(__file__).parents[3]
    sql = (
        project_root / "config/psql/schema/lease/008_hydrate_frontier.sql"
    ).read_text()

    assert "d.state IN ('created', 'active')" in sql
    assert "j.state IN ('created', 'retry')" in sql
    assert "blocker.dag_id = d.id" in sql
    assert "blocker.state::text IN ('failed', 'expired', 'cancelled')" in sql


def test_dag_activation_rechecks_recoverable_state_under_lock() -> None:
    query = mark_as_active_dags(
        "marie_scheduler", ["00000000-0000-0000-0000-000000000001"]
    )

    assert query.count("state IN ('created', 'active')") == 2
    assert "FOR UPDATE" in query
