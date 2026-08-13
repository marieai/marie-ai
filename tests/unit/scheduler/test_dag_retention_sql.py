from pathlib import Path


def test_purge_dags_older_than_is_bounded_to_terminal_work() -> None:
    project_root = Path(__file__).parents[3]
    sql = (
        project_root / "config/psql/schema/084_purge_dags_older_than.sql"
    ).read_text()
    normalized = " ".join(sql.lower().split())

    assert "create or replace function {schema}.purge_dags_older_than(" in normalized
    assert "p_older_than_hours integer" in normalized
    assert "p_planner_name text default null" in normalized
    assert "p_older_than_hours is null or p_older_than_hours <= 0" in normalized
    assert "make_interval(hours => p_older_than_hours)" in normalized
    assert "d.state in ('completed', 'failed', 'cancelled', 'expired')" in normalized
    assert "d.completed_on is not null" in normalized
    assert "d.completed_on < v_cutoff" in normalized
    assert "j.state::text not in (" in normalized
    assert "for update of d skip locked" in normalized
    assert "delete from {schema}.dag as d" in normalized


def test_purge_dags_older_than_preserves_audit_tables() -> None:
    project_root = Path(__file__).parents[3]
    sql = (
        project_root / "config/psql/schema/084_purge_dags_older_than.sql"
    ).read_text()
    normalized = " ".join(sql.lower().split())

    for table in ("dag_history", "job_history", "job_attempt"):
        assert f"delete from {{schema}}.{table}" not in normalized
