from pathlib import Path


def test_purge_dags_older_than_is_bounded_to_terminal_work() -> None:
    project_root = Path(__file__).parents[3]
    sql = (
        project_root / "config/psql/schema/088_purge_job_attempts.sql"
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
    assert (
        "set_config( '{schema}.suppress_dag_delete_events', 'on', true )" in normalized
    )


def test_purge_dags_older_than_deletes_attempts_and_preserves_history() -> None:
    project_root = Path(__file__).parents[3]
    sql = (
        project_root / "config/psql/schema/088_purge_job_attempts.sql"
    ).read_text()
    normalized = " ".join(sql.lower().split())

    assert "delete from {schema}.job_attempt as ja" in normalized
    assert "where ja.dag_id = c.id" in normalized
    for table in ("dag_history", "job_history"):
        assert f"delete from {{schema}}.{table}" not in normalized


def test_purge_suppresses_delete_triggers_without_altering_the_table() -> None:
    project_root = Path(__file__).parents[3]
    sql = (
        project_root / "config/psql/schema/085_suppress_dag_purge_events.sql"
    ).read_text()
    normalized = " ".join(sql.lower().split())

    assert "create trigger dag_delete_trigger" in normalized
    assert "create trigger trg_dag_state_changed" in normalized
    assert (
        normalized.count("current_setting('{schema}.suppress_dag_delete_events', true)")
        == 2
    )
    assert "alter table" not in normalized
    assert "disable trigger" not in normalized
