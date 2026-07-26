from pathlib import Path

from marie.scheduler.repository.plans import complete_jobs_by_attempt

PROJECT_ROOT = Path(__file__).parents[3]


def test_complete_job_only_returns_count_input() -> None:
    query = complete_jobs_by_attempt(
        "marie_scheduler",
        "extract",
        ["00000000-0000-0000-0000-000000000001"],
        {},
        "scheduler-1",
        "00000000-0000-0000-0000-000000000002",
    )

    assert "RETURNING 1" in query
    assert "RETURNING *" not in query


def test_expired_lease_release_is_bounded_and_lock_safe() -> None:
    sql = (
        PROJECT_ROOT
        / "config/psql/schema/lease/011_release_expired_leases_bounded.sql"
    ).read_text()

    assert "_max_rows integer DEFAULT 1000" in sql
    assert "LIMIT COALESCE(_max_rows, 1000)" in sql
    assert "FOR UPDATE OF j SKIP LOCKED" in sql
    assert "WHERE j.name = cand.name" in sql


def test_duration_cron_jobs_are_removed() -> None:
    sql = (PROJECT_ROOT / "config/psql/cron_job_init.sql").read_text()

    assert "cron.schedule(" not in sql
    assert "cron.unschedule(jobid)" in sql
    assert "'refresh_job_priority'" in sql
    assert "'refresh_job_durations'" in sql
    assert "'refresh_dag_durations'" in sql


def test_lease_expiration_indexes_match_maintenance_predicates() -> None:
    sql = (
        PROJECT_ROOT / "config/psql/schema/073_scheduler_hot_path_indexes.sql"
    ).read_text()

    assert "job_expired_acquisition_lease_idx" in sql
    assert "ON {schema}.job (lease_expires_at, id)" in sql
    assert "state IN ('created', 'retry')" in sql
    assert "job_expired_run_lease_idx" in sql
    assert "ON {schema}.job (run_lease_expires_at, id)" in sql
    assert "state = 'active'" in sql
