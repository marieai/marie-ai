from pathlib import Path


def test_lease_jobs_disables_jit() -> None:
    project_root = Path(__file__).resolve().parents[3]
    schema_sql = project_root.joinpath(
        "config/psql/schema/lease/001_lease_jobs_by_id.sql"
    ).read_text()

    assert "SET jit = off" in schema_sql
