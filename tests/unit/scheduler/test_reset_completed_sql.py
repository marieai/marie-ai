from pathlib import Path


def test_completed_reset_refreshes_scheduler_attention_timestamps() -> None:
    project_root = Path(__file__).resolve().parents[3]
    sql = project_root.joinpath(
        "config/psql/schema/034_reset_completed_dags_and_jobs.sql"
    ).read_text()

    job_update = sql.split("UPDATE {schema}.job", maxsplit=1)[1].split(
        "UPDATE {schema}.dag", maxsplit=1
    )[0]
    dag_update = sql.split("UPDATE {schema}.dag", maxsplit=1)[1]

    assert "created_on = reset_at" in job_update
    assert "start_after = reset_at" in job_update
    assert "created_on = reset_at" in dag_update
    assert "updated_on = reset_at" in dag_update
