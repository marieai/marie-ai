from pathlib import Path


def test_admission_candidate_function_uses_dag_projection_ordering() -> None:
    project_root = Path(__file__).parents[3]
    sql = (
        project_root / "config/psql/schema/lease/017_admission_candidate_dags.sql"
    ).read_text()
    normalized = " ".join(sql.lower().split())

    assert "create or replace function {schema}.admission_candidate_dags(" in normalized
    assert "language plpgsql stable set jit = off" in normalized
    assert "set plan_cache_mode = force_custom_plan" in normalized
    assert "begin return query select" in normalized
    assert "d.priority desc" in normalized
    assert "coalesce(d.soft_sla, d.hard_sla) asc nulls last" in normalized
    assert "d.created_on, d.id" in normalized
    assert "from {schema}.job ready" in normalized
    assert "ready.state in ('created', 'retry')" in normalized
    assert "ready.start_after <= current_timestamp" in normalized
    assert "from unnest(" in normalized
    assert "coalesce(p_excluded_dag_ids, array[]::uuid[])" in normalized
    assert "as excluded(dag_id)" in normalized
    assert "excluded.dag_id = d.id" in normalized
    assert "d.id = any" not in normalized
    assert "from {schema}.job blocker" in normalized
    assert "blocker.state in ('failed', 'expired', 'cancelled')" in normalized
    assert "max(" not in normalized
    assert "group by" not in normalized
    assert "job_dependencies" not in normalized
    assert "extract(epoch" not in normalized

    order_by = normalized.rsplit("order by", maxsplit=1)[1]
    assert order_by.index("d.priority desc") < order_by.index(
        "coalesce(d.soft_sla, d.hard_sla) asc nulls last"
    )
    assert order_by.index(
        "coalesce(d.soft_sla, d.hard_sla) asc nulls last"
    ) < order_by.index("d.created_on, d.id")

    for planner_signal in (
        "job_level",
        "dag_remaining",
        "estimated_runtime",
        "free_slots",
        "control_flow",
        "_added_at",
    ):
        assert planner_signal not in normalized


def test_base_dag_schema_contains_admission_projection() -> None:
    project_root = Path(__file__).parents[3]
    dag_sql = (project_root / "config/psql/schema/007_dag.sql").read_text()
    history_sql = (project_root / "config/psql/schema/008_dag_history.sql").read_text()
    trigger_sql = (
        project_root / "config/psql/schema/013_dag_history_trigger.sql"
    ).read_text()
    index_sql = (
        project_root / "config/psql/schema/073_scheduler_hot_path_indexes.sql"
    ).read_text()

    for column in (
        "submission_name",
        "project_id",
        "ref_type",
        "ref_id",
        "policy",
    ):
        assert f"{column} TEXT" in dag_sql
        assert f"{column} TEXT" in history_sql

    for column in ("priority", "task_count"):
        definition = f"{column} INTEGER NOT NULL DEFAULT 0"
        assert definition in dag_sql
        assert definition in history_sql
        assert column in trigger_sql

    assert "CREATE INDEX IF NOT EXISTS dag_admission_order_idx" in index_sql
    assert "WHERE state IN ('created', 'active')" in index_sql
    assert not (
        project_root / "config/psql/schema/074_dag_admission_projection.sql"
    ).exists()


def test_monitoring_queries_expose_dag_submission_projection() -> None:
    project_root = Path(__file__).parents[3]
    monitoring_root = project_root / "config/psql/schema/monitoring"
    lifecycle_sql = (monitoring_root / "submission_lifecycle_analysis.sql").read_text()
    failure_sql = (monitoring_root / "job_failure_analysis.sql").read_text()
    outstanding_sql = (monitoring_root / "outstanding_work_analysis.sql").read_text()
    query_guide = (monitoring_root / "QUERY_GUIDE.md").read_text()

    for field in (
        "submission_name",
        "project_id",
        "ref_type",
        "ref_id",
        "priority",
        "task_count",
    ):
        assert f"d.{field}" in lifecycle_sql
        assert f"d.{field}" in failure_sql
        assert f"d.{field}" in outstanding_sql
        assert field in query_guide


def test_submission_lifecycle_query_uses_indexable_scope_arrays() -> None:
    project_root = Path(__file__).parents[3]
    lifecycle_sql = (
        project_root
        / "config/psql/schema/monitoring/submission_lifecycle_analysis.sql"
    ).read_text()
    normalized = " ".join(lifecycle_sql.lower().split())

    for predicate in (
        "ja.job_id = any(s.job_ids)",
        "ja.dag_id = any(s.dag_ids)",
        "kv.key = any(s.worker_keys)",
        "kh.key = any(s.worker_keys)",
    ):
        assert predicate in normalized

    assert "ja.job_id in (select job_id from task_ids)" not in normalized
    assert "ja.dag_id in (select dag_id from resolved_dags)" not in normalized


def test_submission_lifecycle_lookup_indexes_are_managed() -> None:
    project_root = Path(__file__).parents[3]
    index_sql = (
        project_root
        / "config/psql/schema/083_submission_lifecycle_lookup_indexes.sql"
    ).read_text()
    normalized = " ".join(index_sql.lower().split())

    for statement in (
        "create index if not exists dag_history_lifecycle_id_idx "
        "on {schema}.dag_history (id)",
        "create index if not exists job_history_lifecycle_id_idx "
        "on {schema}.job_history (id)",
        "create index if not exists job_history_lifecycle_dag_id_idx "
        "on {schema}.job_history (dag_id)",
    ):
        assert statement in normalized

    repository_source = (
        project_root / "marie/scheduler/repository/async_job_repository.py"
    ).read_text()
    assert "SCHEDULER_SCHEMA_VERSION = 85" in repository_source
