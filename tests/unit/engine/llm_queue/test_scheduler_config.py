from pathlib import Path

import pytest

from marie.engine.llm_queue.config import DEFAULT_LLM_QUEUE_POOL_ID
from marie.engine.llm_queue.scheduler_config import (
    scheduler_config_from_mapping,
)
from marie.serve.runtimes.gateway.marie.llm_scheduler_config import (
    DEFAULT_FABRIC_CONFIG_TABLE,
    DEFAULT_POOL_TABLE,
    PostgresSchedulerConfigRepository,
)


def test_scheduler_schema_defines_runtime_config_tables():
    root = Path(__file__).resolve().parents[4]
    schema_sql = root.joinpath(
        "config",
        "psql",
        "schema",
        "066_llm_queue_scheduler.sql",
    ).read_text()

    assert (
        f"CREATE TABLE IF NOT EXISTS {{schema}}.{DEFAULT_FABRIC_CONFIG_TABLE}"
        in schema_sql
    )
    assert f"CREATE TABLE IF NOT EXISTS {{schema}}.{DEFAULT_POOL_TABLE}" in schema_sql
    assert "CHECK (policy IN ('fifo', 'drr'))" in schema_sql
    assert "total_concurrent_dispatch INT NOT NULL DEFAULT 0" in schema_sql
    assert "endpoint_url TEXT" in schema_sql
    assert "sort_order INT NOT NULL DEFAULT 100" in schema_sql


def test_postgres_scheduler_config_repository_rejects_unsafe_schema_name():
    with pytest.raises(ValueError, match="Invalid LLM queue scheduler schema"):
        PostgresSchedulerConfigRepository({"schema": "marie_scheduler; DROP SCHEMA"})


def test_drr_scheduler_config_adds_default_catch_all_lane():
    config = scheduler_config_from_mapping(
        {
            "policy": "drr",
            "lanes": [{"pool_id": "interactive", "quantum": 8}],
        },
        default_total_concurrent_dispatch=2,
    )

    assert [lane.pool_id for lane in config.lanes] == [
        "interactive",
        DEFAULT_LLM_QUEUE_POOL_ID,
    ]
    assert config.lanes[-1].display_name == "Default"
    assert config.lanes[-1].enabled is True
