from pathlib import Path

import yaml

from marie.executor.extract.util import layout_config

REPO_ROOT = Path(__file__).resolve().parents[2]
MOCK_SCHEDULER_CONFIG = (
    REPO_ROOT / "config/service/mock/marie-mock-scheduler-test.yml"
)
MOCK_EXTRACT_CONFIG = REPO_ROOT / "config/extract"


def test_mock_scheduler_config_has_annotator_llm_executor() -> None:
    config = yaml.safe_load(MOCK_SCHEDULER_CONFIG.read_text())
    executors = config["executors"]

    annotator = next(
        executor for executor in executors if executor["name"] == "annotator_llm"
    )

    assert annotator["uses"]["jtype"] == "DocumentAnnotatorLLMExecutor"
    assert annotator["uses"]["metas"]["py_modules"] == ["marie.executor.extract"]
    assert annotator["uses"]["with"]["storage"]["psql"]["enabled"] is True
    assert annotator["uses"]["with"]["storage"]["s3"]["enabled"] is True
    assert annotator["uses"]["with"]["llm_tracking"]["worker"]["enabled"] is False
    assert annotator["env"]["OPENAI_API_BASE"] == "${{ ENV.OPENAI_API_BASE }}"
    assert annotator["env"]["OPENAI_API_KEY"] == "${{ ENV.OPENAI_API_KEY }}"
    assert annotator["env"]["LLM_QUEUE_POOL_ID"] == "document-small"


def test_mock_scheduler_config_exposes_annotator_llm_endpoint() -> None:
    config = yaml.safe_load(MOCK_SCHEDULER_CONFIG.read_text())

    expose_endpoints = config["with"]["expose_endpoints"]

    assert "/document/process" in expose_endpoints
    assert expose_endpoints["/annotator/llm"] == {
        "methods": ["POST"],
        "summary": "Mock LLM annotation",
        "tags": ["annotator"],
    }


def test_mock_scheduler_config_registers_mock_query_planners() -> None:
    config = yaml.safe_load(MOCK_SCHEDULER_CONFIG.read_text())

    planners = config["with"]["job_scheduler_kwargs"]["query_planners"]["planners"]

    assert {
        "name": "mock_planners",
        "py_module": "tests.integration.scheduler.mock_query_plans",
    } in planners


def test_mock_extract_layout_resolves_real_llm_annotator_config() -> None:
    config = layout_config(str(MOCK_EXTRACT_CONFIG), "mock-llm")

    annotator = config.annotators["mock-llm"]

    assert annotator["annotator_type"] == "llm"
    assert annotator["model_config"]["model_name"] == "gpt-5.2-mock"
    assert annotator["model_config"]["prompt_path"] == "./mock-llm.j2"
    assert "extract" in (
        MOCK_EXTRACT_CONFIG / "TID-mock-llm/annotator/mock-llm.j2"
    ).read_text()
