from marie.engine import EngineLM, check_if_multimodal
from marie.engine.completion_contract import CompletionCallParams
from marie.engine.llm_queue.scheduler_config import scheduler_config_from_mapping
from marie.engine.output_parser import parse_json_markdown


def test_public_engine_namespace_is_importable() -> None:
    assert EngineLM.__module__ == "marie.engine.base"
    assert check_if_multimodal("qwen2_5_vl_7b") is True


def test_portable_completion_contract_round_trips() -> None:
    call = CompletionCallParams(
        model="test-model",
        messages=[{"role": "user", "content": "hello"}],
    )

    assert call.model == "test-model"
    assert call.messages[0]["content"] == "hello"


def test_scheduler_mapping_does_not_require_server_storage() -> None:
    config = scheduler_config_from_mapping(
        {"policy": "drr", "lanes": [{"pool_id": "interactive"}]},
        default_total_concurrent_dispatch=2,
    )

    assert config.is_drr is True
    assert [lane.pool_id for lane in config.lanes] == ["interactive", "default"]


def test_output_parser_is_reusable() -> None:
    assert parse_json_markdown('```json\n{"ok": true}\n```') == {"ok": True}
