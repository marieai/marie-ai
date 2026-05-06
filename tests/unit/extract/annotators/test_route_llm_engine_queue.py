from unittest import mock

from marie.extract.annotators import util as annotator_util


def test_route_llm_engine_cache_includes_queue_configuration():
    annotator_util.clear_engine_cache()
    engines = [object(), object()]

    with mock.patch.dict(
        "os.environ",
        {
            "OPENAI_API_KEY": "EMPTY",
            "OPENAI_API_BASE": "http://llm-backend/v1",
            "LLM_QUEUE_ENABLED": "false",
            "LLM_QUEUE_VALKEY_URL": "redis://localhost:6379/0",
            "LLM_QUEUE_POOL_ID": "default",
        },
    ):
        with mock.patch.object(
            annotator_util,
            "OpenAIEngine",
            side_effect=engines,
        ) as engine_cls:
            direct_engine = annotator_util.route_llm_engine("model-a", True)

            with mock.patch.dict(
                "os.environ",
                {
                    "LLM_QUEUE_ENABLED": "true",
                    "LLM_QUEUE_VALKEY_URL": "redis://localhost:6379/0",
                    "LLM_QUEUE_POOL_ID": "default",
                },
            ):
                queued_engine = annotator_util.route_llm_engine("model-a", True)

    assert direct_engine is engines[0]
    assert queued_engine is engines[1]
    assert engine_cls.call_count == 2
    assert engine_cls.call_args_list[0].kwargs["queue_enabled"] is False
    assert engine_cls.call_args_list[1].kwargs["queue_enabled"] is True
    assert (
        engine_cls.call_args_list[1].kwargs["queue_valkey_url"]
        == "redis://localhost:6379/0"
    )
    assert engine_cls.call_args_list[1].kwargs["queue_pool_id"] == "default"

    annotator_util.clear_engine_cache()
