import warnings
from traceback import format_exception
from unittest import mock

import pytest
from marie.engine.batch_processor import BatchProcessor
from marie.engine.llm_queue.config import LlmQueueConfig


def test_queue_config_prefers_canonical_url_and_normalizes_it():
    with mock.patch.dict(
        "os.environ",
        {
            "LLM_QUEUE_URL": " redis://canonical:6379/0 ",
            "LLM_QUEUE_VALKEY_URL": "redis://legacy:6379/0",
        },
        clear=True,
    ):
        config = LlmQueueConfig.from_env()

    assert config.queue_url == "redis://canonical:6379/0"
    assert config.valkey_url == "redis://canonical:6379/0"


def test_queue_config_accepts_legacy_url_when_canonical_is_absent():
    with mock.patch.dict(
        "os.environ",
        {"LLM_QUEUE_VALKEY_URL": " redis://legacy:6379/0 "},
        clear=True,
    ):
        config = LlmQueueConfig.from_env()

    assert config.queue_url == "redis://legacy:6379/0"


def test_queue_config_prefers_canonical_explicit_url_over_legacy_explicit_url():
    config = LlmQueueConfig.from_env(
        queue_url="redis://canonical:6379/0",
        valkey_url="redis://legacy:6379/0",
    )

    assert config.queue_url == "redis://canonical:6379/0"


def test_queue_config_prefers_explicit_legacy_url_over_canonical_environment():
    with mock.patch.dict(
        "os.environ",
        {"LLM_QUEUE_URL": "redis://canonical:6379/0"},
        clear=True,
    ):
        config = LlmQueueConfig.from_env(valkey_url="redis://legacy:6379/0")

    assert config.queue_url == "redis://legacy:6379/0"


@pytest.mark.parametrize("value", ["", "  \t "])
def test_queue_config_treats_blank_canonical_url_as_absent(value):
    with mock.patch.dict(
        "os.environ",
        {
            "LLM_QUEUE_URL": value,
            "LLM_QUEUE_VALKEY_URL": "redis://legacy:6379/0",
        },
        clear=True,
    ):
        config = LlmQueueConfig.from_env()

    assert config.queue_url == "redis://legacy:6379/0"


def test_queue_config_rejects_invalid_selected_url_without_disclosing_it():
    secret_url = "http://user:secret-token@invalid.example/queue"

    with pytest.raises(ValueError) as error:
        LlmQueueConfig.from_env(
            queue_url=secret_url,
            valkey_url="redis://valid:6379/0",
        )

    assert "queue_url" in str(error.value)
    assert secret_url not in str(error.value)
    assert "secret-token" not in str(error.value)


def test_queue_config_sanitizes_malformed_authority_parse_errors():
    malformed_url = "redis://user:dummy-secret＠host/0"

    with pytest.raises(ValueError) as error:
        LlmQueueConfig.from_env(queue_url=malformed_url)

    traceback = "".join(
        format_exception(error.type, error.value, error.tb)
    )
    assert "queue_url" in str(error.value)
    assert "dummy-secret" not in str(error.value)
    assert "dummy-secret" not in traceback


def test_queue_config_rejects_invalid_canonical_environment_without_fallback():
    with mock.patch.dict(
        "os.environ",
        {
            "LLM_QUEUE_URL": "http://not-redis.example/queue",
            "LLM_QUEUE_VALKEY_URL": "redis://legacy:6379/0",
        },
        clear=True,
    ):
        with pytest.raises(ValueError, match="LLM_QUEUE_URL"):
            LlmQueueConfig.from_env()


def test_queue_config_legacy_constructor_alias_hides_url_from_repr():
    secret_url = "redis://user:password@queue:6379/0"

    config = LlmQueueConfig(enabled=True, valkey_url=secret_url)

    assert config.queue_url == secret_url
    assert config.valkey_url == secret_url
    assert secret_url not in repr(config)


def test_queue_config_warns_once_when_resolved_url_is_forwarded_to_consumer():
    with mock.patch.dict(
        "os.environ",
        {
            "LLM_QUEUE_URL": "redis://canonical:6379/0",
            "LLM_QUEUE_VALKEY_URL": "redis://legacy:6379/0",
        },
        clear=True,
    ):
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            producer_config = LlmQueueConfig.from_env()
            with mock.patch(
                "marie.engine.batch_processor.AsyncOpenAI",
                new=object,
            ):
                consumer = BatchProcessor(
                    client=object(),
                    model_string="test-model",
                    logger=mock.Mock(),
                    queue_enabled=True,
                    queue_url=producer_config.queue_url,
                )

    assert consumer._queue_config.queue_url == "redis://canonical:6379/0"
    assert len(recorded) == 1
    assert "LLM_QUEUE_URL" in str(recorded[0].message)
    assert "redis://" not in str(recorded[0].message)
