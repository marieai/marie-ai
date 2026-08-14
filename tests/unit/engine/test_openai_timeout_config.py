from unittest import mock

import pytest
from marie.engine.batch_processor import BatchProcessor
from marie.engine.openai_compat import build_async_openai_client


def test_openai_client_uses_default_read_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LLM_HTTP_READ_TIMEOUT_S", raising=False)

    with (
        mock.patch("httpx.AsyncClient") as http_client,
        mock.patch("openai.AsyncOpenAI"),
    ):
        build_async_openai_client("test-key")

    assert http_client.call_args.kwargs["timeout"].read == 600.0


def test_openai_client_uses_configured_read_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_HTTP_READ_TIMEOUT_S", "725")

    with (
        mock.patch("httpx.AsyncClient") as http_client,
        mock.patch("openai.AsyncOpenAI") as openai_client,
    ):
        build_async_openai_client("test-key")

    assert http_client.call_args.kwargs["timeout"].read == 725.0
    openai_client.assert_called_once()


def test_openai_client_rejects_non_positive_read_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_HTTP_READ_TIMEOUT_S", "0")

    with pytest.raises(ValueError, match="LLM_HTTP_READ_TIMEOUT_S"):
        build_async_openai_client("test-key")


def test_batch_processor_uses_default_batch_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LLM_BATCH_TIMEOUT_S", raising=False)

    with mock.patch("marie.engine.batch_processor.AsyncOpenAI", new=object):
        processor = BatchProcessor(
            client=object(),
            model_string="test-model",
            logger=mock.MagicMock(),
        )

    assert processor.batch_timeout == 900.0


def test_batch_processor_uses_configured_batch_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_BATCH_TIMEOUT_S", "975")

    with mock.patch("marie.engine.batch_processor.AsyncOpenAI", new=object):
        processor = BatchProcessor(
            client=object(),
            model_string="test-model",
            logger=mock.MagicMock(),
        )

    assert processor.batch_timeout == 975.0


def test_batch_processor_rejects_non_positive_batch_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_BATCH_TIMEOUT_S", "0")

    with (
        mock.patch("marie.engine.batch_processor.AsyncOpenAI", new=object),
        pytest.raises(ValueError, match="batch timeout"),
    ):
        BatchProcessor(
            client=object(),
            model_string="test-model",
            logger=mock.MagicMock(),
        )


def test_explicit_batch_timeout_overrides_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_BATCH_TIMEOUT_S", "975")

    with mock.patch("marie.engine.batch_processor.AsyncOpenAI", new=object):
        processor = BatchProcessor(
            client=object(),
            model_string="test-model",
            logger=mock.MagicMock(),
            batch_timeout=1200.0,
        )

    assert processor.batch_timeout == 1200.0
