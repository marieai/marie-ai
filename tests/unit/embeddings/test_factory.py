from unittest.mock import MagicMock

import pytest

from marie.embeddings import factory


@pytest.mark.parametrize(
    ("embedding_type", "class_name"),
    [
        ("qwen", "QwenVLEmbeddings"),
        ("JinaEmbeddings", "JinaEmbeddings"),
        ("gemini", "GeminiEmbeddings"),
    ],
)
def test_setup_embeddings_selects_configured_type(
    monkeypatch, embedding_type, class_name
):
    constructor = MagicMock(return_value=object())
    monkeypatch.setattr(factory, class_name, constructor)

    result = factory.setup_embeddings(
        {
            "type": embedding_type,
            "model_name_or_path": "provider/model",
            "dimension": 1024,
            "batch_size": 8,
            "use_gpu": False,
        }
    )

    assert result is constructor.return_value
    constructor.assert_called_once()


def test_setup_embeddings_infers_qwen_for_legacy_config(monkeypatch):
    constructor = MagicMock(return_value=object())
    monkeypatch.setattr(factory, "QwenVLEmbeddings", constructor)

    factory.setup_embeddings(
        {
            "model_name_or_path": "hf://Qwen/Qwen3-VL-Embedding-2B",
            "embedding_dim": 512,
        }
    )

    assert constructor.call_args.kwargs["truncate_dim"] == 512


def test_setup_embeddings_passes_jina_v4_multimodal_configuration(monkeypatch):
    constructor = MagicMock(return_value=object())
    monkeypatch.setattr(factory, "JinaEmbeddings", constructor)

    factory.setup_embeddings(
        {
            "type": "jina",
            "model_name_or_path": "hf://jinaai/jina-embeddings-v4",
            "dimension": 1024,
            "task": "retrieval",
            "batch_size": 8,
            "use_gpu": False,
        }
    )

    constructor.assert_called_once_with(
        model_name_or_path="hf://jinaai/jina-embeddings-v4",
        task="retrieval",
        truncate_dim=1024,
        use_gpu=False,
        batch_size=8,
    )


def test_setup_embeddings_rejects_unknown_type():
    with pytest.raises(ValueError, match="Invalid embedding type"):
        factory.setup_embeddings(
            {"type": "unknown", "model_name_or_path": "provider/model"}
        )


def test_setup_embeddings_requires_model_name():
    with pytest.raises(ValueError, match="model_name_or_path"):
        factory.setup_embeddings({"type": "qwen"})
