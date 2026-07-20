from unittest.mock import MagicMock

import numpy as np
import torch

from marie.embeddings.jina import jina_embeddings


def _build_jina_v4(monkeypatch, tmp_path):
    model = MagicMock()
    model.to.return_value = model
    model.encode_text.return_value = np.array([[0.1, 0.2]], dtype=np.float32)
    model.encode_image.return_value = np.array([[0.3, 0.4]], dtype=np.float32)

    monkeypatch.setattr(
        jina_embeddings.ModelRegistry,
        "get",
        MagicMock(return_value=str(tmp_path)),
    )
    monkeypatch.setattr(
        jina_embeddings,
        "initialize_device_settings",
        MagicMock(return_value=([torch.device("cpu")], None)),
    )
    monkeypatch.setattr(
        jina_embeddings,
        "_ensure_transformers_rope_compatibility",
        MagicMock(),
    )
    monkeypatch.setattr(
        jina_embeddings.AutoModel,
        "from_pretrained",
        MagicMock(return_value=model),
    )

    embeddings = jina_embeddings.JinaEmbeddings(
        model_name_or_path="hf://jinaai/jina-embeddings-v4",
        use_gpu=False,
        batch_size=2,
        task="retrieval",
        truncate_dim=1024,
    )
    return embeddings, model


def test_jina_v4_uses_query_and_passage_text_adapters(monkeypatch, tmp_path):
    embeddings, model = _build_jina_v4(monkeypatch, tmp_path)

    query = embeddings.embed_text(["invoice"], is_query=True)
    model.encode_text.assert_called_once_with(
        ["invoice"],
        task="retrieval",
        max_length=32768,
        batch_size=2,
        return_numpy=True,
        truncate_dim=1024,
        prompt_name="query",
    )
    assert query.tolist() == [[0.10000000149011612, 0.20000000298023224]]

    model.encode_text.reset_mock()
    embeddings.embed_text(["document"], is_query=False)
    assert model.encode_text.call_args.kwargs["prompt_name"] == "passage"


def test_jina_v4_exposes_multimodal_image_embeddings(monkeypatch, tmp_path):
    embeddings, model = _build_jina_v4(monkeypatch, tmp_path)

    result = embeddings.embed_images(["page.png"])

    assert embeddings.supports_image_embeddings is True
    model.encode_image.assert_called_once_with(
        ["page.png"],
        task="retrieval",
        batch_size=2,
        return_numpy=True,
        truncate_dim=1024,
    )
    assert result.tolist() == [[0.30000001192092896, 0.4000000059604645]]
