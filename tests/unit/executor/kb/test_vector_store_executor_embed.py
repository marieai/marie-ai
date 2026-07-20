import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from docarray import DocList
from docarray.documents import TextDoc

from marie.api.docs import AssetKeyDoc
from marie.executor.kb.vector_store_executor import (
    VectorStoreExecutor,
    _connection_string_from_storage,
    _has_usable_text,
    _run_param,
)
from marie.storage import StorageManager


def _make_executor() -> VectorStoreExecutor:
    executor = object.__new__(VectorStoreExecutor)
    executor.logger = MagicMock()
    executor._initialized = True
    executor._embeddings = MagicMock()
    executor._embeddings.embed_text.side_effect = lambda texts, is_query=False: [
        np.array([0.1, 0.2]) for _ in texts
    ]
    executor._embeddings.embed_images.side_effect = lambda images: [
        np.array([0.3, 0.4]) for _ in images
    ]
    executor._embeddings.supports_image_embeddings = True
    executor._vector_store = MagicMock()
    executor._vector_store.add_node = AsyncMock(return_value="uuid-1")
    executor._vector_store.add_nodes_batch = AsyncMock(
        side_effect=lambda nodes, **kw: len(nodes)
    )
    executor._batch_size = 4
    return executor


# --- module-level helpers -------------------------------------------------


def test_has_usable_text_true_for_text_doc():
    assert _has_usable_text(TextDoc(id="d1", text="hello")) is True


def test_has_usable_text_false_for_blank_or_missing():
    assert _has_usable_text(TextDoc(id="d1", text="   ")) is False
    assert _has_usable_text(TextDoc(id="d1", text=None)) is False
    assert _has_usable_text(AssetKeyDoc(asset_key="s3://bucket/doc.pdf")) is False


def test_run_param_prefers_nested_run_params_over_top_level():
    parameters = {"chunk_size": 999, "run_params": {"chunk_size": 256}}
    assert _run_param(parameters, "chunk_size", 1024) == 256


def test_run_param_falls_back_to_top_level_then_default():
    assert _run_param({"chunk_size": 512}, "chunk_size", 1024) == 512
    assert _run_param({}, "chunk_size", 1024) == 1024


# --- _resolve_extraction_output --------------------------------------------


def test_resolve_extraction_output_reads_meta_json_and_flattens_pages(monkeypatch):
    executor = _make_executor()
    calls = {}

    monkeypatch.setattr(StorageManager, "ensure_connection", lambda *a, **k: True)

    def fake_exists(uri, **kwargs):
        calls["exists_uri"] = uri
        return True

    def fake_read(uri, **kwargs):
        calls["read_uri"] = uri
        return json.dumps(
            {
                "ocr": [
                    {
                        "lines": [
                            {"text": "page one line one"},
                            {"text": "page one line two"},
                        ]
                    },
                    {"lines": [{"text": "page two line one"}]},
                ]
            }
        ).encode("utf-8")

    monkeypatch.setattr(StorageManager, "exists", fake_exists)
    monkeypatch.setattr(StorageManager, "read", fake_read)

    # ref_id mirrors a KB document's raw S3 key (slash-containing)
    full_text, page_ranges = executor._resolve_extraction_output(
        "tenants/t1/kb-indexes/i1/sources/s1/doc.pdf", "kb_document"
    )

    assert calls["exists_uri"] == calls["read_uri"]
    assert calls["exists_uri"].endswith("/doc.pdf.meta.json")
    assert "page one line one" in full_text
    assert "page two line one" in full_text
    assert len(page_ranges) == 2
    assert page_ranges[0][0] == 0
    assert page_ranges[1][0] == 1


def test_resolve_extraction_output_raises_when_missing(monkeypatch):
    executor = _make_executor()
    monkeypatch.setattr(StorageManager, "ensure_connection", lambda *a, **k: True)
    monkeypatch.setattr(StorageManager, "exists", lambda uri, **k: False)

    with pytest.raises(ValueError, match="No extraction output found"):
        executor._resolve_extraction_output("doc.pdf", "kb_document")


def test_resolve_extraction_output_raises_when_ocr_empty(monkeypatch):
    executor = _make_executor()
    monkeypatch.setattr(StorageManager, "ensure_connection", lambda *a, **k: True)
    monkeypatch.setattr(StorageManager, "exists", lambda uri, **k: True)
    monkeypatch.setattr(
        StorageManager, "read", lambda uri, **k: json.dumps({"ocr": []}).encode("utf-8")
    )

    with pytest.raises(ValueError, match="no OCR results"):
        executor._resolve_extraction_output("doc.pdf", "kb_document")


def test_resolve_page_image_uris_prefers_frames(monkeypatch):
    executor = _make_executor()
    root = "s3://marie/kb_document/doc"
    monkeypatch.setattr(
        executor,
        "_read_extraction_metadata",
        MagicMock(
            return_value=(
                {
                    "assets": [
                        f"{root}/burst/doc_00001.tif",
                        f"{root}/frames/00002.png",
                        f"{root}/frames/00001.png",
                        "s3://other-bucket/frames/untrusted.png",
                    ]
                },
                f"{root}/doc.pdf.meta.json",
            )
        ),
    )

    assert executor._resolve_page_image_uris("doc.pdf", "kb_document") == [
        f"{root}/frames/00001.png",
        f"{root}/frames/00002.png",
    ]


def test_resolve_page_image_uris_requires_page_assets(monkeypatch):
    executor = _make_executor()
    monkeypatch.setattr(
        executor,
        "_read_extraction_metadata",
        MagicMock(return_value=({"assets": []}, "s3://marie/doc.meta.json")),
    )

    with pytest.raises(ValueError, match="has no page image assets"):
        executor._resolve_page_image_uris("doc.pdf", "kb_document")


# --- embed_and_store branch logic ------------------------------------------


@pytest.mark.asyncio
async def test_embed_and_store_resolves_asset_key_doc_via_storage_read(monkeypatch):
    executor = _make_executor()
    monkeypatch.setattr(
        executor,
        "_resolve_extraction_output",
        MagicMock(return_value=("some extracted document text", [(0, 0, 28)])),
    )

    docs = DocList[AssetKeyDoc]([AssetKeyDoc(asset_key="s3://bucket/doc.pdf")])
    parameters = {"source_id": "s1", "ref_id": "doc.pdf", "ref_type": "kb_document"}

    result = await executor.embed_and_store(docs, parameters)

    assert result is docs
    executor._resolve_extraction_output.assert_called_once_with(
        "doc.pdf", "kb_document"
    )
    executor._vector_store.add_nodes_batch.assert_awaited_once()
    _, kwargs = executor._vector_store.add_nodes_batch.call_args
    nodes = kwargs["nodes"]
    assert len(nodes) >= 1
    assert all(n["ref_doc_id"] == "doc.pdf" for n in nodes)
    assert all(n["node_type"] == "text" for n in nodes)
    assert nodes[0]["node_id"] == "doc.pdf_0"
    assert kwargs["source_id"] == "s1"


@pytest.mark.asyncio
async def test_embed_and_store_persists_text_and_page_image_nodes(monkeypatch):
    executor = _make_executor()
    monkeypatch.setattr(
        executor,
        "_resolve_extraction_output",
        MagicMock(return_value=("page one\npage two", [(0, 0, 8), (1, 9, 17)])),
    )
    monkeypatch.setattr(
        executor,
        "_resolve_page_image_uris",
        MagicMock(
            return_value=[
                "s3://marie/kb_document/doc/frames/00001.png",
                "s3://marie/kb_document/doc/frames/00002.png",
            ]
        ),
    )

    def fake_read_to_file(uri, destination, overwrite=False, **kwargs):
        Path(destination).write_bytes(b"image")
        return True

    monkeypatch.setattr(StorageManager, "read_to_file", fake_read_to_file)

    docs = DocList[AssetKeyDoc]([AssetKeyDoc(asset_key="s3://bucket/doc.pdf")])
    parameters = {
        "source_id": "s1",
        "ref_id": "doc.pdf",
        "ref_type": "kb_document",
        "run_params": {"multimodal": True},
    }

    result = await executor.embed_and_store(docs, parameters)

    assert result is docs
    assert executor._vector_store.add_nodes_batch.await_count == 2
    image_call = executor._vector_store.add_nodes_batch.await_args_list[1]
    image_nodes = image_call.kwargs["nodes"]
    assert [node["node_type"] for node in image_nodes] == ["image", "image"]
    assert [node["metadata"]["page"] for node in image_nodes] == [0, 1]
    assert image_nodes[0]["metadata"]["image_url"].endswith("/00001.png")
    assert image_nodes[0]["content"] == "page one"
    executor._embeddings.embed_images.assert_called_once()


@pytest.mark.asyncio
async def test_embed_and_store_rejects_multimodal_for_text_only_model(monkeypatch):
    executor = _make_executor()
    executor._embeddings.supports_image_embeddings = False
    monkeypatch.setattr(
        executor,
        "_resolve_extraction_output",
        MagicMock(return_value=("some extracted document text", [(0, 0, 28)])),
    )
    resolve_images = MagicMock()
    monkeypatch.setattr(executor, "_resolve_page_image_uris", resolve_images)

    docs = DocList[AssetKeyDoc]([AssetKeyDoc(asset_key="s3://bucket/doc.pdf")])
    parameters = {
        "source_id": "s1",
        "ref_id": "doc.pdf",
        "ref_type": "kb_document",
        "multimodal": True,
    }

    with pytest.raises(ValueError, match="text-only"):
        await executor.embed_and_store(docs, parameters)

    resolve_images.assert_not_called()
    executor._vector_store.add_nodes_batch.assert_not_awaited()


@pytest.mark.asyncio
async def test_embed_and_store_fails_when_text_embedding_count_is_incomplete(
    monkeypatch,
):
    executor = _make_executor()
    executor._embeddings.embed_text.side_effect = None
    executor._embeddings.embed_text.return_value = []
    monkeypatch.setattr(
        executor,
        "_resolve_extraction_output",
        MagicMock(return_value=("some extracted document text", [(0, 0, 28)])),
    )

    docs = DocList[AssetKeyDoc]([AssetKeyDoc(asset_key="s3://bucket/doc.pdf")])
    parameters = {
        "source_id": "s1",
        "ref_id": "doc.pdf",
        "ref_type": "kb_document",
    }

    with pytest.raises(RuntimeError, match="0 text embeddings for 1 chunks"):
        await executor.embed_and_store(docs, parameters)

    executor._vector_store.add_nodes_batch.assert_not_awaited()


@pytest.mark.asyncio
async def test_embed_and_store_raises_when_ref_id_missing(monkeypatch):
    executor = _make_executor()
    docs = DocList[AssetKeyDoc]([AssetKeyDoc(asset_key="s3://bucket/doc.pdf")])
    parameters = {"source_id": "s1"}

    with pytest.raises(ValueError, match="ref_id is required"):
        await executor.embed_and_store(docs, parameters)


@pytest.mark.asyncio
async def test_embed_and_store_propagates_missing_extraction_output(monkeypatch):
    executor = _make_executor()
    monkeypatch.setattr(
        executor,
        "_resolve_extraction_output",
        MagicMock(side_effect=ValueError("No extraction output found")),
    )
    docs = DocList[AssetKeyDoc]([AssetKeyDoc(asset_key="s3://bucket/doc.pdf")])
    parameters = {"source_id": "s1", "ref_id": "doc.pdf", "ref_type": "kb_document"}

    with pytest.raises(ValueError, match="No extraction output found"):
        await executor.embed_and_store(docs, parameters)


@pytest.mark.asyncio
async def test_embed_and_store_preserves_direct_text_doc_behavior(monkeypatch):
    executor = _make_executor()
    resolve_spy = MagicMock()
    monkeypatch.setattr(executor, "_resolve_extraction_output", resolve_spy)

    docs = DocList[TextDoc]([TextDoc(id="d1", text="already extracted text")])
    parameters = {"source_id": "s1"}

    result = await executor.embed_and_store(docs, parameters)

    assert result is docs
    resolve_spy.assert_not_called()
    executor._vector_store.add_node.assert_awaited_once()
    _, kwargs = executor._vector_store.add_node.call_args
    assert kwargs["content"] == "already extracted text"
    assert kwargs["source_id"] == "s1"


# --- /embed endpoint -------------------------------------------------------


@pytest.mark.asyncio
async def test_embed_returns_vectors_in_parameters_payload():
    executor = _make_executor()
    executor._embedding_dim = 2

    result = await executor.embed(
        DocList[TextDoc](),
        {"texts": ["invoice", "receipt"], "is_query": True},
    )

    # embeddings converted from numpy arrays to plain lists, dim reported
    assert result["dim"] == 2
    assert result["embeddings"] == [[0.1, 0.2], [0.1, 0.2]]
    _, kwargs = executor._embeddings.embed_text.call_args
    assert kwargs["is_query"] is True
    # JSON-safe (no numpy arrays leak through)
    json.dumps(result)


@pytest.mark.asyncio
async def test_embed_defaults_is_query_false():
    executor = _make_executor()
    executor._embedding_dim = 2

    await executor.embed(DocList[TextDoc](), {"texts": ["invoice"]})

    _, kwargs = executor._embeddings.embed_text.call_args
    assert kwargs["is_query"] is False


@pytest.mark.asyncio
async def test_embed_requires_texts():
    executor = _make_executor()
    with pytest.raises(ValueError, match="texts list is required"):
        await executor.embed(DocList[TextDoc](), {})


def test_connection_string_from_storage_full_psql_block():
    storage = {
        "psql": {
            "provider": "postgresql",
            "hostname": "db.internal",
            "port": 5433,
            "username": "marie",
            "password": "p@ss:word",
            "database": "postgres",
        }
    }
    assert (
        _connection_string_from_storage(storage)
        == "postgresql://marie:p%40ss%3Aword@db.internal:5433/postgres"
    )


def test_connection_string_from_storage_defaults_port_and_database():
    storage = {"psql": {"hostname": "localhost", "username": "u", "password": "p"}}
    assert (
        _connection_string_from_storage(storage)
        == "postgresql://u:p@localhost:5432/postgres"
    )


def test_connection_string_from_storage_absent_or_incomplete():
    assert _connection_string_from_storage(None) is None
    assert _connection_string_from_storage({}) is None
    assert _connection_string_from_storage({"psql": {}}) is None
    assert _connection_string_from_storage({"s3": {"enabled": True}}) is None
