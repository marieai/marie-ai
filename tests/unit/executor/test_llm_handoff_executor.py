import json

import numpy as np
import pytest
from docarray import DocList

from marie.api.docs import AssetKeyDoc
from marie.executor.mock import llm_handoff_executor as module
from marie.extract.readers.meta_reader.meta_reader import MetaReader
from marie.serve.runtimes.worker.request_handling import WorkerRequestHandler
from marie.storage import StorageManager
from marie.types_core.request.data import DataRequest


@pytest.fixture
def storage(monkeypatch):
    objects = {}
    monkeypatch.setattr(StorageManager, "read", lambda uri: objects[uri])
    monkeypatch.setattr(StorageManager, "exists", lambda uri: uri in objects)

    def write(source, destination, **kwargs):
        objects[destination] = source.read()
        return True

    monkeypatch.setattr(StorageManager, "write", write)
    monkeypatch.setattr(module, "s3_asset_path", lambda *_args: "s3://test/run")
    return objects


@pytest.fixture
def parameters():
    return {
        "dag_id": "dag-1",
        "job_id": "job-1",
        "ref_id": "run",
        "ref_type": "mock_llm_handoff",
    }


@pytest.fixture
def executor():
    return object.__new__(module.LlmHandoffExecutorMock)


async def test_prepare_writes_metadata_usable_by_real_annotator(
    monkeypatch, storage, parameters, executor
) -> None:
    frames = [np.zeros((32, 32, 3), dtype=np.uint8)]
    monkeypatch.setattr(module, "docs_from_asset", lambda *_args: frames)
    monkeypatch.setattr(module, "frames_from_docs", lambda docs: docs)
    docs = DocList[AssetKeyDoc]([AssetKeyDoc(asset_key="s3://test/input.png")])

    result = await executor.prepare(docs, parameters)

    metadata = json.loads(storage["s3://test/run/run.meta.json"])
    document = MetaReader.from_data(
        frames, metadata["ocr"], {"source_metadata": metadata}
    )
    assert document.page_count == 1
    assert result["status"] == "success"
    assert metadata["mock_handoff"]["dag_id"] == "dag-1"


async def test_prepare_rejects_reused_reference(storage, parameters, executor) -> None:
    storage["s3://test/run/run.meta.json"] = b'{"mock_handoff": {"dag_id": "old-dag"}}'
    docs = DocList[AssetKeyDoc]([AssetKeyDoc(asset_key="s3://test/input.png")])
    with pytest.raises(ValueError, match="fresh ref_id"):
        await executor.prepare(docs, parameters)
    assert (
        json.loads(storage["s3://test/run/run.meta.json"])["mock_handoff"]["dag_id"]
        == "old-dag"
    )


@pytest.mark.parametrize('endpoint', ['/mock/llm/prepare', '/mock/llm/read'])
def test_worker_decodes_handoff_asset(endpoint) -> None:
    asset = AssetKeyDoc(asset_key='s3://test/input.png', pages=None)
    outgoing = DataRequest()
    outgoing.data.docs = DocList[AssetKeyDoc]([asset])
    incoming = DataRequest(outgoing.proto.SerializeToString())
    worker = object.__new__(WorkerRequestHandler)
    worker._executor = object.__new__(module.LlmHandoffExecutorMock)
    worker._setup_req_doc_array_cls([incoming], endpoint)
    assert incoming.docs[0].asset_key == 's3://test/input.png'


async def test_read_downloads_annotation_and_persists_receipt(
    storage, parameters, executor
) -> None:
    storage["s3://test/run/run.meta.json"] = json.dumps(
        {"pages": 1, "mock_handoff": {"dag_id": "dag-1"}}
    ).encode()
    storage["s3://test/run/agent-output/mock-llm/_SUCCESS.yaml"] = (
        b"files: [00001.json]\n"
    )
    storage["s3://test/run/agent-output/mock-llm/00001.json"] = (
        b'{"answer": "queued result"}'
    )

    result = await executor.read([], parameters)

    assert result["status"] == "success"
    receipt = json.loads(storage[result["receipt_uri"]])
    assert receipt["dag_id"] == "dag-1"
    assert receipt["annotations"] == [{"answer": "queued result"}]
    assert receipt["pages_read"] == 1


@pytest.mark.parametrize(
    "failure", ["missing", "invalid", "scalar", "stale", "no_marker"]
)
async def test_read_rejects_incomplete_or_stale_output(
    storage, parameters, executor, failure
) -> None:
    storage["s3://test/run/run.meta.json"] = json.dumps(
        {
            "pages": 1,
            "mock_handoff": {"dag_id": "old" if failure == "stale" else "dag-1"},
        }
    ).encode()
    if failure != "no_marker":
        storage["s3://test/run/agent-output/mock-llm/_SUCCESS.yaml"] = (
            b"files: [00001.json]\n"
        )
    if failure != "missing":
        storage["s3://test/run/agent-output/mock-llm/00001.json"] = (
            b"{broken"
            if failure == "invalid"
            else b"null" if failure == "scalar" else b"{}"
        )
    with pytest.raises((KeyError, ValueError)):
        await executor.read([], parameters)
    assert "s3://test/run/handoff-receipt.json" not in storage
