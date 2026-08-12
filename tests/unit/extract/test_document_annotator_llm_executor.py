from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from marie.engine.exceptions import BatchExecutionError
from marie.engine.llm_queue.result_types import BatchResult
from omegaconf import OmegaConf

from marie.executor.extract import document_annotator_executor as annotator_module
from marie.executor.extract.document_annotator_llm_executor import (
    DocumentAnnotatorLLMExecutor,
)


@pytest.mark.asyncio
async def test_annotator_llm_returns_process_annotation_result():
    executor = object.__new__(DocumentAnnotatorLLMExecutor)
    executor.logger = MagicMock()

    expected = {
        "status": "error",
        "runtime_info": {"worker": "test"},
        "error": "batch failed",
    }

    async def fake_process(*args, **kwargs):
        return expected

    executor._process_annotation_request = fake_process

    result = await executor.annotator_llm([], {"job_id": "job-1"})

    assert result is expected


@pytest.mark.asyncio
async def test_annotation_error_reports_batch_root_cause(monkeypatch, tmp_path):
    class ContextWindowExceededError(Exception):
        pass

    root_error = ContextWindowExceededError(
        "maximum context length is 42768 tokens"
    )
    batch_error = BatchExecutionError(
        request_id="request-1",
        failed_results=[BatchResult("request-1_task_4", None, root_error)],
        total=8,
    )

    class FailingAnnotator:
        def __init__(self, **_kwargs):
            pass

        async def aannotate(self, _document, _frames):
            raise batch_error

    executor = object.__new__(DocumentAnnotatorLLMExecutor)
    executor.logger = MagicMock()
    executor.root_config_dir = str(tmp_path)
    executor.runtime_info = {"worker": "test"}
    executor.show_error = True
    executor._setup_request = lambda *_args, **_kwargs: None

    source_doc = SimpleNamespace(asset_key="s3://bucket/document.tif", pages=None)
    annotators = OmegaConf.create({"claims": {"enabled": True}})
    monkeypatch.setattr(
        annotator_module,
        "layout_config",
        lambda *_args: SimpleNamespace(annotators=annotators),
    )
    monkeypatch.setattr(
        annotator_module,
        "docs_from_asset",
        lambda *_args, **_kwargs: ([source_doc], str(tmp_path / "document.tif")),
    )
    monkeypatch.setattr(annotator_module, "frames_from_docs", lambda _docs: [])
    monkeypatch.setattr(
        annotator_module,
        "prepare_asset_directory",
        lambda **_kwargs: (
            str(tmp_path),
            str(tmp_path / "frames"),
            str(tmp_path / "metadata.json"),
        ),
    )
    monkeypatch.setattr(
        annotator_module, "load_json_file", lambda _path: {"ocr": {}}
    )
    monkeypatch.setattr(
        annotator_module.MetaReader,
        "from_data",
        lambda **_kwargs: SimpleNamespace(page_count=1),
    )
    monkeypatch.setattr(
        annotator_module, "get_payload_features", lambda *_args, **_kwargs: []
    )
    monkeypatch.setattr(annotator_module, "MARIE_KERNEL_AVAILABLE", False)
    monkeypatch.setattr(annotator_module, "torch_gc", lambda: None)

    response = await executor._process_annotation_request(
        [source_doc],
        {
            "job_id": "job-1",
            "ref_id": "document",
            "ref_type": "lbxid",
            "payload": {"op_params": {"key": "claims", "layout": "122418"}},
        },
        FailingAnnotator,
    )

    assert response["status"] == "error"
    assert response["error"] == (str(batch_error),)
    assert response["error_details"] == {
        "type": "ContextWindowExceededError",
        "message": str(batch_error),
    }
