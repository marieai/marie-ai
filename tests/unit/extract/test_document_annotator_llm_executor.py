from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from marie.engine.exceptions import BatchExecutionError
from marie.engine.llm_queue.result_types import BatchResult
from omegaconf import OmegaConf

from marie.executor.extract import document_annotator_executor as annotator_module
from marie.executor.extract import (
    document_annotator_llm_executor as llm_executor_module,
)
from marie.executor.extract import (
    document_annotator_table_llm_executor as table_llm_executor_module,
)
from marie.executor.extract.document_annotator_executor import (
    DocumentAnnotatorExecutor,
)
from marie.executor.extract.document_annotator_llm_executor import (
    DocumentAnnotatorLLMExecutor,
)
from marie.executor.extract.document_annotator_table_llm_executor import (
    DocumentAnnotatorTableLLMExecutor,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("true", {"llm_dispatch": {"enabled": True, "mode": "queued-dispatch"}}),
        ("false", {"llm_dispatch": {"enabled": False, "mode": "direct-batch"}}),
    ],
)
@pytest.mark.parametrize(
    "executor_type",
    [DocumentAnnotatorLLMExecutor, DocumentAnnotatorTableLLMExecutor],
)
def test_deployment_status_details_report_llm_mode(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
    expected: dict[str, object],
    executor_type: type,
) -> None:
    monkeypatch.setenv("LLM_QUEUE_ENABLED", value)
    executor = object.__new__(executor_type)

    assert executor.deployment_status_details() == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("true", "LLM request submission mode: queued-dispatch (LLM_QUEUE_ENABLED=true)"),
        ("false", "LLM request submission mode: direct-batch (LLM_QUEUE_ENABLED=false)"),
    ],
)
@pytest.mark.parametrize(
    ("executor_module", "executor_type"),
    [
        (llm_executor_module, DocumentAnnotatorLLMExecutor),
        (table_llm_executor_module, DocumentAnnotatorTableLLMExecutor),
    ],
)
def test_llm_executor_logs_request_submission_mode_at_startup(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
    expected: str,
    executor_module,
    executor_type: type,
) -> None:
    runtime_logger = MagicMock()

    def initialize_base(executor, **_kwargs) -> None:
        executor.metas = SimpleNamespace(name="annotator_llm")

    monkeypatch.setenv("LLM_QUEUE_ENABLED", value)
    monkeypatch.setattr(DocumentAnnotatorExecutor, "__init__", initialize_base)
    monkeypatch.setattr(
        executor_module,
        "MarieLogger",
        lambda *_args, **_kwargs: SimpleNamespace(logger=runtime_logger),
    )

    executor_type()

    runtime_logger.info.assert_called_once_with(expected)


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
async def test_annotation_restores_agent_output_before_execution(monkeypatch, tmp_path):
    prepare_kwargs = {}

    class Annotator:
        def __init__(self, **_kwargs):
            self.requires_frames = True

        async def aannotate(self, _document, _frames):
            return None

    executor = object.__new__(DocumentAnnotatorLLMExecutor)
    executor.logger = MagicMock()
    executor.root_config_dir = str(tmp_path)
    executor.runtime_info = {"worker": "test"}
    executor.show_error = True
    executor._setup_request = lambda *_args, **_kwargs: None
    executor._record_annotation_assets = lambda **_kwargs: None

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

    def prepare_asset_directory(**kwargs):
        prepare_kwargs.update(kwargs)
        return (
            str(tmp_path),
            str(tmp_path / "frames"),
            str(tmp_path / "metadata.json"),
        )

    monkeypatch.setattr(
        annotator_module, "prepare_asset_directory", prepare_asset_directory
    )
    monkeypatch.setattr(annotator_module, "load_json_file", lambda _path: {"ocr": {}})
    monkeypatch.setattr(
        annotator_module.MetaReader,
        "from_data",
        lambda **_kwargs: SimpleNamespace(page_count=1),
    )
    monkeypatch.setattr(
        annotator_module, "get_payload_features", lambda *_args, **_kwargs: []
    )
    monkeypatch.setattr(annotator_module, "MARIE_KERNEL_AVAILABLE", False)
    monkeypatch.setattr(annotator_module, "store_assets", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(annotator_module, "torch_gc", lambda: None)

    response = await executor._process_annotation_request(
        [source_doc],
        {
            "job_id": "job-1",
            "ref_id": "document",
            "ref_type": "lbxid",
            "payload": {
                "op_params": {"key": "claims", "layout": "122418"},
            },
        },
        Annotator,
    )

    assert response["status"] == "success"
    assert prepare_kwargs["restore_dirs"] == ["agent-output"]


@pytest.mark.asyncio
async def test_annotation_error_reports_batch_root_cause(monkeypatch, tmp_path):
    captured_annotator_kwargs = {}

    class ContextWindowExceededError(Exception):
        pass

    root_error = ContextWindowExceededError("maximum context length is 42768 tokens")
    batch_error = BatchExecutionError(
        request_id="request-1",
        failed_results=[BatchResult("request-1_task_4", None, root_error)],
        total=8,
    )

    class FailingAnnotator:
        def __init__(self, **kwargs):
            captured_annotator_kwargs.update(kwargs)

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
    monkeypatch.setattr(annotator_module, "load_json_file", lambda _path: {"ocr": {}})
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
            "payload": {
                "pool_id": "document-small",
                "op_params": {"key": "claims", "layout": "122418"},
            },
        },
        FailingAnnotator,
    )

    assert "pool_id" not in captured_annotator_kwargs
    assert response["status"] == "error"
    assert response["error"] == (str(batch_error),)
    assert response["error_details"]["type"] == "ContextWindowExceededError"
    assert response["error_details"]["message"] == str(batch_error)
    assert set(response["error_details"]) == {
        "type",
        "message",
        "filename",
        "name",
        "line_no",
    }
