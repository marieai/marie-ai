import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from marie.executor.pipeline import document_llm_pipeline_executor as executor_module
from marie.executor.pipeline.document_llm_pipeline_executor import (
    DocumentLLMPipelineExecutor,
)


def test_frame_loading_failure_runs_request_cleanup(monkeypatch):
    page_error = IndexError("Page index out of range: 2")
    get_frames = Mock(side_effect=page_error)
    torch_gc = Mock()
    remove_mdc = Mock()
    pipeline = SimpleNamespace(
        pipelines_config=[{"pipeline": {"name": "medical"}}],
        execute_frames_pipeline=Mock(),
    )
    executor = SimpleNamespace(logger=Mock(), pipeline=pipeline)

    monkeypatch.setattr(executor_module, "get_frames_from_docs", get_frames)
    monkeypatch.setattr(executor_module, "torch_gc", torch_gc)
    monkeypatch.setattr(executor_module.MDC, "remove", remove_mdc)

    parameters = {
        "job_id": "job-1",
        "ref_id": "document-1",
        "ref_type": "document",
        "queue_id": "queue-1",
        "payload": {
            "features": [{"type": "pipeline", "name": "medical", "pages": [2]}]
        },
    }

    with pytest.raises(IndexError) as raised:
        DocumentLLMPipelineExecutor.run_llm_pipeline(executor, [], parameters)

    assert raised.value is page_error
    get_frames.assert_called_once_with([], [2])
    pipeline.execute_frames_pipeline.assert_not_called()
    torch_gc.assert_called_once_with()
    remove_mdc.assert_called_once_with("request_id")


def test_merge_metadata_preserves_request_fields(tmp_path, monkeypatch):
    from marie.executor.pipeline import document_pipeline_executor as module

    features = [{'type': 'pipeline', 'name': 'medical'}]
    (tmp_path / 'medical').mkdir()
    (tmp_path / 'medical' / 'doc.meta.json').write_text('{}')
    restore = Mock(return_value='s3://test')
    monkeypatch.setattr(module, 'restore_assets', restore)
    monkeypatch.setattr(module, 'store_assets', Mock(return_value=[]))
    monkeypatch.setattr(module, 'get_frames_from_docs', lambda _: [])
    monkeypatch.setattr(module, 'create_working_dir', lambda *a, **kw: str(tmp_path))
    executor = SimpleNamespace(logger=Mock(), runtime_info={})
    params = {
        'job_id': 'job',
        'ref_id': 'doc',
        'ref_type': 'test',
        'payload': {'features': features, 'planner': 'plan'},
    }
    module.PipelineExecutor.merge_metadata(executor, [], params)
    restore.assert_called_once_with(
        'doc', 'test', str(tmp_path), overwrite=True, dirs_to_restore=['medical']
    )
    metadata = json.loads((tmp_path / 'doc.meta.json').read_text())
    assert metadata['features'] == features
    assert metadata['planner'] == 'plan'
    restore.return_value = None
    with pytest.raises(ConnectionError):
        module.PipelineExecutor.merge_metadata(executor, [], params)
