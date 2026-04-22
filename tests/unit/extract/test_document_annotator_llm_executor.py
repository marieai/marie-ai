from unittest.mock import MagicMock

import pytest

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
