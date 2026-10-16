from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from marie.extract.annotators.context_provider import ProcessingUnit

_received = None


def _response_handler(**kwargs):
    global _received
    _received = kwargs
    return {"validated": kwargs["response"]}


class _ContextManager:
    def __init__(self, units):
        self.units = units
        self.providers = [self]

    def has_providers(self):
        return True

    def get_processing_units(self, document):
        return self.units

    def collect_for_unit(self, document, unit):
        return {}

    def get_variables(self, document, page_number, unit=None):
        return {}


class _FakeEngine:
    model_string = "test-model"
    system_prompt = "engine default system"

    def __init__(self, response="not generic json"):
        self.response = response
        self.batch_processor = SimpleNamespace(
            max_concurrency=1, uses_queued_dispatch=False
        )

    def __call__(self, content, **kwargs):
        from marie.engine.completion_observer import CompletionAttempt

        task_id = "request_task_0"
        kwargs["attempt_observer"](
            CompletionAttempt(
                task_id=task_id,
                attempt_number=1,
                raw_response=self.response,
                finish_reason="stop",
                response_model="observed-model",
                input_tokens=7,
                output_tokens=3,
                elapsed_ms=12.5,
                error=None,
            )
        )
        kwargs["on_result"](task_id, self.response)
        return [self.response]


class _FailingEngine(_FakeEngine):
    def __call__(self, content, **kwargs):
        from marie.engine.completion_observer import CompletionAttempt
        from marie.engine.exceptions import BatchExecutionError
        from marie.engine.llm_queue.result_types import BatchResult

        task_id = "request_task_0"
        kwargs["attempt_observer"](
            CompletionAttempt(
                task_id=task_id,
                attempt_number=1,
                raw_response=None,
                finish_reason=None,
                response_model=None,
                input_tokens=None,
                output_tokens=None,
                elapsed_ms=4.0,
                error="RuntimeError: transport failed",
            )
        )
        raise BatchExecutionError(
            request_id="request",
            failed_results=[
                BatchResult(task_id, None, RuntimeError("transport failed"))
            ],
            total=1,
        )


class _ObserverFailureWrappingEngine(_FakeEngine):
    def __call__(self, content, **kwargs):
        from marie.engine.completion_observer import CompletionAttempt
        from marie.engine.exceptions import BatchExecutionError
        from marie.engine.llm_queue.result_types import BatchResult

        task_id = "request_task_0"
        try:
            kwargs["attempt_observer"](
                CompletionAttempt(
                    task_id=task_id,
                    attempt_number=1,
                    raw_response=self.response,
                    finish_reason="stop",
                    response_model="observed-model",
                    input_tokens=7,
                    output_tokens=3,
                    elapsed_ms=12.5,
                    error=None,
                )
            )
        except Exception as error:
            raise BatchExecutionError(
                request_id="request",
                failed_results=[BatchResult(task_id, None, error)],
                total=1,
            ) from error
        raise AssertionError("observer failure was expected")


def _make_annotator(tmp_path, **config_overrides):
    work_dir = tmp_path / "work"
    frames_dir = work_dir / "frames"
    prompt_dir = tmp_path / "prompts"
    frames_dir.mkdir(parents=True)
    prompt_dir.mkdir()
    (prompt_dir / "test.j2").write_text("Extract OCR_DATA", encoding="utf-8")

    model_config = {
        "model_name": "test-model",
        "prompt_path": "./test.j2",
        "expect_output": "json",
        "multimodal": True,
        "refine_passes": 0,
    }
    model_config.update(config_overrides.pop("model_config", {}))
    annotator_conf = {
        "name": "test-ann",
        "annotator_type": "llm",
        "mode": "per-page",
        "model_config": model_config,
        "response_handler": (
            "tests.unit.extract.annotators.test_response_handler._response_handler"
        ),
    }
    annotator_conf.update(config_overrides)

    with (
        patch(
            "marie.extract.annotators.llm_annotator.route_llm_engine",
            return_value=MagicMock(),
        ),
        patch(
            "marie.extract.annotators.llm_annotator.ContextProviderManager",
            return_value=MagicMock(has_providers=lambda: False),
        ),
    ):
        from marie.extract.annotators.llm_annotator import LLMAnnotator

        return LLMAnnotator(
            working_dir=str(work_dir),
            annotator_conf=annotator_conf,
            layout_conf={"layout_id": "999"},
            prompt_dir=str(prompt_dir),
        )


def _document():
    return SimpleNamespace(
        page_count=1,
        source_metadata={
            "pages": 1,
            "ocr": [
                {
                    "meta": {
                        "page": 0,
                        "imageSize": {"width": 32, "height": 32},
                    },
                    "words": [],
                    "lines": [],
                }
            ],
        },
    )


def test_response_handler_rejects_conflicting_output_transform(tmp_path):
    with pytest.raises(ValueError, match="response_handler.*output_transform"):
        _make_annotator(tmp_path, output_transform="example.transform")


def test_response_handler_rejects_refinement(tmp_path):
    with pytest.raises(ValueError, match="response_handler.*refinement"):
        _make_annotator(tmp_path, model_config={"refine_passes": 1})


def test_response_handler_requires_json_output(tmp_path):
    with pytest.raises(ValueError, match="response_handler.*JSON"):
        _make_annotator(tmp_path, model_config={"expect_output": "markdown"})


@pytest.mark.asyncio
async def test_response_handler_receives_actual_unit_and_persists_validated_result(
    tmp_path,
):
    global _received
    _received = None
    annotator = _make_annotator(tmp_path)
    unit = ProcessingUnit(page_number=1, data={"source_page": 0})
    annotator.context_manager = _ContextManager([unit])
    annotator.engine = _FakeEngine()
    image = Image.new("RGB", (32, 32), color=(10, 20, 30))
    frame_path = tmp_path / "work" / "frames" / "frame_0001.png"
    image.save(frame_path)

    document = _document()
    await annotator.aannotate(document, [image])

    output_dir = tmp_path / "work" / "agent-output" / "test-ann"
    assert json.loads((output_dir / "frame_0001.json").read_text()) == {
        "validated": "not generic json"
    }
    assert (output_dir / "_SUCCESS.yaml").is_file()
    assert _received["prompt"] == "Extract "
    assert _received["prompt_template"] == "Extract OCR_DATA"
    assert _received["image_path"] == str(frame_path)
    assert _received["image"].size == (32, 32)
    assert _received["unit"] == unit
    assert _received["document"] is document
    assert _received["run_context"] is annotator.run_context
    assert _received["annotator_conf"] == annotator.annotator_conf
    assert _received["output_path"] == str(output_dir)
    assert _received["error"] is None
    assert len(_received["attempts"]) == 1

    attempt_dir = output_dir / "attempts" / "request_task_0"
    request = json.loads((attempt_dir / "request.json").read_text())
    attempt = json.loads((attempt_dir / "attempt-0001.json").read_text())
    assert request["prompt"] == "Extract "
    assert request["system_prompt"] == "engine default system"
    assert request["image"]["width"] == 32
    assert request["image"]["height"] == 32
    assert request["image"]["byte_count"] == (attempt_dir / "image.png").stat().st_size
    assert len(request["image"]["sha256"]) == 64
    assert attempt["raw_response"] == "not generic json"
    assert attempt["input_tokens"] == 7
    assert attempt["cost_usd"] is None


@pytest.mark.asyncio
async def test_response_handler_receives_raw_response_from_real_engine_lifecycle(
    tmp_path,
):
    import os
    import unittest.mock as mock

    import marie.engine.batch_processor as batch_processor_module
    import marie.engine.openai_engine as openai_engine_module
    from marie.engine.openai_engine import OpenAIEngine

    global _received
    _received = None
    annotator = _make_annotator(tmp_path)
    unit = ProcessingUnit(page_number=1)
    annotator.context_manager = _ContextManager([unit])
    image = Image.new("RGB", (32, 32))
    image.save(tmp_path / "work" / "frames" / "frame_0001.png")
    raw_response = '  <think>private reasoning</think>\n{"candidate": true}  '
    completion = {
        "model": "observed-model",
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": raw_response},
            }
        ],
        "usage": {"prompt_tokens": 17, "completion_tokens": 5},
    }

    with (
        mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
        mock.patch.object(
            openai_engine_module,
            "build_async_openai_client",
            return_value=MagicMock(),
        ),
        mock.patch.object(batch_processor_module, "AsyncOpenAI", new=object),
    ):
        engine = OpenAIEngine(
            model_name="test-model",
            is_multimodal=True,
            base_url="http://test-backend:8000",
            queue_enabled=False,
        )
    annotator.engine = engine

    try:
        with (
            mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
            mock.patch.object(
                batch_processor_module,
                "execute_completion_call",
                new=mock.AsyncMock(return_value=completion),
            ),
        ):
            await annotator.aannotate(_document(), [image])
    finally:
        engine.close()

    output_dir = tmp_path / "work" / "agent-output" / "test-ann"
    assert _received["response"] == raw_response
    assert json.loads((output_dir / "frame_0001.json").read_text()) == {
        "validated": raw_response
    }


def test_response_handler_sync_path_persists_validated_result(tmp_path):
    global _received
    _received = None
    annotator = _make_annotator(tmp_path)
    unit = ProcessingUnit(page_number=1)
    annotator.context_manager = _ContextManager([unit])
    annotator.engine = _FakeEngine()
    image = Image.new("RGB", (32, 32))
    image.save(tmp_path / "work" / "frames" / "frame_0001.png")

    annotator.annotate(_document(), [image])

    output_dir = tmp_path / "work" / "agent-output" / "test-ann"
    assert json.loads((output_dir / "frame_0001.json").read_text()) == {
        "validated": "not generic json"
    }
    assert _received["unit"] == unit
    assert (output_dir / "_SUCCESS.yaml").is_file()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "existing_path",
    ["stale.json", "attempts/request_task_0/request.json"],
)
async def test_response_handler_rejects_nonfresh_output_without_deleting_it(
    tmp_path,
    existing_path,
):
    annotator = _make_annotator(tmp_path)
    annotator.context_manager = _ContextManager([ProcessingUnit(page_number=1)])
    annotator.engine = MagicMock(side_effect=AssertionError("must not infer"))
    annotator.purge_annotators = [annotator.name]
    image = Image.new("RGB", (32, 32))
    image.save(tmp_path / "work" / "frames" / "frame_0001.png")
    artifact_path = tmp_path / "work" / "agent-output" / "test-ann" / existing_path
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text("prior artifact", encoding="utf-8")

    with pytest.raises(ValueError, match="fresh output directory"):
        await annotator.aannotate(_document(), [image])

    assert artifact_path.read_text(encoding="utf-8") == "prior artifact"


@pytest.mark.asyncio
async def test_response_handler_failure_keeps_attempts_without_result_or_success(
    tmp_path,
):
    annotator = _make_annotator(tmp_path)
    annotator.context_manager = _ContextManager([ProcessingUnit(page_number=1)])
    annotator.engine = _FakeEngine(response="bad candidate")
    image = Image.new("RGB", (32, 32))
    image.save(tmp_path / "work" / "frames" / "frame_0001.png")

    def failing_handler(**kwargs):
        raise ValueError("candidate validation failed")

    annotator.response_handler = failing_handler
    with pytest.raises(ValueError, match="candidate validation failed"):
        await annotator.aannotate(_document(), [image])

    output_dir = tmp_path / "work" / "agent-output" / "test-ann"
    assert not (output_dir / "frame_0001.json").exists()
    assert not (output_dir / "_SUCCESS.yaml").exists()
    assert (output_dir / "attempts" / "request_task_0" / "attempt-0001.json").is_file()


@pytest.mark.asyncio
async def test_response_handler_preserves_attempt_persistence_failure(
    tmp_path,
):
    from marie.engine.exceptions import BatchExecutionError

    import marie.extract.annotators.util as annotator_util

    global _received
    _received = None
    annotator = _make_annotator(tmp_path)
    annotator.context_manager = _ContextManager([ProcessingUnit(page_number=1)])
    annotator.engine = _ObserverFailureWrappingEngine(response="raw candidate")
    image = Image.new("RGB", (32, 32))
    image.save(tmp_path / "work" / "frames" / "frame_0001.png")
    persistence_error = OSError("attempt persistence failed")
    failure_artifact = (
        tmp_path / "work" / "agent-output" / "test-ann" / "failed" / "failure.txt"
    )

    def failure_handler(**kwargs):
        global _received
        _received = kwargs
        failure_artifact.parent.mkdir(parents=True, exist_ok=True)
        failure_artifact.write_text("failure recorded", encoding="utf-8")
        raise ValueError("failure artifact handler failed after write")

    annotator.response_handler = failure_handler
    with (
        patch.object(
            annotator_util,
            "_persist_attempt_evidence",
            side_effect=persistence_error,
        ),
        pytest.raises(BatchExecutionError) as exc_info,
    ):
        await annotator.aannotate(_document(), [image])

    assert exc_info.value.primary_error is persistence_error
    assert isinstance(exc_info.value.__cause__, ValueError)
    assert _received["error"] is persistence_error
    assert _received["attempts"] == ()
    assert failure_artifact.read_text(encoding="utf-8") == "failure recorded"


@pytest.mark.asyncio
async def test_response_handler_receives_final_transport_error(tmp_path):
    global _received
    _received = None
    annotator = _make_annotator(tmp_path)
    annotator.context_manager = _ContextManager([ProcessingUnit(page_number=1)])
    annotator.engine = _FailingEngine()
    image = Image.new("RGB", (32, 32))
    image.save(tmp_path / "work" / "frames" / "frame_0001.png")

    from marie.engine.exceptions import BatchExecutionError

    with pytest.raises(BatchExecutionError):
        await annotator.aannotate(_document(), [image])

    output_dir = tmp_path / "work" / "agent-output" / "test-ann"
    assert _received["response"] is None
    assert str(_received["error"]) == "transport failed"
    assert len(_received["attempts"]) == 1
    assert not (output_dir / "frame_0001.json").exists()
    assert not (output_dir / "_SUCCESS.yaml").exists()


@pytest.mark.asyncio
async def test_response_handler_rejects_empty_units_before_inference(tmp_path):
    annotator = _make_annotator(tmp_path)
    annotator.context_manager = _ContextManager([])
    annotator.engine = MagicMock(side_effect=AssertionError("must not infer"))
    image = Image.new("RGB", (8, 8))
    image.save(tmp_path / "work" / "frames" / "frame_0001.png")

    with pytest.raises(ValueError, match="at least one processing unit"):
        await annotator.aannotate(_document(), [image])


@pytest.mark.asyncio
async def test_response_handler_rejects_missing_unit_frame_before_inference(tmp_path):
    annotator = _make_annotator(tmp_path)
    annotator.context_manager = _ContextManager([ProcessingUnit(page_number=2)])
    annotator.engine = MagicMock(side_effect=AssertionError("must not infer"))
    image = Image.new("RGB", (8, 8))
    image.save(tmp_path / "work" / "frames" / "frame_0001.png")

    with pytest.raises(ValueError, match="No frame found for processing unit page 2"):
        await annotator.aannotate(_document(), [image])


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_response_handler_uses_disk_frames_after_runtime_frames_are_released(
    tmp_path, asynchronous
):
    global _received
    _received = None
    annotator = _make_annotator(tmp_path)
    assert annotator.requires_frames is False
    annotator.context_manager = _ContextManager([ProcessingUnit(page_number=1)])
    annotator.engine = _FakeEngine()
    Image.new("RGB", (32, 32)).save(tmp_path / "work" / "frames" / "frame_0001.png")

    if asynchronous:
        await annotator.aannotate(_document(), [])
    else:
        annotator.annotate(_document(), [])

    output_dir = tmp_path / "work" / "agent-output" / "test-ann"
    assert _received["image"].size == (32, 32)
    assert json.loads((output_dir / "frame_0001.json").read_text()) == {
        "validated": "not generic json"
    }
    assert (output_dir / "_SUCCESS.yaml").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_response_handler_rejects_missing_disk_frames(tmp_path, asynchronous):
    annotator = _make_annotator(tmp_path)
    annotator.context_manager = _ContextManager([ProcessingUnit(page_number=1)])
    annotator.engine = MagicMock(side_effect=AssertionError("must not infer"))
    image = Image.new("RGB", (8, 8))

    with pytest.raises(ValueError, match="requires source frames"):
        if asynchronous:
            await annotator.aannotate(_document(), [image])
        else:
            annotator.annotate(_document(), [image])

    annotator.engine.assert_not_called()
    assert not (
        tmp_path / "work" / "agent-output" / "test-ann" / "_SUCCESS.yaml"
    ).exists()
