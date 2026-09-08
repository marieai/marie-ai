import asyncio
import gc
import traceback
import weakref
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest
from marie.engine.output_parser import JSONOutputParserError
from omegaconf import OmegaConf
from PIL import Image

from marie.executor.extract import document_annotator_executor as executor_module
from marie.extract.annotators import util


@pytest.mark.parametrize("second_fails", [False, True])
def test_failed_scan_releases_job_state_without_cyclic_gc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, second_fails: bool
) -> None:
    class JobState:
        pass

    for name in ("00001.png", "00002.png", "00003.png"):
        (tmp_path / name).touch()

    state_refs: list[weakref.ReferenceType[JobState]] = []
    processed: list[str] = []
    second_started = asyncio.Event()
    first_failed = asyncio.Event()
    second_finished = asyncio.Event()

    def prepare(*, file_batch: list[str], **kwargs: object) -> Iterator[list]:
        image_state = JobState()
        state_refs.append(weakref.ref(image_state))
        yield [(file_batch[0], image_state)]

    async def process(batch: list, *args: object, **kwargs: object) -> None:
        filename, image_state = batch[0]
        processed.append(filename)
        if filename == "00001.png":
            await second_started.wait()
            first_failed.set()
            raise RuntimeError("first batch failed")
        if filename == "00002.png":
            second_started.set()
            await first_failed.wait()
            await asyncio.sleep(0)
            second_finished.set()
            if second_fails:
                raise ValueError("second batch failed")
            return
        raise AssertionError("started queued work after failure")

    monkeypatch.setattr(util, "prepare_batch_with_meta_units", prepare)
    monkeypatch.setattr(util, "process_batch", process)

    async def attempt() -> None:
        document = JobState()
        state_refs.append(weakref.ref(document))
        try:
            await util.ascan_and_process_images(
                source_dir=str(tmp_path),
                output_dir=str(tmp_path),
                prompt=SimpleNamespace(),
                document=document,
                engine=SimpleNamespace(
                    batch_processor=SimpleNamespace(max_concurrency=2)
                ),
                expect_output="json",
                mini_batch_size=1,
            )
        except RuntimeError as error:
            assert str(error) == "first batch failed"
            names = [frame.name for frame in traceback.extract_tb(error.__traceback__)]
            assert "process" in names
            assert "ascan_and_process_images" in names
        else:
            pytest.fail("expected the first batch failure to propagate")

    async def run() -> None:
        await attempt()
        await asyncio.sleep(0)
        assert second_finished.is_set()
        assert processed == ["00001.png", "00002.png"]
        assert len(state_refs) == 3
        assert all(ref() is None for ref in state_refs)

    was_enabled = gc.isenabled()
    gc.disable()
    try:
        asyncio.run(run())
    finally:
        if was_enabled:
            gc.enable()
        gc.collect()


@pytest.mark.parametrize("entrypoint", ["scan", "executor", "executor_gc"])
@pytest.mark.parametrize(
    ("failure", "invalid_count"),
    [
        ("call", 0),
        ("json", 1),
        ("json", 2),
        ("json_then_call", 1),
        ("json_then_call", 2),
    ],
)
def test_real_batch_releases_images_after_output_or_call_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entrypoint: str,
    failure: str,
    invalid_count: int,
) -> None:
    class Document:
        page_count = 3

    refs: list[weakref.ReferenceType] = []
    for index in range(1, 4):
        (tmp_path / f"{index:05d}.png").touch()

    def make_document(**kwargs: object) -> Document:
        document = Document()
        refs.append(weakref.ref(document))
        return document

    def make_images(*args: object) -> list[Image.Image]:
        images = [Image.new("RGB", (24, 32)) for _ in range(3)]
        refs.extend(weakref.ref(image) for image in images)
        return images

    def prepare(*, file_batch: list[str], **kwargs: object) -> Iterator[list]:
        yield [
            (image, "prompt", name) for image, name in zip(make_images(), file_batch)
        ]

    async def engine(
        prompts: list[str], *, on_result: Callable[[str, str], None], **kwargs: object
    ) -> list[str]:
        if failure == "call":
            raise RuntimeError("model call failed")
        responses = ["not JSON"] * invalid_count + ['{"ok": true}'] * (
            3 - invalid_count
        )
        for index, response in enumerate(responses):
            on_result(f"request_task_{index}", response)
        if failure == "json_then_call":
            raise RuntimeError("model call failed")
        return responses

    async def annotate(document: Document, frames: list[Image.Image]) -> None:
        await util.ascan_and_process_images(
            source_dir=str(tmp_path),
            output_dir=str(tmp_path),
            prompt=SimpleNamespace(),
            document=document,
            engine=engine,
            expect_output="json",
            mini_batch_size=3,
        )

    class Annotator:
        def __init__(self, **kwargs: object) -> None:
            pass

        async def aannotate(
            self, document: Document, frames: list[Image.Image]
        ) -> None:
            await annotate(document, frames)

    monkeypatch.setattr(util, "prepare_batch_with_meta_units", prepare)
    monkeypatch.setattr(
        executor_module, "docs_from_asset", lambda *a, **k: ([], "input.pdf")
    )
    monkeypatch.setattr(executor_module, "frames_from_docs", make_images)
    monkeypatch.setattr(executor_module, "get_payload_features", lambda *a, **k: [])
    monkeypatch.setattr(
        executor_module,
        "layout_config",
        lambda *a: OmegaConf.create({"annotators": {"test": {"enabled": True}}}),
    )
    monkeypatch.setattr(
        executor_module,
        "prepare_asset_directory",
        lambda **k: (str(tmp_path), str(tmp_path), "meta.json"),
    )
    monkeypatch.setattr(executor_module, "load_json_file", lambda *a: {"ocr": {}})
    monkeypatch.setattr(
        executor_module, "MetaReader", SimpleNamespace(from_data=make_document)
    )
    monkeypatch.setattr(executor_module, "MARIE_KERNEL_AVAILABLE", False)
    monkeypatch.setattr(
        executor_module,
        "torch_gc",
        gc.collect if entrypoint == "executor_gc" else lambda: None,
    )
    monkeypatch.setattr(executor_module, "MDC", SimpleNamespace(remove=lambda *a: None))
    expected_error = JSONOutputParserError if failure == "json" else RuntimeError

    async def attempt() -> None:
        if entrypoint == "scan":
            document = make_document()
            frames = make_images()
            try:
                await annotate(document, frames)
            except expected_error as error:
                names = [
                    frame.name for frame in traceback.extract_tb(error.__traceback__)
                ]
                assert "process_batch" in names
                if failure == "json":
                    assert "_write_single_result" in names
                else:
                    assert str(error) == "model call failed"
            else:
                pytest.fail("expected batch failure")
        else:
            executor = SimpleNamespace(
                _setup_request=lambda *a: None,
                logger=SimpleNamespace(
                    info=lambda *a, **k: None, error=lambda *a, **k: None
                ),
                root_config_dir=str(tmp_path),
                show_error=True,
                runtime_info={},
            )
            response = await executor_module.DocumentAnnotatorExecutor._process_annotation_request(
                executor,
                docs=[SimpleNamespace(asset_key="synthetic", pages=[])],
                parameters={
                    "job_id": "synthetic",
                    "payload": {"op_params": {"key": "test", "layout": "test"}},
                },
                annotator_class=Annotator,
            )
            assert response["status"] == "error"
            assert response["error_details"]["type"] == expected_error.__name__
            if failure != "json":
                assert response["error_details"]["message"] == "model call failed"

    async def run() -> None:
        await attempt()
        await asyncio.sleep(0)
        assert len(list(tmp_path.glob("*.json"))) == (
            0 if failure == "call" else 3 - invalid_count
        )
        assert len(refs) == 7
        assert all(ref() is None for ref in refs)

    was_enabled = gc.isenabled()
    gc.disable()
    try:
        asyncio.run(run())
    finally:
        if was_enabled:
            gc.enable()
        gc.collect()
