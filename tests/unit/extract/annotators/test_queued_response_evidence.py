import hashlib
import json
import threading
from types import SimpleNamespace

import pytest
from marie.engine.completion_contract import CompletionCallParams, RequestContext
from marie.engine.completion_observer import CompletionAttempt
from PIL import Image

from marie.extract.annotators.context_provider import ProcessingUnit
from marie.extract.annotators.util import process_batch


@pytest.mark.asyncio
@pytest.mark.parametrize("tamper", [False, True])
async def test_lazy_handler_binds_units_and_images_before_out_of_order_delivery(
    tmp_path,
    tamper,
):
    images = []
    units = [ProcessingUnit(page_number=i + 1, data={"source": i}) for i in range(2)]
    conf = {"model_config": {"temperature": 0.0}}
    received = []

    def prepare(index):
        image = Image.new("RGB", (24, 24), color=(index + 10, 20, 30))
        images.append(image)
        return [
            image,
            f"prompt-{index}",
            str(tmp_path / f"frame_{index + 1:04d}.png"),
            "",
        ]

    def execute(*, calls, attempt_observer, on_result, **kwargs):
        for i in range(len(calls)):
            calls[i]
            assert list(tmp_path.glob("requests/**/request.json"))
            with pytest.raises(ValueError):
                images[-1].getpixel((0, 0))
        units[0].data["source"] = 99
        conf["model_config"]["temperature"] = 1.0
        if tamper:
            path = next(tmp_path.glob("requests/**/0/image.png"))
            Image.new("RGB", (24, 24), "white").save(path)
        for i in [1, 0]:
            task_id = f"batch_task_{i}"
            attempt_observer(
                CompletionAttempt(
                    task_id, 1, f"raw-{i}", "stop", "model", 2, 3, 5, None
                )
            )
            on_result(task_id, f"processed-{i}")
        return ["processed-0", "processed-1"]

    def handler(**kw):
        received.append(
            (
                kw["unit"].data["source"],
                kw["image"].getpixel((0, 0)),
                kw["prompt"],
                kw["response"],
                kw["annotator_conf"]["model_config"]["temperature"],
            )
        )
        return {"source": kw["unit"].data["source"]}

    engine = SimpleNamespace(
        batch_processor=SimpleNamespace(
            uses_queued_dispatch=True, batch_timeout=5, batch_generate_calls=execute
        ),
        _build_completion_calls=lambda **kwargs: [
            CompletionCallParams(model="model", messages=[])
        ],
    )
    work = process_batch(
        [(None, "", str(tmp_path / f"frame_{i + 1:04d}.png"), "") for i in range(2)],
        engine,
        str(tmp_path),
        expect_output="json",
        prepare_item=prepare,
        preparation_lock=threading.Lock(),
        response_handler=handler,
        processing_units=units,
        document=SimpleNamespace(),
        annotator_conf=conf,
        request_context=RequestContext(job_id="job", work_unit_id="work"),
    )
    if tamper:
        with pytest.raises(ValueError, match="image evidence changed"):
            await work
        assert not (tmp_path / "frame_0001.json").exists()
        return
    result = await work
    assert result == [{"source": 0}, {"source": 1}]
    assert received == [
        (1, (11, 20, 30), "prompt-1", "raw-1", 0.0),
        (0, (10, 20, 30), "prompt-0", "raw-0", 0.0),
    ]
    for i in range(2):
        directory = tmp_path / "attempts" / f"batch_task_{i}"
        request = json.loads((directory / "request.json").read_text())
        assert (
            request["image"]["sha256"]
            == hashlib.sha256((directory / "image.png").read_bytes()).hexdigest()
        )
        assert request["source_identity"]["job_id"] == "job"
        assert request["source_identity"]["page_number"] == i + 1
        assert request["processing_unit"]["page_number"] == i + 1
