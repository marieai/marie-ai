import asyncio
import json
import logging
import threading
from types import SimpleNamespace
from uuid import uuid4

import pytest
from marie.engine.batch_processor import BatchProcessor
from marie.engine.completion_contract import RequestContext
from marie.engine.llm_queue.endpoint import ExecutionOutcome, RegisteredEndpoint
from marie.engine.llm_queue.request_dispatcher import DispatchLane, RequestDispatcher
from marie.engine.llm_queue.store import RequestStore, RoutingManifestProjection
from marie.engine.openai_engine import OpenAIEngine
from openai import AsyncOpenAI
from PIL import Image

from marie.extract.annotators.context_provider import ProcessingUnit
from marie.extract.annotators.util import process_batch
from tests.integration.engine.test_completion_evidence import queue_url as queue_url


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_candidate", [False, True])
async def test_lazy_batch_validates_real_dispatcher_evidence_before_publication(
    queue_url, tmp_path, invalid_candidate
):
    store = RequestStore(queue_url, fabric_id=uuid4().hex)
    transport = AsyncOpenAI(api_key="synthetic-test-key")
    engine = object.__new__(OpenAIEngine)
    engine.model_string = "model"
    engine.system_prompt = "system"
    engine.is_multimodal = True
    engine.logger = logging.getLogger("evidence-test")
    engine.batch_processor = BatchProcessor(
        transport,
        "model",
        engine.logger,
        queue_enabled=True,
        queue_url=queue_url,
        queue_fabric_group_id=store.keys.fabric_id,
        batch_timeout=5,
    )
    prepared = []

    class Client:
        def __init__(self, endpoint):
            pass

        async def execute(self, call, **kwargs):
            assert call.context.page_number == 1
            assert any(
                part.get("type") == "image_url" for part in call.messages[1]["content"]
            )
            with pytest.raises(ValueError):
                prepared[0].getpixel((0, 0))
            return ExecutionOutcome(
                response={
                    "model": "model",
                    "choices": [
                        {
                            "message": {"content": ' {"ok":true} '},
                            "finish_reason": "stop",
                        }
                    ],
                },
                remote_settled=True,
            )

        async def close(self):
            pass

    runtime = RequestDispatcher(
        store=store,
        endpoints=[
            RegisteredEndpoint("endpoint", "http://127.0.0.1:9/v1", allow_loopback=True)
        ],
        lanes=[DispatchLane("pool", "endpoint")],
        client_factory=Client,
        poll_seconds=0.01,
    )

    def prepare(index):
        image = Image.new("RGB", (32, 32), (10, 20, 30))
        prepared.append(image)
        return [image, "prompt", str(tmp_path / "frame_0001.png"), ""]

    def validate(**kw):
        assert kw["image"].getpixel((0, 0)) == (10, 20, 30)
        assert kw["attempts"][0].raw_response == ' {"ok":true} '
        assert list(tmp_path.glob("attempts/*/attempt-0001.json"))
        if invalid_candidate:
            raise ValueError("candidate failed validation")
        return json.loads(kw["response"])

    try:
        await runtime.start()
        for _ in range(500):
            if runtime.owner is not None and store.resolve_route("pool"):
                break
            await asyncio.sleep(0.01)
        store.project_routing_manifest(
            RoutingManifestProjection(
                "job",
                "work",
                store.keys.fabric_id,
                1,
                "a" * 64,
                "pool",
                "endpoint",
                "r1",
            )
        )
        work = process_batch(
            [(None, "", str(tmp_path / "frame_0001.png"), "")],
            engine,
            str(tmp_path),
            is_multimodal=True,
            expect_output="json",
            prepare_item=prepare,
            preparation_lock=threading.Lock(),
            response_handler=validate,
            processing_units=[ProcessingUnit(page_number=1)],
            document=SimpleNamespace(),
            annotator_conf={},
            request_context=RequestContext(job_id="job", work_unit_id="work"),
        )
        if invalid_candidate:
            with pytest.raises(ValueError, match="candidate failed validation"):
                await work
            assert not (tmp_path / "frame_0001.json").exists()
        else:
            assert await work == [{"ok": True}]
            assert json.loads((tmp_path / "frame_0001.json").read_text()) == {
                "ok": True
            }
        assert store.usage()["reserved_items"] == 0
    finally:
        await asyncio.to_thread(engine.batch_processor.close)
        await transport.close()
        await runtime.stop()
        store.close()
