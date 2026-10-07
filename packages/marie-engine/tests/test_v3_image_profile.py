"""Representative synthetic image profile; native allocation and live references.

Optional setup in an isolated profiling environment:
    python -m pip install memray==1.20.0
With project source paths and MARIE_LLM_QUEUE_TEST_STORES configured:
    python -m pytest packages/marie-engine/tests/test_v3_image_profile.py -q
The module skips when Memray is absent. These profiles require the explicitly
coordinated 512 MiB owned-store window described in the Task 6 report.
"""

import asyncio
import json
import os
import time
import weakref
from dataclasses import asdict, replace
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import psutil
import pytest
from marie.engine.llm_queue.endpoint import RegisteredEndpoint
from marie.engine.llm_queue.request_dispatcher import DispatchLane, RequestDispatcher
from marie.engine.llm_queue.store import RequestStore
from PIL import Image
from test_request_dispatcher import eventually, store
from test_v3_producer import engine_for
from v3_qualification_evidence import record_evidence

from marie.extract.annotators import util

memray = pytest.importorskip('memray')


@pytest.mark.parametrize('content', ['compressible', 'high_entropy'])
async def test_representative_pages_release_between_batches_during_outage(
    store, monkeypatch, tmp_path, content
):
    # A separately owned fabric admits the representative high-entropy inline page.
    # The production16MiB limit is qualified separately as an explicit rejection.
    limits = replace(
        store.limits,
        max_inline_payload_bytes=64 * 1024**2,
        max_storage_bytes=512 * 1024**2,
        max_attempts=100,
        claim_lease_ms=10_000,
    )
    measured = RequestStore(
        store.test_url, fabric_id='profile-' + uuid4().hex, version='v3', limits=limits
    )
    measured.test_url = store.test_url
    source = tmp_path / 'source'
    output = tmp_path / 'output'
    source.mkdir()
    output.mkdir()
    if content == 'high_entropy':
        pixels = np.random.default_rng(42).integers(
            0, 256, (3293, 2525, 3), dtype=np.uint8
        )
        image = Image.fromarray(pixels)
        del pixels
    else:
        image = Image.new('RGB', (2525, 3293), 'white')
    image.save(source / 'page_01.png')
    image.close()
    del image
    for index in range(2, 16):
        os.link(source / 'page_01.png', source / f'page_{index:02d}.png')
    png_bytes = (source / 'page_01.png').stat().st_size
    refs = []
    max_live = 0
    original_prepare = util.preprocess_images_for_inference

    def observed(*args, **kwargs):
        nonlocal max_live
        images = original_prepare(*args, **kwargs)
        refs.extend(weakref.ref(image) for image in images)
        max_live = max(max_live, sum(ref() is not None for ref in refs))
        return images

    monkeypatch.setattr(util, 'preprocess_images_for_inference', observed)
    received = 0
    active = set()

    async def handle(reader, writer):
        nonlocal received
        active.add(asyncio.current_task())
        try:
            header = await reader.readuntil(b'\r\n\r\n')
            headers = dict(
                line.split(': ', 1)
                for line in header.decode().split('\r\n')
                if ': ' in line
            )
            body = await reader.readexactly(int(headers['Content-Length']))
            del body
            received += 1
            if received == 5:
                server.close()
            payload = b'{"choices":[{"message":{"content":"synthetic-result"}}]}'
            writer.write(
                b'HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: '
                + str(len(payload)).encode()
                + b'\r\n\r\n'
                + payload
            )
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()
            active.discard(asyncio.current_task())

    server = await asyncio.start_server(handle, '127.0.0.1', 0)
    port = server.sockets[0].getsockname()[1]
    runtime = RequestDispatcher(
        store=measured,
        endpoints=[
            RegisteredEndpoint(
                'endpoint',
                f'http://127.0.0.1:{port}/v1',
                allow_loopback=True,
                execution_limit=1,
            )
        ],
        lanes=[DispatchLane('pool', 'endpoint', execution_limit=1)],
        poll_seconds=0.03,
        retry_min_ms=1000,
        retry_max_ms=1000,
        circuit_open_ms=1000,
    )
    engine = engine_for(measured, monkeypatch, multimodal=True, timeout=60)
    engine.batch_processor.max_concurrency = 5
    samples = []
    stop_sampling = asyncio.Event()

    async def sample():
        while not stop_sampling.is_set():
            samples.append(
                {
                    'elapsed': time.monotonic() - start,
                    'rss': psutil.Process().memory_info().rss,
                    'prepared': len(refs),
                    'live_images': sum(ref() is not None for ref in refs),
                    'provider_count': received,
                    'writes': len(list(output.glob('*.md'))),
                    'usage': measured.usage(),
                    'local_gateway_tasks': len(runtime._tasks),
                }
            )
            await asyncio.sleep(0.2)

    task = sampler = None
    capture = tmp_path / 'profile.bin'
    util._prompt_lines_by_page(SimpleNamespace(source_metadata={'ocr': []}))
    await runtime.start()
    try:
        await eventually(lambda: measured.resolve_route('pool'))
        start = time.monotonic()
        with memray.Tracker(
            str(capture), native_traces=True, trace_python_allocators=False
        ):
            sampler = asyncio.create_task(sample())
            task = asyncio.create_task(
                util.ascan_and_process_images(
                    str(source),
                    str(output),
                    SimpleNamespace(render=lambda _: 'synthetic'),
                    SimpleNamespace(source_metadata={'ocr': []}),
                    engine=engine,
                    is_multimodal=True,
                    expect_output='none',
                    mini_batch_size=5,
                )
            )
            await eventually(
                lambda: (
                    received == 5
                    and measured.usage()['active_items'] == 5
                    and len(refs) == 10
                    and all(ref() is None for ref in refs)
                ),
                seconds=45,
            )
            assert not task.done()
            assert max_live == 1
            # Hold through multiple retry/circuit cycles after completed siblings accumulate.
            await asyncio.sleep(3)
            assert received == 5
            assert len(list(output.glob('*.md'))) == 5
            assert all(ref() is None for ref in refs)
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await asyncio.to_thread(engine.close)
            await eventually(lambda: measured.usage()['payload_bytes'] == 0)
            stop_sampling.set()
            await sampler
        allocations = memray.FileReader(str(capture))
        peak = sum(
            record.size
            for record in allocations.get_high_watermark_allocation_records()
        )
        assert all(
            sample['usage']['reserved_bytes'] <= limits.max_execution_bytes
            for sample in samples
        )
        assert all(
            sample['usage']['payload_bytes'] <= limits.max_payload_bytes
            for sample in samples
        )
        record_evidence(
            'image-profile-' + content,
            measured,
            {
                'dimensions': [2525, 3293],
                'source_pages': 15,
                'prepared_pages': len(refs),
                'png_bytes': png_bytes,
                'rgb_pixel_bytes_per_page': 24944475,
                'preparation_estimate_per_page': 99777900,
                'preparation_budget': 134217728,
                'max_live_prepared_images': max_live,
                'native_peak_allocation_bytes': peak,
                'samples': samples,
                'limits': asdict(limits),
                'measurement_boundary': 'single process producer+gateway+owned HTTP; staged PNG input, source-document ingress measured in executor fixture',
            },
        )
    finally:
        if samples:
            record_evidence(
                'image-profile-last-sample-' + content,
                measured,
                {
                    'last_sample': samples[-1],
                    'endpoint': measured.endpoint_status('endpoint'),
                    'gateway': runtime._counts,
                    'category': runtime._category,
                    'claim_lease_ms': limits.claim_lease_ms,
                },
            )
        stop_sampling.set()
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if sampler is not None:
            await sampler
        await asyncio.to_thread(engine.close)
        await runtime.stop()
        server.close()
        await server.wait_closed()
        for active_task in list(active):
            active_task.cancel()
        await asyncio.gather(*active, return_exceptions=True)
        keys = list(store.client.scan_iter(match=measured.keys.prefix + '*'))
        if keys:
            store.client.delete(*keys)
