"""Scanner-owned image lifetime through the real V3 transport."""

import asyncio
import weakref
from types import SimpleNamespace

import pytest
from PIL import Image
from test_request_dispatcher import dispatcher_for, eventually, http_endpoint, store
from test_v3_producer import engine_for

from marie.extract.annotators import util


async def test_scanner_releases_prepared_images_while_provider_waits(
    store, http_endpoint, monkeypatch, tmp_path
):
    url, received, releases, _ = http_endpoint
    releases['model'] = asyncio.Event()
    runtime = dispatcher_for(store, url)
    engine = engine_for(store, monkeypatch, multimodal=True)
    source = tmp_path / 'source'
    output = tmp_path / 'output'
    source.mkdir()
    output.mkdir()
    for index in range(1, 10):
        with Image.new('RGB', (64, 96), (index, 20, 30)) as image:
            image.save(source / f'page_{index}.png')
    refs = []
    original = util.preprocess_images_for_inference

    def observed(*args, **kwargs):
        images = original(*args, **kwargs)
        refs.extend(weakref.ref(image) for image in images)
        return images

    monkeypatch.setattr(util, 'preprocess_images_for_inference', observed)
    await runtime.start()
    try:
        await eventually(lambda: store.resolve_route('pool'))
        task = asyncio.create_task(
            util.ascan_and_process_images(
                str(source),
                str(output),
                SimpleNamespace(render=lambda _: 'prompt'),
                SimpleNamespace(source_metadata={'ocr': []}),
                engine=engine,
                is_multimodal=True,
                expect_output='none',
                mm_processor_kwargs={'min_pixels': 784, 'max_pixels': 1568},
            )
        )
        await eventually(lambda: store.usage()['active_items'] == 9)
        assert not task.done()
        assert len(refs) == 9
        assert all(ref() is None for ref in refs)
        assert len(list(output.glob('*.png'))) == 9
        assert not list(output.glob('*.md'))
        releases['model'].set()
        await task
        image_part = received[0][0]['messages'][-1]['content'][0]
        assert image_part['min_pixels'] == 784
        assert image_part['max_pixels'] == 1568
        for source_path in source.glob('*.png'):
            assert source_path.read_bytes() == (output / source_path.name).read_bytes()
    finally:
        releases['model'].set()
        await asyncio.to_thread(engine.close)
        await runtime.stop()


async def test_scanner_rejects_decoded_oversize_before_preparation(
    store, http_endpoint, monkeypatch, tmp_path
):
    url, received, _, _ = http_endpoint
    runtime = dispatcher_for(store, url)
    engine = engine_for(store, monkeypatch, multimodal=True)
    with Image.new('RGB', (64, 96)) as image:
        image.save(tmp_path / 'page_1.png')
    monkeypatch.setenv('MARIE_LLM_PREPARATION_BYTES', '1024')
    prepared = []
    original = util.preprocess_images_for_inference

    def observed(*args, **kwargs):
        prepared.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(util, 'preprocess_images_for_inference', observed)
    await runtime.start()
    try:
        await eventually(lambda: store.resolve_route('pool'))
        with pytest.raises(ValueError, match='preparation'):
            await util.ascan_and_process_images(
                str(tmp_path),
                str(tmp_path),
                SimpleNamespace(render=lambda _: 'prompt'),
                SimpleNamespace(source_metadata={'ocr': []}),
                engine=engine,
                is_multimodal=True,
                expect_output='none',
            )
        assert not prepared
        assert not received
        assert store.usage()['active_items'] == 0
    finally:
        await asyncio.to_thread(engine.close)
        await runtime.stop()


@pytest.mark.parametrize('llm_path', [True, False])
async def test_executor_releases_only_llm_source_frames(
    monkeypatch, tmp_path, llm_path
):
    import logging

    import numpy as np
    from omegaconf import OmegaConf

    from marie.executor.extract import document_annotator_executor as module
    from marie.executor.extract.document_annotator_llm_executor import (
        DocumentAnnotatorLLMExecutor,
    )
    from marie.extract.annotators.llm_annotator import LLMAnnotator

    refs = []

    class NeedsFrames:
        def __init__(self, **kwargs):
            pass

        async def aannotate(self, document, frames):
            assert len(frames) == 2
            assert all(ref() is not None for ref in refs)

    class LLM(LLMAnnotator):
        def __init__(self, **kwargs):
            pass

        async def aannotate(self, document, frames):
            assert frames == []
            assert all(ref() is None for ref in refs)

    def load(*args, **kwargs):
        docs = [
            SimpleNamespace(tensor=np.zeros((48, 32, 3), dtype=np.uint8))
            for _ in range(2)
        ]
        refs.extend(weakref.ref(doc.tensor) for doc in docs)
        return docs, 'synthetic.tif'

    monkeypatch.setattr(module, 'docs_from_asset', load)
    monkeypatch.setattr(
        module, 'frames_from_docs', lambda docs: [d.tensor for d in docs]
    )
    monkeypatch.setattr(module, 'get_payload_features', lambda *a, **k: [])
    monkeypatch.setattr(
        module,
        'layout_config',
        lambda *a: OmegaConf.create({'annotators': {'test': {'enabled': True}}}),
    )
    monkeypatch.setattr(
        module,
        'prepare_asset_directory',
        lambda **k: (str(tmp_path), str(tmp_path), 'meta.json'),
    )
    monkeypatch.setattr(module, 'load_json_file', lambda *a: {'ocr': []})
    monkeypatch.setattr(
        module,
        'MetaReader',
        SimpleNamespace(from_data=lambda **k: SimpleNamespace(page_count=2)),
    )
    monkeypatch.setattr(module, 'MARIE_KERNEL_AVAILABLE', False)
    monkeypatch.setattr(module, 'store_assets', lambda *a, **k: None)
    monkeypatch.setattr(module, 'torch_gc', lambda: None)
    executor = object.__new__(DocumentAnnotatorLLMExecutor)
    executor.logger = logging.getLogger('executor-test')
    executor.root_config_dir = str(tmp_path)
    executor.runtime_info = {'worker': 'synthetic'}
    executor.show_error = True
    executor._setup_request = lambda *a: None
    executor._record_annotation_assets = lambda **k: None
    result = await executor._process_annotation_request(
        [SimpleNamespace(asset_key='synthetic.tif', pages=[0, 1])],
        {
            'job_id': 'same-job',
            'ref_id': 'synthetic',
            'ref_type': 'test',
            'payload': {'op_params': {'key': 'test', 'layout': 'test'}},
        },
        LLM if llm_path else NeedsFrames,
    )
    assert result['status'] == 'success', result


def executor_request(tmp_path, monkeypatch, engine, count=9, dimensions=(2525, 3293)):
    """Real executor and annotator; only external assets/config/models are supplied."""
    import logging

    from omegaconf import OmegaConf

    from marie.executor.extract import document_annotator_executor as module
    from marie.executor.extract.document_annotator_llm_executor import (
        DocumentAnnotatorLLMExecutor,
    )
    from marie.extract.annotators import llm_annotator
    from marie.utils import docs as docs_module

    source = tmp_path / 'frames'
    source.mkdir()
    for index in range(count):
        with Image.new('RGB', dimensions, (index, 20, 30)) as image:
            image.save(source / f'page_{index}.png')
        (source / f'page_{index}_INJECTED_TEXT.txt').write_text(str(index))
    document_path = tmp_path / 'synthetic.tif'
    pages = [Image.open(source / f'page_{i}.png') for i in range(count)]
    try:
        pages[0].save(
            document_path,
            save_all=True,
            append_images=pages[1:],
            compression='tiff_deflate',
        )
    finally:
        for page in pages:
            page.close()
    del pages
    monkeypatch.setattr(
        docs_module, 'fetch_asset_to_temp', lambda _: (str(document_path), 'tiff')
    )
    original_load = module.docs_from_asset
    prompt_dir = tmp_path / 'extract' / 'TID-test' / 'annotator'
    prompt_dir.mkdir(parents=True)
    prompt = prompt_dir / 'prompt.txt'
    prompt.write_text('{{ INJECTED_TEXT }}')
    refs = []
    phases = []

    def load(*args, **kwargs):
        docs, path = original_load(*args, **kwargs)
        refs.extend(weakref.ref(doc.tensor) for doc in docs)
        import psutil

        phases.append(
            {
                'phase': 'source_decoded',
                'rss': psutil.Process().memory_info().rss,
                'source_pixel_bytes': count * dimensions[0] * dimensions[1] * 3,
            }
        )
        return docs, path

    conf = {
        'enabled': True,
        'mode': 'per-page',
        'model_config': {
            'model_name': 'model',
            'prompt_path': prompt.name,
            'multimodal': True,
            'expect_output': 'none',
            'mini_batch_size': 16,
            'temperature': 0.25,
            'max_tokens': 512,
            'min_pixels': 784,
            'max_pixels': 1568,
            'system_prompt_text': 'fixture-system',
        },
    }
    monkeypatch.setattr(module, 'docs_from_asset', load)
    monkeypatch.setattr(module, 'get_payload_features', lambda *a, **k: [])
    monkeypatch.setattr(
        module,
        'layout_config',
        lambda *a: OmegaConf.create({'annotators': {'test': conf}}),
    )
    monkeypatch.setattr(
        module,
        'prepare_asset_directory',
        lambda **k: (str(tmp_path), str(source), 'meta.json'),
    )
    metadata = {
        'pages': count,
        'ocr': [
            {
                'meta': {
                    'page': i,
                    'lines': [],
                    'lines_bboxes': [],
                    'imageSize': {'width': dimensions[0], 'height': dimensions[1]},
                },
                'lines': [],
                'words': [],
            }
            for i in range(count)
        ],
    }
    monkeypatch.setattr(module, 'load_json_file', lambda *a: metadata)
    monkeypatch.setattr(module, 'MARIE_KERNEL_AVAILABLE', False)
    monkeypatch.setattr(
        module, 'store_assets', lambda *a, **k: phases.append({'phase': 'asset_upload'})
    )
    monkeypatch.setattr(module, 'torch_gc', lambda: None)
    monkeypatch.setattr(llm_annotator, '__config_dir__', str(tmp_path))
    monkeypatch.setattr(llm_annotator, 'route_llm_engine', lambda *a: engine)
    executor = object.__new__(DocumentAnnotatorLLMExecutor)
    executor.logger = logging.getLogger('executor-test')
    executor.root_config_dir = str(tmp_path)
    executor.runtime_info = {'worker': 'synthetic-worker'}
    executor.show_error = True
    executor._setup_request = lambda *a: None
    executor._record_annotation_assets = lambda **k: phases.append(
        {'phase': 'annotation_assets', **k}
    )
    request = executor.annotator_llm(
        [SimpleNamespace(asset_key='synthetic.tif', pages=list(range(count)))],
        {
            'job_id': 'same-job',
            'node_task_id': 'same-node',
            'dag_id': 'same-dag',
            'ref_id': 'synthetic',
            'ref_type': 'test',
            'pool_id': 'pool',
            'payload': {'op_params': {'key': 'test', 'layout': 'test'}},
        },
    )
    return request, refs, phases


async def test_generic_v3_group_rejects_total_preparation_before_formatting(
    store, http_endpoint, monkeypatch
):
    from marie.engine.exceptions import BatchExecutionError

    url, received, _, _ = http_endpoint
    runtime = dispatcher_for(store, url)
    engine = engine_for(store, monkeypatch, multimodal=True)
    monkeypatch.setenv('MARIE_LLM_PREPARATION_BYTES', '1024')
    with Image.new('RGB', (8, 8)) as first, Image.new('RGB', (8, 8)) as second:
        inputs = [[first, second, 'ordered pair']]
        await runtime.start()
        try:
            await eventually(lambda: store.resolve_route('pool'))
            with pytest.raises((ValueError, BatchExecutionError), match='preparation'):
                await asyncio.to_thread(engine.batch_generate, inputs)
            assert not received
            assert inputs == [[first, second, 'ordered pair']]
            assert first.getpixel((0, 0)) == (0, 0, 0)
        finally:
            await asyncio.to_thread(engine.close)
            await runtime.stop()


@pytest.mark.parametrize('cancel_call', [False, True])
async def test_unknown_remote_releases_local_body_but_retains_reservation(
    store, http_endpoint, monkeypatch, cancel_call
):
    import gc

    from marie.engine.completion_contract import CompletionCallParams
    from marie.engine.llm_queue.endpoint import RegisteredEndpoint
    from marie.engine.llm_queue.request_dispatcher import (
        DispatchLane,
        RequestDispatcher,
    )
    from test_request_dispatcher import admit_request
    from v3_qualification_evidence import record_evidence

    class BodyPart(dict):
        pass

    refs = []
    original = store.fetch_payload

    def observed(*args, **kwargs):
        payload = original(*args, **kwargs)
        part = BodyPart(payload['call']['messages'][0])
        payload['call']['messages'][0] = part
        refs.append(weakref.ref(part))
        return payload

    monkeypatch.setattr(store, 'fetch_payload', observed)
    url, received, releases, connections = http_endpoint
    releases['slow'] = asyncio.Event()
    runtime = RequestDispatcher(
        store=store,
        endpoints=[
            RegisteredEndpoint(
                'endpoint',
                url,
                allow_loopback=True,
                call_timeout_seconds=5 if cancel_call else 0.1,
            )
        ],
        lanes=[DispatchLane('pool', 'endpoint')],
        poll_seconds=0.01,
    )
    await runtime.start()
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        await eventually(lambda: store.resolve_route('pool'))
        request = admit_request(
            store,
            call=CompletionCallParams(
                model='slow',
                messages=[
                    {
                        'role': 'user',
                        'content': [
                            {
                                'type': 'image_url',
                                'image_url': {
                                    'url': 'data:image/png;base64,c3ludGhldGlj'
                                },
                            }
                        ],
                    }
                ],
            ),
        )
        if cancel_call:
            await eventually(lambda: bool(received))
            runtime._tasks[request.attempt_id].cancel()
        await eventually(
            lambda: store.metadata(request.attempt_id).state == 'outcome_unknown'
        )
        await eventually(lambda: not runtime._tasks)
        assert refs and all(ref() is None for ref in refs)
        assert len(received) == 1 and not releases['slow'].is_set()
        store.close_producer(request.producer_id)
        await eventually(
            lambda: store.metadata(request.attempt_id).state == 'abandoned'
        )
        record = store.client.hgetall(store.keys.request(request.attempt_id))
        assert b'payload' not in record and b'result' not in record
        assert store.usage()['reserved_items'] == 1
        reservation = store.metadata(request.attempt_id)
        await asyncio.sleep(0.25)
        assert store.usage()['reserved_items'] == 1
        usage_before_settlement = store.usage()
        with pytest.raises(ValueError):
            store.settle_remote(
                runtime.owner,
                request.attempt_id,
                claim_id=reservation.claim_id,
                execution_seq=reservation.execution_seq,
                evidence='uncertainty_elapsed',
            )
        assert store.usage()['reserved_items'] == 1
        releases['slow'].set()
        await eventually(
            lambda: all(connection.is_closing() for connection in connections)
        )
        await runtime.stop()
        maintenance_owner = store.acquire_owner('quiescent-maintenance', lease_ms=5000)
        settled = store.settle_remote(
            maintenance_owner,
            request.attempt_id,
            claim_id=reservation.claim_id,
            execution_seq=reservation.execution_seq,
            evidence='remote_completed',
        )
        assert settled.disposition == 'settled'
        assert store.usage()['reserved_items'] == 0
        assert releases['slow'].is_set()
        record_evidence(
            'unknown-body-cancel' if cancel_call else 'unknown-body',
            store,
            {
                'attempt': request.attempt_id,
                'local_body_refs': len(refs),
                'live_body_refs': sum(r() is not None for r in refs),
                'before_explicit_settlement': usage_before_settlement,
                'after_explicit_settlement': store.usage(),
                'remote_still_blocked_when_reservation_released': False,
                'settlement_policy': 'affirmative completion after fixture quiescence and owner stop',
            },
        )
    finally:
        if was_enabled:
            gc.enable()
        releases['slow'].set()
        await runtime.stop()


async def test_scanner_streaming_rejected_before_prepared_images(
    store, monkeypatch, tmp_path
):
    from marie.engine.completion_contract import UnsupportedQueueStreaming

    engine = engine_for(store, monkeypatch, multimodal=True)
    with Image.new('RGB', (8, 8)) as image:
        image.save(tmp_path / 'page_1.png')
    output = tmp_path / 'output'
    output.mkdir()
    try:
        with pytest.raises(UnsupportedQueueStreaming):
            await util.ascan_and_process_images(
                str(tmp_path),
                str(output),
                SimpleNamespace(render=lambda _: 'prompt'),
                SimpleNamespace(source_metadata={'ocr': []}),
                engine=engine,
                is_multimodal=True,
                expect_output='none',
                completion_params={'extra_body': {'stream': True}},
            )
        assert not list(output.iterdir())
        assert store.usage()['records'] == 0
    finally:
        await asyncio.to_thread(engine.close)


@pytest.mark.parametrize('sizes', [[(8, 8), (16, 8)], [(2525, 3293), (2525, 3293)]])
async def test_actual_two_image_group_preserves_wire_and_caller_images(
    store, http_endpoint, monkeypatch, sizes
):
    import base64
    import io

    from marie.engine.completion_contract import RequestContext

    url, received, _, _ = http_endpoint
    runtime = dispatcher_for(store, url)
    engine = engine_for(store, monkeypatch, multimodal=True)
    outputs = []
    monkeypatch.setenv('MARIE_LLM_PREPARATION_BYTES', str(256 * 1024**2))
    with (
        Image.new('RGB', sizes[0], 'red') as first,
        Image.new('RGB', sizes[1], 'blue') as second,
    ):
        inputs = [[first, second, 'ordered pair']]
        await runtime.start()
        try:
            await eventually(lambda: store.resolve_route('pool'))
            result = await asyncio.to_thread(
                engine.batch_generate,
                inputs,
                request_contexts=[RequestContext(ref_id='synthetic', page_number=1)],
                on_result=lambda ident, result: outputs.append((ident, result)),
            )
            assert result == ['model'] and len(outputs) == 1
            assert len(received) == 1
            parts = received[0][0]['messages'][-1]['content']
            for part, size, color in zip(parts, sizes, [(255, 0, 0), (0, 0, 255)]):
                with Image.open(
                    io.BytesIO(
                        base64.b64decode(part['image_url']['url'].split(',', 1)[1])
                    )
                ) as decoded:
                    assert decoded.size == size
                    assert decoded.getpixel((0, 0)) == color
            assert first.getpixel((0, 0)) == (255, 0, 0)
            assert second.getpixel((0, 0)) == (0, 0, 255)
            assert inputs == [[first, second, 'ordered pair']]
        finally:
            await asyncio.to_thread(engine.close)
            await runtime.stop()


async def test_high_entropy_page_rejects_inline_limit_after_releasing_pixels(
    store, http_endpoint, monkeypatch, tmp_path
):
    import numpy as np
    from marie.engine.exceptions import BatchExecutionError
    from v3_qualification_evidence import record_evidence

    url, received, _, _ = http_endpoint
    runtime = dispatcher_for(store, url)
    engine = engine_for(store, monkeypatch, multimodal=True, timeout=20)
    pixels = np.random.default_rng(42).integers(0, 256, (3293, 2525, 3), dtype=np.uint8)
    with Image.fromarray(pixels) as image:
        image.save(tmp_path / 'page_1.png')
    del pixels
    output = tmp_path / 'output'
    output.mkdir()
    refs = []
    original = util.preprocess_images_for_inference

    def observed(*args, **kwargs):
        images = original(*args, **kwargs)
        refs.extend(weakref.ref(image) for image in images)
        return images

    monkeypatch.setattr(util, 'preprocess_images_for_inference', observed)
    await runtime.start()
    try:
        await eventually(lambda: store.resolve_route('pool'))
        with pytest.raises(BatchExecutionError) as raised:
            await util.ascan_and_process_images(
                str(tmp_path),
                str(output),
                SimpleNamespace(render=lambda _: 'prompt'),
                SimpleNamespace(source_metadata={'ocr': []}),
                engine=engine,
                is_multimodal=True,
                expect_output='none',
            )
        assert raised.value.primary_error.category == 'invalid_request'
        assert not received and store.usage()['active_items'] == 0
        assert all(ref() is None for ref in refs)
        png_bytes = (tmp_path / 'page_1.png').stat().st_size
        assert png_bytes > 16 * 1024**2
        record_evidence(
            'default-inline-rejection',
            store,
            {
                'dimensions': [2525, 3293],
                'png_bytes': png_bytes,
                'base64_bytes': 4 * ((png_bytes + 2) // 3),
                'max_inline_payload_bytes': store.limits.max_inline_payload_bytes,
                'error_category': raised.value.primary_error.category,
                'live_prepared_images': sum(ref() is not None for ref in refs),
                'provider_count': len(received),
                'usage': store.usage(),
            },
        )
    finally:
        await asyncio.to_thread(engine.close)
        await runtime.stop()


@pytest.mark.parametrize('waiting', ['retry', 'commit'])
async def test_gateway_drops_body_before_retry_or_result_commit_wait(
    store, http_endpoint, monkeypatch, waiting
):
    import gc
    import socket

    from marie.engine.completion_contract import CompletionCallParams
    from marie.engine.llm_queue.store import StoreUnavailable
    from test_request_dispatcher import admit_request
    from v3_qualification_evidence import record_evidence

    class BodyPart(dict):
        pass

    refs = []
    original_fetch = store.fetch_payload

    def fetch(*args, **kwargs):
        payload = original_fetch(*args, **kwargs)
        part = BodyPart(payload['call']['messages'][0])
        payload['call']['messages'][0] = part
        refs.append(weakref.ref(part))
        return payload

    monkeypatch.setattr(store, 'fetch_payload', fetch)
    blocked = True
    commit_started = False
    original_finish = store.finish

    def finish(*args, **kwargs):
        nonlocal commit_started
        commit_started = True
        if blocked:
            raise StoreUnavailable('owned commit reply outage')
        return original_finish(*args, **kwargs)

    url, received, _, _ = http_endpoint
    if waiting == 'retry':
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0))
            port = listener.getsockname()[1]
        url = f'http://127.0.0.1:{port}/v1'
    else:
        monkeypatch.setattr(store, 'finish', finish)
    runtime = dispatcher_for(store, url, retry_min_ms=3000, retry_max_ms=3000)
    await runtime.start()
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        await eventually(lambda: store.resolve_route('pool'))
        request = admit_request(
            store,
            call=CompletionCallParams(
                model='synthetic',
                messages=[{'role': 'user', 'content': 'synthetic inline body'}],
            ),
        )
        if waiting == 'retry':
            await eventually(
                lambda: store.metadata(request.attempt_id).state == 'delayed'
            )
            await eventually(lambda: not runtime._tasks)
        else:
            await eventually(lambda: commit_started)
            assert runtime._tasks
        assert refs and all(ref() is None for ref in refs)
        assert len(received) == (1 if waiting == 'commit' else 0)
        record_evidence(
            'body-' + waiting,
            store,
            {
                'live_body_references': sum(ref() is not None for ref in refs),
                'cyclic_gc_disabled': True,
                'local_tasks': len(runtime._tasks),
                'provider_count': len(received),
                'usage': store.usage(),
            },
        )
        blocked = False
        if waiting == 'commit':
            await eventually(
                lambda: store.metadata(request.attempt_id).state == 'succeeded'
            )
            assert len(received) == 1
    finally:
        blocked = False
        if was_enabled:
            gc.enable()
        await runtime.stop()


@pytest.mark.parametrize('failure', ['invalid_response', 'endpoint_policy'])
async def test_transport_releases_body_after_other_sanitized_errors(
    monkeypatch, failure
):
    import gc
    import socket

    from marie.engine.completion_contract import CompletionCallParams
    from marie.engine.llm_queue.endpoint import EndpointClient, RegisteredEndpoint

    class BodyPart(dict):
        pass

    active = set()

    async def handle(reader, writer):
        active.add(asyncio.current_task())
        try:
            headers = await reader.readuntil(b'\r\n\r\n')
            size = next(
                int(line.split(b':')[1])
                for line in headers.split(b'\r\n')
                if line.lower().startswith(b'content-length:')
            )
            await reader.readexactly(size)
            writer.write(
                b'HTTP/1.1 200 OK\r\nContent-Length: 1\r\nConnection: close\r\n\r\n{'
            )
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()
            active.discard(asyncio.current_task())

    server = await asyncio.start_server(handle, '127.0.0.1', 0)
    port = server.sockets[0].getsockname()[1]
    host = '127.0.0.1'
    if failure == 'endpoint_policy':
        host = 'registered.test'

        async def rebinding(*args, **kwargs):
            return [
                (socket.AF_INET, socket.SOCK_STREAM, 6, '', ('169.254.169.254', port))
            ]

        monkeypatch.setattr(asyncio.get_running_loop(), 'getaddrinfo', rebinding)
    client = EndpointClient(
        RegisteredEndpoint('endpoint', f'http://{host}:{port}/v1', allow_loopback=True)
    )
    part = BodyPart({'role': 'user', 'content': 'synthetic image body'})
    ref = weakref.ref(part)
    call = CompletionCallParams(model='test', messages=[part])
    del part
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        result = await client.execute(call, timeout_seconds=1)
        del call
        assert result.category == failure
        assert ref() is None
    finally:
        if was_enabled:
            gc.enable()
        await client.close()
        server.close()
        await server.wait_closed()
        await asyncio.gather(*active, return_exceptions=True)
