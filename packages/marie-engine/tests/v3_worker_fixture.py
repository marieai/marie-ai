"""Owned subprocess fixture: actual BatchProcessor producer, no parent heartbeat."""

import asyncio
import json
import os
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from test_v3_producer import calls, processor_for


def record(path, event):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(event) + '\n')


def main():
    config = json.loads(sys.argv[1])
    if config.get('mode') == 'dispatcher':
        asyncio.run(dispatcher(config))
        return
    os.environ['LLM_QUEUE_PRODUCER_TTL_SECONDS'] = '1'
    os.environ['LLM_QUEUE_PRODUCER_REFRESH_INTERVAL_SECONDS'] = '0.1'
    store = SimpleNamespace(
        test_url=config['url'], keys=SimpleNamespace(fabric_id=config['fabric'])
    )
    processor = processor_for(store, batch_timeout=30)
    events = Path(config['events'])
    lock = threading.Lock()

    def record(event):
        with lock, events.open('a') as stream:
            stream.write(json.dumps(event) + '\n')

    def submit():
        from marie.engine.engine_utils import open_ai_like_formatting
        from PIL import Image

        prepared = calls(config['count'])
        with Image.new('RGB', (16, 16), 'white') as image:
            for call in prepared:
                call.messages.append(
                    {'role': 'user', 'content': open_ai_like_formatting([image], True)}
                )
        try:
            result = processor.batch_generate_calls(
                calls=prepared,
                request_id='same-node',
                on_result=lambda task, text: record(
                    {'event': 'output', 'task': task, 'text': text}
                ),
            )
            record({'event': 'complete', 'results': result})
        except Exception as error:
            record({'event': 'error', 'type': type(error).__name__})

    worker = threading.Thread(target=submit)
    worker.start()
    while (
        processor._queued_executor is None
        or processor._queued_executor.producer_id is None
    ):
        time.sleep(0.01)
    record(
        {
            'event': 'session',
            'producer': processor._queued_executor.producer_id,
            'pid': os.getpid(),
        }
    )
    worker.join()
    processor.close()


async def dispatcher(config):
    if config.get('trace'):
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            SimpleSpanProcessor,
            SpanExporter,
            SpanExportResult,
        )

        class EventSpanExporter(SpanExporter):
            def export(self, spans):
                for span in spans:
                    record(
                        config['events'],
                        {
                            'event': 'span',
                            'name': span.name,
                            'attributes': dict(span.attributes),
                            'status': span.status.status_code.name,
                            'trace_id': format(span.context.trace_id, '032x'),
                            'span_id': format(span.context.span_id, '016x'),
                        },
                    )
                return SpanExportResult.SUCCESS

        if os.environ.get('OTEL_SDK_DISABLED') != 'false':
            raise RuntimeError('history child requires tracing')
        provider = trace.get_tracer_provider()
        if not isinstance(provider, TracerProvider):
            provider = TracerProvider()
            trace.set_tracer_provider(provider)
        provider.add_span_processor(SimpleSpanProcessor(EventSpanExporter()))

    from marie.engine.llm_queue.store import RequestStore, StoreLimits
    from test_request_dispatcher import dispatcher_for

    store = RequestStore(
        config['url'],
        fabric_id=config['fabric'],
        version='v3',
        limits=StoreLimits(**config['limits']),
    )
    runtime = dispatcher_for(store, config['endpoint'], owner_lease_ms=500)
    barrier = config.get('barrier', 'before_fetch')

    def block(event):
        record(config['events'], event)
        time.sleep(60)

    if barrier == 'before_fetch':
        original_fetch = store.fetch_payload

        def blocked_fetch(owner, attempt, **kwargs):
            block({'event': 'claimed', 'attempt': attempt, 'pid': os.getpid()})
            return original_fetch(owner, attempt, **kwargs)

        store.fetch_payload = blocked_fetch
    elif barrier in {'before_finish', 'after_finish'}:
        original_finish = store.finish
        first_finish = True
        finish_lock = threading.Lock()

        def controlled_finish(owner, attempt, **kwargs):
            nonlocal first_finish
            with finish_lock:
                should_block = first_finish
                first_finish = False
            if not should_block:
                return original_finish(owner, attempt, **kwargs)
            event = {
                'event': barrier,
                'attempt': attempt,
                'claim_id': kwargs['claim_id'],
                'execution_seq': kwargs['execution_seq'],
                'expires_at_ms': store.metadata(attempt).expires_at_ms,
                'pid': os.getpid(),
            }
            if barrier == 'before_finish':
                block(event)
                return original_finish(owner, attempt, **kwargs)
            reply = original_finish(owner, attempt, **kwargs)
            event['reply'] = asdict(reply)
            block(event)
            return reply

        store.finish = controlled_finish
    elif barrier is not None:
        raise ValueError(f'Unknown dispatcher barrier: {barrier}')
    await runtime.start()
    await asyncio.Event().wait()


if __name__ == '__main__':
    main()
