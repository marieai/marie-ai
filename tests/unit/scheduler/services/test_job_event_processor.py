import asyncio
from unittest.mock import MagicMock

import pytest

import marie.scheduler.services.job_event_processor as processor_module
from marie.scheduler.services.job_event_processor import SchedulerJobEventProcessor


async def _stop_workers(*workers: asyncio.Task[None]) -> None:
    for worker in workers:
        worker.cancel()
    await asyncio.gather(*workers, return_exceptions=True)


def _job_ids_for_different_workers(
    processor: SchedulerJobEventProcessor,
) -> tuple[str, str]:
    first_job_id = 'job-0'
    first_worker = processor._worker_for(first_job_id)
    for index in range(1, 100):
        candidate = f'job-{index}'
        if processor._worker_for(candidate) != first_worker:
            return first_job_id, candidate
    raise AssertionError('Could not find job IDs assigned to different workers')


@pytest.mark.asyncio
async def test_processor_preserves_per_job_order_while_other_jobs_progress() -> None:
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    other_job_processed = asyncio.Event()
    terminal_processed = asyncio.Event()
    observed: list[tuple[str, str]] = []

    async def handler(event_type: str, message: dict) -> None:
        job_id = message['job_id']
        observed.append((job_id, event_type))
        if job_id == first_job_id and event_type == 'RUNNING':
            first_started.set()
            await release_first.wait()
        elif job_id == first_job_id:
            terminal_processed.set()
        else:
            other_job_processed.set()

    processor = SchedulerJobEventProcessor(
        handler=handler,
        logger=MagicMock(),
        worker_count=2,
        queue_size=8,
    )
    first_job_id, other_job_id = _job_ids_for_different_workers(processor)
    workers = tuple(
        asyncio.create_task(processor.run_worker(worker_id))
        for worker_id in range(processor.worker_count)
    )

    try:
        await processor.enqueue('RUNNING', {'job_id': first_job_id})
        await processor.enqueue('SUCCEEDED', {'job_id': first_job_id})
        await processor.enqueue('SUCCEEDED', {'job_id': other_job_id})

        await asyncio.wait_for(first_started.wait(), timeout=1.0)
        await asyncio.wait_for(other_job_processed.wait(), timeout=1.0)
        assert (first_job_id, 'SUCCEEDED') not in observed

        release_first.set()
        await asyncio.wait_for(terminal_processed.wait(), timeout=1.0)
        first_job_events = [event for job, event in observed if job == first_job_id]
        assert first_job_events == ['RUNNING', 'SUCCEEDED']
    finally:
        await _stop_workers(*workers)


@pytest.mark.asyncio
async def test_processor_continues_after_handler_failure() -> None:
    second_processed = asyncio.Event()

    async def handler(event_type: str, _message: dict) -> None:
        if event_type == 'FAILED':
            raise RuntimeError('expected failure')
        second_processed.set()

    processor = SchedulerJobEventProcessor(
        handler=handler,
        logger=MagicMock(),
        worker_count=1,
        queue_size=2,
    )
    worker = asyncio.create_task(processor.run_worker(0))

    try:
        await processor.enqueue('FAILED', {'job_id': 'job-1'})
        await processor.enqueue('SUCCEEDED', {'job_id': 'job-1'})
        await asyncio.wait_for(second_processed.wait(), timeout=1.0)
        assert not worker.done()
    finally:
        await _stop_workers(worker)


@pytest.mark.asyncio
async def test_processor_traces_its_queue_and_processing_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    traces: list[tuple[str, dict]] = []
    processed = asyncio.Event()

    async def handler(_event_type: str, _message: dict) -> None:
        processed.set()

    monkeypatch.setattr(
        processor_module,
        'scheduler_trace',
        lambda event, **fields: traces.append((event, fields)),
    )
    processor = SchedulerJobEventProcessor(
        handler=handler,
        logger=MagicMock(),
        worker_count=1,
        queue_size=2,
    )
    worker = asyncio.create_task(processor.run_worker(0))

    try:
        await processor.enqueue('SUCCEEDED', {'job_id': 'job-1'})
        await asyncio.wait_for(processed.wait(), timeout=1.0)
        await asyncio.sleep(0)
    finally:
        await _stop_workers(worker)

    assert [event for event, _ in traces] == [
        'scheduler_job_event_enqueued',
        'scheduler_job_event_dequeued',
        'scheduler_job_event_processed',
    ]
    assert all(fields['worker_id'] == 0 for _, fields in traces)


def test_processor_rejects_capacity_smaller_than_worker_count() -> None:
    async def handler(_event_type: str, _message: dict) -> None:
        return None

    with pytest.raises(ValueError, match='at least worker_count'):
        SchedulerJobEventProcessor(
            handler=handler,
            logger=MagicMock(),
            worker_count=2,
            queue_size=1,
        )
