from __future__ import annotations

import asyncio
import time
import zlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from marie.logging_core.logger import MarieLogger
from marie.utils.scheduler_trace import scheduler_trace


@dataclass(frozen=True, slots=True)
class _QueuedJobEvent:
    event_type: str
    message: Any
    enqueued_at: float


class SchedulerJobEventProcessor:
    """Process job events concurrently while preserving order for each job."""

    def __init__(
        self,
        *,
        handler: Callable[[str, Any], Awaitable[None]],
        logger: MarieLogger,
        worker_count: int,
        queue_size: int,
    ) -> None:
        if worker_count <= 0:
            raise ValueError('worker_count must be greater than zero')
        if queue_size < worker_count:
            raise ValueError('queue_size must be at least worker_count')

        self._handler = handler
        self._logger = logger
        self.worker_count = worker_count
        self.queue_capacity = queue_size

        base_size, extra_slots = divmod(queue_size, worker_count)
        self._queues = tuple(
            asyncio.Queue[_QueuedJobEvent](
                maxsize=base_size + (1 if worker_id < extra_slots else 0)
            )
            for worker_id in range(worker_count)
        )

    @property
    def queue_size(self) -> int:
        return sum(queue.qsize() for queue in self._queues)

    @property
    def queue_sizes(self) -> tuple[int, ...]:
        return tuple(queue.qsize() for queue in self._queues)

    async def enqueue(self, event_type: str, message: Any) -> None:
        job_id = message.get('job_id') if isinstance(message, dict) else None
        worker_id = self._worker_for(job_id)
        queue = self._queues[worker_id]
        enqueue_started = time.perf_counter()
        queued_message = dict(message) if isinstance(message, dict) else message

        await queue.put(
            _QueuedJobEvent(
                event_type=event_type,
                message=queued_message,
                enqueued_at=time.perf_counter(),
            )
        )
        scheduler_trace(
            'scheduler_job_event_enqueued',
            job_id=job_id,
            status=str(event_type),
            worker_id=worker_id,
            queue_size=queue.qsize(),
            total_queue_size=self.queue_size,
            enqueue_wait_ms=(time.perf_counter() - enqueue_started) * 1000.0,
        )

    async def run_worker(self, worker_id: int) -> None:
        queue = self._queues[worker_id]
        self._logger.info('Scheduler job event worker started: %s', worker_id)
        while True:
            queued_event: _QueuedJobEvent | None = None
            try:
                queued_event = await queue.get()
                processing_started = time.perf_counter()
                message = queued_event.message
                job_id = message.get('job_id') if isinstance(message, dict) else None
                trace_fields = {
                    'job_id': job_id,
                    'status': str(queued_event.event_type),
                    'worker_id': worker_id,
                }
                scheduler_trace(
                    'scheduler_job_event_dequeued',
                    **trace_fields,
                    queue_size=queue.qsize(),
                    total_queue_size=self.queue_size,
                    queue_wait_ms=(processing_started - queued_event.enqueued_at)
                    * 1000.0,
                )

                try:
                    await self._handler(queued_event.event_type, message)
                except Exception as error:
                    scheduler_trace(
                        'scheduler_job_event_failed',
                        **trace_fields,
                        error_type=type(error).__name__,
                        elapsed_ms=(time.perf_counter() - processing_started) * 1000.0,
                    )
                    self._logger.error(
                        'Scheduler job event failed for job %s: %s',
                        job_id,
                        error,
                    )
                else:
                    scheduler_trace(
                        'scheduler_job_event_processed',
                        **trace_fields,
                        queue_size=queue.qsize(),
                        total_queue_size=self.queue_size,
                        elapsed_ms=(time.perf_counter() - processing_started) * 1000.0,
                    )
            except asyncio.CancelledError:
                raise
            finally:
                if queued_event is not None:
                    queue.task_done()

    def abort_pending(self) -> int:
        aborted = 0
        for queue in self._queues:
            while True:
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                else:
                    queue.task_done()
                    aborted += 1
        return aborted

    def _worker_for(self, job_id: Any) -> int:
        routing_key = job_id if isinstance(job_id, str) and job_id else '<invalid>'
        return zlib.crc32(routing_key.encode('utf-8')) % self.worker_count
