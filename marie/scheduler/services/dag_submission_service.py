from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Awaitable, Callable, Iterable
from typing import Any

from marie.logging_core.logger import MarieLogger
from marie.messaging import mark_as_failed as mark_as_failed_toast
from marie.messaging import mark_as_scheduled as mark_as_scheduled_toast
from marie.scheduler.job_scheduler import JobSubmissionRequest
from marie.scheduler.models import ExistingWorkPolicy, WorkInfo
from marie.scheduler.planner_util import query_plan_work_items
from marie.scheduler.repository import JobRepository
from marie.utils.scheduler_trace import scheduler_trace
from marie.utils.utils import current_milli_time


class DagSubmissionService:
    """Queue, persist, and account for submitted DAGs."""

    def __init__(
        self,
        *,
        repository: JobRepository,
        dag_admission_callback: Callable[[str], Awaitable[bool]],
        known_queues: set[str],
        notify_callback: Callable[[], Awaitable[bool]],
        is_running: Callable[[], bool],
        submission_processed_callback: Callable[[int], Awaitable[None]],
        logger: MarieLogger,
        queue_size: int,
        initial_submission_count: int = 0,
    ) -> None:
        self.repository = repository
        self._dag_admission_callback = dag_admission_callback
        self.known_queues = known_queues
        self._notify_callback = notify_callback
        self._is_running = is_running
        self._submission_processed_callback = submission_processed_callback
        self.logger = logger
        self._queue: asyncio.Queue[JobSubmissionRequest] = asyncio.Queue(
            maxsize=queue_size
        )
        self._pending: dict[str, JobSubmissionRequest] = {}
        self.submission_count = initial_submission_count

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    async def submit(self, work_info: WorkInfo, overwrite: bool = True) -> str:
        if not self._is_running():
            raise RuntimeError('Job scheduler is not running')

        self.logger.debug(f'Submitting job : {work_info.id}')
        await self._ensure_queue(work_info.name)

        result_future = asyncio.get_running_loop().create_future()
        request_id = str(uuid.uuid4())
        wait_for_result = False
        request = JobSubmissionRequest(
            work_info=work_info,
            overwrite=overwrite,
            request_id=request_id,
            result_future=result_future,
            wait_for_result=wait_for_result,
        )

        if not self._is_running():
            raise RuntimeError('Job scheduler stopped during submission')

        self._pending[request_id] = request
        try:
            queue_size_before = self._queue.qsize()
            enqueue_started = time.perf_counter()
            await self._queue.put(request)
            enqueue_wait_ms = (time.perf_counter() - enqueue_started) * 1000
            scheduler_trace(
                'scheduler_submission_enqueued',
                job_id=work_info.id,
                dag_id=work_info.id,
                job_name=work_info.name,
                request_id=request_id,
                queue_size_before=queue_size_before,
                queue_size=self._queue.qsize(),
                queue_capacity=self._queue.maxsize,
                enqueue_wait_ms=enqueue_wait_ms,
            )
            self.logger.debug(
                f'Job {work_info.id} queued successfully (request: {request_id})'
            )
            if wait_for_result:
                return await result_future
            return work_info.id
        except asyncio.CancelledError:
            self._pending.pop(request_id, None)
            if not result_future.done():
                result_future.cancel()
            raise
        except Exception as error:
            self._pending.pop(request_id, None)
            if wait_for_result and not result_future.done():
                result_future.set_exception(error)
            raise

    async def run_worker(self, worker_id: int) -> None:
        self.logger.info(f'Background job submission worker started # {worker_id}')
        while self._is_running():
            request = None
            try:
                request = await self._queue.get()
                scheduler_trace(
                    'scheduler_submission_dequeued',
                    job_id=request.work_info.id,
                    dag_id=request.work_info.id,
                    job_name=request.work_info.name,
                    request_id=request.request_id,
                    worker_id=worker_id,
                    queue_size=self._queue.qsize(),
                )
                try:
                    result = await self.persist(
                        request.work_info,
                        request.overwrite,
                    )
                    await self._send_scheduled_toast(request.work_info)
                    self.submission_count += 1
                    await self._submission_processed_callback(self.submission_count)
                    if request.wait_for_result and not request.result_future.done():
                        request.result_future.set_result(result)
                    self.logger.debug(
                        f'Successfully processed job: {request.work_info.id} '
                        f'(queue size: {self._queue.qsize()})'
                    )
                except ValueError as error:
                    self.logger.error(
                        f'Job submission failed for {request.work_info.id}: {error}'
                    )
                    if request.wait_for_result and not request.result_future.done():
                        request.result_future.set_exception(error)
                except Exception as error:
                    if request.wait_for_result and not request.result_future.done():
                        request.result_future.set_exception(error)
                    self.logger.error(
                        f'Failed to process job {request.work_info.id}: {error}'
                    )
                    await self._send_failure_toast(request.work_info, error)
                finally:
                    self._pending.pop(request.request_id, None)
            except asyncio.CancelledError:
                self.logger.info('Background job submission worker cancelled')
                break
            except Exception as error:
                self.logger.error(f'Unexpected error in submission worker: {error}')
                await asyncio.sleep(1)
            finally:
                if request is not None:
                    self._queue.task_done()

    async def persist(self, work_info: WorkInfo, overwrite: bool = True) -> str:
        submission_id = work_info.id
        policy = ExistingWorkPolicy.create(
            work_info.policy,
            default_policy=ExistingWorkPolicy.REJECT_DUPLICATE,
        )
        if not await self.is_valid_submission(work_info, policy):
            raise ValueError(
                f'Job with submission_id {submission_id} already exists.'
                f'For work item : {work_info}.'
            )

        plan, dag_nodes = query_plan_work_items(work_info)
        scheduler_trace(
            'dag_plan_built',
            job_id=submission_id,
            dag_id=submission_id,
            job_name=work_info.name,
            job_count=len(dag_nodes),
        )
        for dag_work_info in dag_nodes:
            dag_work_info.dag_id = submission_id

        scheduler_trace(
            'dag_persist_start',
            job_id=submission_id,
            dag_id=submission_id,
            job_name=work_info.name,
            job_count=len(dag_nodes),
        )
        new_key_added, new_dag_key = await self.repository.create_dag_with_jobs(
            dag_id=submission_id,
            plan=plan,
            dag_nodes=dag_nodes,
            work_info=work_info,
        )
        if not new_key_added:
            raise ValueError(
                f'Job with submission_id {submission_id} already exists. '
                'Please use a different submission_id.'
            )

        scheduler_trace(
            'dag_persisted',
            job_id=submission_id,
            dag_id=submission_id,
            job_name=work_info.name,
            job_count=len(dag_nodes),
            new_dag_key=new_dag_key,
        )
        await self._dag_admission_callback('submission')
        scheduler_trace(
            'dag_admission_requested',
            job_id=submission_id,
            dag_id=submission_id,
            job_name=work_info.name,
            job_count=len(dag_nodes),
        )
        await self._notify_callback()
        return submission_id

    async def is_valid_submission(
        self,
        work_info: WorkInfo,
        policy: ExistingWorkPolicy,
    ) -> bool:
        try:
            if policy in (
                ExistingWorkPolicy.ALLOW_ALL,
                ExistingWorkPolicy.ALLOW_DUPLICATE,
            ):
                return True
            if policy == ExistingWorkPolicy.REJECT_ALL:
                return False

            metadata = work_info.data.get('metadata', {})
            ref_type = (
                metadata.get('ref_type', '') if isinstance(metadata, dict) else ''
            )
            ref_id = metadata.get('ref_id', '') if isinstance(metadata, dict) else ''
            existing_job = await self.repository.get_job_by_policy(ref_type, ref_id)
            if policy == ExistingWorkPolicy.REJECT_DUPLICATE:
                return existing_job is None
            if policy == ExistingWorkPolicy.REPLACE:
                return not existing_job or (
                    existing_job.state is not None and existing_job.state.is_terminal()
                )
            raise ValueError(f'Unsupported policy: {policy}')
        except Exception as error:
            self.logger.error(
                f"Error validating submission for work '{work_info.name}' "
                f"with policy '{policy}': {error}"
            )
            return False

    def abort_pending(self) -> None:
        while True:
            try:
                request = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            self._pending.pop(request.request_id, None)
            if request.wait_for_result and not request.result_future.done():
                request.result_future.set_exception(
                    RuntimeError('Scheduler stopped before submission was processed')
                )
            self._queue.task_done()

    def status(self, worker_tasks: Iterable[asyncio.Task[Any]]) -> dict[str, Any]:
        tasks = list(worker_tasks)
        active_workers = sum(1 for task in tasks if not task.done())
        total_workers = len(tasks)
        return {
            'queue_size': self.queue_size,
            'queue_capacity': self._queue.maxsize,
            'pending_requests': self.pending_count,
            'total_submissions': self.submission_count,
            'workers': {
                'total': total_workers,
                'active': active_workers,
                'utilization': (
                    f'{(active_workers / total_workers) * 100:.1f}%'
                    if total_workers > 0
                    else '0%'
                ),
            },
        }

    async def _ensure_queue(self, queue_name: str) -> None:
        if queue_name in self.known_queues:
            return
        self.logger.info(f'Checking for queue: {queue_name}')
        await self.repository.create_queue(queue_name)
        await self.repository.create_queue(f'${queue_name}_dlq')
        self.known_queues.add(queue_name)

    async def _send_failure_toast(
        self,
        work_info: WorkInfo,
        error: Exception,
    ) -> None:
        try:
            event_name = work_info.data.get('name', work_info.name)
            api_key = work_info.data.get('api_key')
            metadata = work_info.data.get('metadata', {})
            metadata = metadata if isinstance(metadata, dict) else {}
            ref_type = metadata.get('ref_type')
            if not api_key or not event_name:
                self.logger.warning(
                    f'Cannot send failure toast for {work_info.id}: '
                    f'missing api_key={api_key} or event_name={event_name}'
                )
                return

            await mark_as_failed_toast(
                api_key=api_key,
                job_id=work_info.id,
                event_name=event_name,
                job_tag=ref_type,
                status='FAILED',
                timestamp=current_milli_time(),
                payload={**metadata, 'error': str(error)},
            )
        except Exception as toast_error:
            self.logger.error(
                f'Failed to send failure toast for {work_info.id}: {toast_error}'
            )

    async def _send_scheduled_toast(self, work_info: WorkInfo) -> None:
        event_name = work_info.data.get('name', work_info.name)
        api_key = work_info.data.get('api_key')
        metadata = work_info.data.get('metadata', {})
        metadata = metadata if isinstance(metadata, dict) else {}
        ref_type = metadata.get('ref_type')
        if not api_key or not event_name:
            self.logger.warning(
                f'Cannot send scheduled toast for {work_info.id}: '
                f'missing api_key={api_key} or event_name={event_name}'
            )
            return

        published = await mark_as_scheduled_toast(
            api_key=api_key,
            job_id=work_info.id,
            event_name=event_name,
            job_tag=ref_type,
            status='OK',
            timestamp=current_milli_time(),
            payload=metadata,
        )
        if not published:
            self.logger.warning(f'Failed to send scheduled toast for {work_info.id}')
