from __future__ import annotations

from collections.abc import Awaitable, Callable

from marie.logging_core.logger import MarieLogger
from marie.messaging import mark_as_failed as mark_as_failed_toast
from marie.messaging import mark_as_scheduled as mark_as_scheduled_toast
from marie.scheduler.models import ExistingWorkPolicy, WorkInfo
from marie.scheduler.planner_util import query_plan_work_items
from marie.scheduler.repository import JobRepository
from marie.utils.scheduler_trace import scheduler_trace
from marie.utils.utils import current_milli_time


class DagSubmissionService:
    """Persist and account for submitted DAGs."""

    def __init__(
        self,
        *,
        repository: JobRepository,
        dag_admission_callback: Callable[[str], Awaitable[bool]],
        known_queues: set[str],
        notify_callback: Callable[[], Awaitable[bool]],
        is_running: Callable[[], bool],
        submission_processed_callback: Callable[[int], Awaitable[None]] | None,
        logger: MarieLogger,
        initial_submission_count: int = 0,
    ) -> None:
        self.repository = repository
        self._dag_admission_callback = dag_admission_callback
        self.known_queues = known_queues
        self._notify_callback = notify_callback
        self._is_running = is_running
        self._submission_processed_callback = submission_processed_callback
        self.logger = logger
        self.submission_count = initial_submission_count

    async def submit(self, work_info: WorkInfo, overwrite: bool = True) -> str:
        self.logger.debug(f'Submitting job : {work_info.id}')
        try:
            if not self._is_running():
                raise RuntimeError('Job scheduler is not running')
            await self._ensure_queue(work_info.name)
            if not self._is_running():
                raise RuntimeError('Job scheduler stopped during submission')
            result = await self.persist(work_info, overwrite)
        except Exception as error:
            self.logger.error(f'Job submission failed for {work_info.id}: {error}')
            await self._send_failure_toast(work_info, error)
            raise

        await self._send_scheduled_toast(work_info)
        self.submission_count += 1
        if self._submission_processed_callback is not None:
            try:
                await self._submission_processed_callback(self.submission_count)
            except Exception as error:
                self.logger.error(
                    f'Failed to record submission count for {work_info.id}: {error}'
                )
        self.logger.debug(f'Successfully persisted job: {work_info.id}')
        return result

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
        try:
            await self._dag_admission_callback('submission')
            scheduler_trace(
                'dag_admission_requested',
                job_id=submission_id,
                dag_id=submission_id,
                job_name=work_info.name,
                job_count=len(dag_nodes),
            )
        except Exception as error:
            self.logger.error(
                f'Failed to request admission for durable DAG {submission_id}: {error}'
            )
        try:
            await self._notify_callback()
        except Exception as error:
            self.logger.error(
                f'Failed to notify scheduler for durable DAG {submission_id}: {error}'
            )
        return submission_id

    async def is_valid_submission(
        self,
        work_info: WorkInfo,
        policy: ExistingWorkPolicy,
    ) -> bool:
        if policy in (
            ExistingWorkPolicy.ALLOW_ALL,
            ExistingWorkPolicy.ALLOW_DUPLICATE,
        ):
            return True
        if policy == ExistingWorkPolicy.REJECT_ALL:
            return False

        metadata = work_info.data.get('metadata', {})
        ref_type = metadata.get('ref_type', '') if isinstance(metadata, dict) else ''
        ref_id = metadata.get('ref_id', '') if isinstance(metadata, dict) else ''
        existing_job = await self.repository.get_job_by_policy(ref_type, ref_id)
        if policy == ExistingWorkPolicy.REJECT_DUPLICATE:
            return existing_job is None
        if policy == ExistingWorkPolicy.REPLACE:
            return not existing_job or (
                existing_job.state is not None and existing_job.state.is_terminal()
            )
        raise ValueError(f'Unsupported policy: {policy}')

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
        try:
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
                self.logger.warning(
                    f'Failed to send scheduled toast for {work_info.id}'
                )
        except Exception as toast_error:
            self.logger.error(
                f'Failed to send scheduled toast for {work_info.id}: {toast_error}'
            )
