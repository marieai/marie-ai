from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from marie.job.common import JobStatus
from marie.logging_core.logger import MarieLogger
from marie.scheduler.job_lock import AsyncJobLock
from marie.scheduler.memory_frontier import MemoryFrontier
from marie.scheduler.models import WorkInfo
from marie.scheduler.repository import JobRepository
from marie.scheduler.services.control_flow_execution_service import (
    ControlFlowExecutionService,
)
from marie.scheduler.services.dag_management_service import DAGManagementService
from marie.scheduler.state import WorkState
from marie.utils.scheduler_trace import scheduler_trace

DEFAULT_SCHEMA = 'marie_scheduler'
TERMINAL_EVENT_STALE_ATTEMPT_TOTAL = 'terminal_event_stale_attempt_total'


class AttemptLifecycleService:
    """Apply fenced terminal transitions and reconcile their in-memory state."""

    def __init__(
        self,
        repository: JobRepository,
        frontier: MemoryFrontier,
        dag_service: DAGManagementService,
        control_flow_service: ControlFlowExecutionService,
        status_update_lock: AsyncJobLock,
        job_cache: dict[str, WorkInfo],
        scheduler_lease_owner: str,
        gateway_instance_id: str,
        notify_callback: Callable[[], Awaitable[bool]],
        counter_callback: Callable[..., None],
    ) -> None:
        self.logger = MarieLogger(AttemptLifecycleService.__name__)
        self.repository = repository
        self.frontier = frontier
        self.dag_service = dag_service
        self.control_flow_service = control_flow_service
        self._status_update_lock = status_update_lock
        self._job_cache = job_cache
        self._scheduler_lease_owner = scheduler_lease_owner
        self._gateway_instance_id = gateway_instance_id
        self._notify_callback = notify_callback
        self._counter_callback = counter_callback

    async def transition_terminal(
        self,
        job_id: str,
        work_item: WorkInfo,
        status: JobStatus,
        *,
        run_owner: str | None,
        run_attempt_id: str | None,
        source: str,
        output_metadata: dict[str, Any] | None = None,
        message: Any = None,
        runtime_env: Any = None,
    ) -> bool:
        """Apply one terminal event if it matches the active durable attempt."""
        if not run_owner or not run_attempt_id:
            self._trace_missing_attempt(job_id, work_item, status, source)
            return False

        metadata = dict(output_metadata or {})
        if status == JobStatus.SUCCEEDED:
            accepted = await self._complete(
                job_id,
                work_item,
                run_owner,
                run_attempt_id,
                source,
                metadata,
            )
        elif status == JobStatus.FAILED:
            metadata.update(
                self._failure_metadata(
                    message=message,
                    runtime_env=runtime_env,
                    source=source,
                )
            )
            accepted = await self._fail(
                job_id,
                work_item,
                run_owner,
                run_attempt_id,
                source,
                metadata,
            )
        elif status == JobStatus.STOPPED:
            accepted = await self._stop(
                job_id,
                work_item,
                run_owner,
                run_attempt_id,
                source,
            )
        else:
            raise ValueError(f'Expected terminal status, received {status}')

        return accepted

    async def _complete(
        self,
        job_id: str,
        work_item: WorkInfo,
        run_owner: str,
        run_attempt_id: str,
        source: str,
        output_metadata: dict[str, Any],
    ) -> bool:
        guardrail_commit = (
            await self.control_flow_service.commit_guardrail_route_if_needed(
                job_id,
                work_item,
                run_owner=run_owner,
                run_attempt_id=run_attempt_id,
            )
        )
        if guardrail_commit is None:
            async with self._status_update_lock[job_id]:
                completed = await self.repository.complete_job(
                    job_id=job_id,
                    queue_name=work_item.name,
                    output_metadata=output_metadata,
                    schema=DEFAULT_SCHEMA,
                    run_owner=run_owner,
                    run_attempt_id=run_attempt_id,
                )
            reject_reason = 'db_update_zero_rows'
        else:
            committed, _, reject_reason = guardrail_commit
            completed = int(committed)

        if completed <= 0:
            await self._reject(
                job_id,
                work_item,
                JobStatus.SUCCEEDED,
                run_owner,
                run_attempt_id,
                source,
                reject_reason or 'db_update_zero_rows',
            )
            return False

        work_item.state = WorkState.COMPLETED
        self._job_cache[job_id] = work_item
        await self._accept(
            job_id,
            work_item,
            JobStatus.SUCCEEDED,
            run_owner,
            run_attempt_id,
            WorkState.COMPLETED,
            source,
        )
        if guardrail_commit is None:
            await self.control_flow_service.handle_successful_job_completion(
                job_id, work_item
            )
        await self._finish(job_id, work_item, source, terminal=True)
        return True

    async def _fail(
        self,
        job_id: str,
        work_item: WorkInfo,
        run_owner: str,
        run_attempt_id: str,
        source: str,
        output_metadata: dict[str, Any],
    ) -> bool:
        self.logger.error(
            f'Job failure received: job_id={job_id} '
            f'dag_id={work_item.dag_id} details={output_metadata}'
        )
        async with self._status_update_lock[job_id]:
            _, final_state = await self.repository.fail_job(
                job_id=job_id,
                queue_name=work_item.name,
                output_metadata=output_metadata,
                schema=DEFAULT_SCHEMA,
                run_owner=run_owner,
                run_attempt_id=run_attempt_id,
            )

        if final_state is None:
            await self._reject(
                job_id,
                work_item,
                JobStatus.FAILED,
                run_owner,
                run_attempt_id,
                source,
                'db_update_zero_rows',
            )
            return False

        work_state = WorkState(final_state)
        work_item.state = work_state
        self._job_cache[job_id] = work_item
        await self._accept(
            job_id,
            work_item,
            JobStatus.FAILED,
            run_owner,
            run_attempt_id,
            work_state,
            source,
        )
        if work_state == WorkState.RETRY:
            await self.frontier.on_job_retry(job_id, work_item)
        else:
            await self.frontier.on_job_failed(job_id)
        await self._finish(
            job_id,
            work_item,
            source,
            terminal=work_state == WorkState.FAILED,
        )
        return True

    async def _stop(
        self,
        job_id: str,
        work_item: WorkInfo,
        run_owner: str,
        run_attempt_id: str,
        source: str,
    ) -> bool:
        async with self._status_update_lock[job_id]:
            cancelled_ids = await self.repository.cancel_job_attempt(
                job_id=job_id,
                queue_name=work_item.name,
                run_owner=run_owner,
                run_attempt_id=run_attempt_id,
                schema=DEFAULT_SCHEMA,
            )

        if job_id not in cancelled_ids:
            await self._reject(
                job_id,
                work_item,
                JobStatus.STOPPED,
                run_owner,
                run_attempt_id,
                source,
                'db_update_zero_rows',
            )
            return False

        work_item.state = WorkState.CANCELLED
        work_item.run_owner = None
        work_item.run_attempt_id = None
        self._job_cache[job_id] = work_item
        await self._accept(
            job_id,
            work_item,
            JobStatus.STOPPED,
            run_owner,
            run_attempt_id,
            WorkState.CANCELLED,
            source,
        )
        await self.frontier.on_job_cancelled(job_id)
        await self._finish(job_id, work_item, source, terminal=True)
        return True

    async def _finish(
        self,
        job_id: str,
        work_item: WorkInfo,
        source: str,
        *,
        terminal: bool,
    ) -> None:
        if terminal:
            self._status_update_lock.release(job_id)
            await self.dag_service.resolve_dag_status_with_retry(
                job_id,
                work_item,
                source=source,
            )
        await self._notify_callback()

    async def _accept(
        self,
        job_id: str,
        work_item: WorkInfo,
        status: JobStatus,
        run_owner: str,
        run_attempt_id: str,
        work_state: WorkState,
        source: str,
    ) -> None:
        await self._record_audit(
            job_id=job_id,
            work_item=work_item,
            status=status,
            run_owner=run_owner,
            run_attempt_id=run_attempt_id,
            terminal_work_state=work_state.value,
            source=source,
            accepted=True,
        )
        scheduler_trace(
            'job_terminal_attempt_accepted',
            job_id=job_id,
            dag_id=work_item.dag_id,
            status=status.value,
            run_owner=run_owner,
            run_attempt_id=run_attempt_id,
            final_state=work_state.value,
            source=source,
            **self._ha_trace_fields(),
        )

    async def _reject(
        self,
        job_id: str,
        work_item: WorkInfo,
        status: JobStatus,
        run_owner: str,
        run_attempt_id: str,
        source: str,
        reason: str,
    ) -> None:
        scheduler_trace(
            'job_terminal_attempt_rejected',
            job_id=job_id,
            dag_id=work_item.dag_id,
            status=status.value,
            run_owner=run_owner,
            run_attempt_id=run_attempt_id,
            reason=reason,
            source=source,
            **self._ha_trace_fields(),
        )
        await self._record_audit(
            job_id=job_id,
            work_item=work_item,
            status=status,
            run_owner=run_owner,
            run_attempt_id=run_attempt_id,
            terminal_work_state=None,
            source=source,
            accepted=False,
            reject_reason=reason,
        )
        self._counter_callback(
            TERMINAL_EVENT_STALE_ATTEMPT_TOTAL,
            job_id=job_id,
            dag_id=work_item.dag_id,
            status=status.value,
            run_owner=run_owner,
            run_attempt_id=run_attempt_id,
            source=source,
        )

    async def _record_audit(
        self,
        *,
        job_id: str,
        work_item: WorkInfo,
        status: JobStatus,
        run_owner: str,
        run_attempt_id: str,
        terminal_work_state: str | None,
        source: str,
        accepted: bool,
        reject_reason: str | None = None,
    ) -> None:
        try:
            await self.repository.record_job_attempt_terminal(
                job_id=job_id,
                job_name=work_item.name,
                dag_id=str(work_item.dag_id),
                run_owner=run_owner,
                run_attempt_id=run_attempt_id,
                scheduler_lease_owner=self._scheduler_lease_owner,
                gateway_instance_id=self._gateway_instance_id,
                terminal_status=status.value,
                terminal_work_state=terminal_work_state,
                source=source,
                accepted=accepted,
                reject_reason=reject_reason,
            )
        except Exception as audit_error:
            scheduler_trace(
                'job_attempt_audit_failed',
                job_id=job_id,
                dag_id=work_item.dag_id,
                run_owner=run_owner,
                run_attempt_id=run_attempt_id,
                source=source,
                accepted=accepted,
                error=repr(audit_error),
                **self._ha_trace_fields(),
            )
            self.logger.warning(
                f'Failed to record terminal audit for attempt '
                f'{run_attempt_id}: {audit_error}'
            )

    def _trace_missing_attempt(
        self,
        job_id: str,
        work_item: WorkInfo,
        status: JobStatus,
        source: str,
    ) -> None:
        scheduler_trace(
            'job_terminal_attempt_rejected',
            job_id=job_id,
            dag_id=work_item.dag_id,
            status=status.value,
            reason='missing_attempt',
            source=source,
            **self._ha_trace_fields(),
        )
        self.logger.warning(
            f'Ignoring {status.value.lower()} event without run attempt: '
            f'job_id={job_id}'
        )

    def _ha_trace_fields(self) -> dict[str, str]:
        return {
            'gateway_instance_id': self._gateway_instance_id,
            'scheduler_lease_owner': self._scheduler_lease_owner,
        }

    @staticmethod
    def _failure_metadata(
        *, message: Any, runtime_env: Any, source: str
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {'failure_source': source}
        if message:
            metadata['error_message'] = str(message)
        if not isinstance(runtime_env, dict):
            return metadata

        error = runtime_env.get('error')
        if isinstance(error, dict):
            metadata['error'] = {
                key: error[key]
                for key in ('type', 'message', 'filename', 'name', 'line_no')
                if key in error
            }
        elif error:
            metadata['error'] = str(error)
        return metadata
