from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from marie.scheduler.memory_frontier import MemoryFrontier
from marie.scheduler.repository import JobRepository
from marie.scheduler.services.dag_submission_service import DagSubmissionService


class SchedulerDiagnostics:
    """Build scheduler debug snapshots from live component state."""

    def __init__(
        self,
        *,
        repository: JobRepository,
        frontier: MemoryFrontier,
        submission_service: DagSubmissionService,
        event_queue: asyncio.Queue[Any],
        active_dags: dict[str, Any],
        known_queues: set[str],
        scheduling_engine: Any,
        gateway_instance_id: str,
        lease_owner: str,
        max_concurrent_dags: int,
        start_time: datetime,
        sla_warning_top_n: int,
        frontier_batch_size: int,
        lease_ttl_seconds: int,
    ) -> None:
        self.repository = repository
        self.frontier = frontier
        self.submission_service = submission_service
        self.event_queue = event_queue
        self.active_dags = active_dags
        self.known_queues = known_queues
        self.scheduling_engine = scheduling_engine
        self.gateway_instance_id = gateway_instance_id
        self.lease_owner = lease_owner
        self.max_concurrent_dags = max_concurrent_dags
        self.start_time = start_time
        self.sla_warning_top_n = sla_warning_top_n
        self.frontier_batch_size = frontier_batch_size
        self.lease_ttl_seconds = lease_ttl_seconds
        self._job_state_counts: dict[str, Any] = {'queues': {}}
        self._dag_state_counts: dict[str, Any] = {'queues': {}}

    async def jobs(
        self,
        *,
        limit: int,
        offset: int,
        states: list[str] | None,
        attention: str,
        queue: str | None,
        search: str | None,
        sort: str,
    ) -> dict[str, Any]:
        return await self.repository.list_operational_jobs(
            limit=limit,
            offset=offset,
            states=states,
            attention=attention,
            queue=queue,
            search=search,
            sort=sort,
        )

    async def job(self, job_id: str) -> dict[str, Any] | None:
        return await self.repository.get_operational_job(job_id)

    async def execution_history(
        self,
        *,
        job_id: str | None = None,
        dag_id: str | None = None,
        limit: int,
        offset: int,
    ) -> dict[str, Any] | None:
        return await self.repository.list_operational_execution_history(
            job_id=job_id,
            dag_id=dag_id,
            limit=limit,
            offset=offset,
        )

    async def attempts(
        self,
        *,
        limit: int,
        offset: int,
        states: list[str] | None,
        attention: str,
        gateway: str | None,
        executor: str | None,
        search: str | None,
        sort: str,
    ) -> dict[str, Any]:
        return await self.repository.list_operational_attempts(
            limit=limit,
            offset=offset,
            states=states,
            attention=attention,
            gateway=gateway,
            executor=executor,
            search=search,
            sort=sort,
        )

    async def events(
        self,
        *,
        limit: int,
        before_at: datetime | None,
        before_id: str | None,
        window_seconds: int,
        severity: str | None,
        component: str | None,
        search: str | None,
    ) -> dict[str, Any]:
        return await self.repository.list_operational_events(
            limit=limit,
            before_at=before_at,
            before_id=before_id,
            window_seconds=window_seconds,
            severity=severity,
            component=component,
            search=search,
        )

    async def flow(
        self,
        *,
        window_seconds: int,
        queue: str | None,
        queue_limit: int,
    ) -> dict[str, Any]:
        return await self.repository.get_operational_flow(
            window_seconds=window_seconds,
            queue=queue,
            queue_limit=queue_limit,
        )

    async def database_health(self) -> dict[str, Any]:
        return await self.repository.get_operational_database_health()

    async def dags(
        self,
        *,
        limit: int,
        offset: int,
        states: list[str] | None,
        attention: str,
        queue: str | None,
        search: str | None,
        sort: str,
    ) -> dict[str, Any]:
        return await self.repository.list_operational_dags(
            limit=limit,
            offset=offset,
            states=states,
            attention=attention,
            queue=queue,
            search=search,
            sort=sort,
        )

    async def dag(
        self, dag_id: str, *, job_limit: int, job_offset: int
    ) -> dict[str, Any] | None:
        return await self.repository.get_operational_dag(
            dag_id,
            job_limit=job_limit,
            job_offset=job_offset,
        )

    async def throughput(
        self,
        *,
        lookback_hours: int,
        planner: str | None,
        planner_limit: int,
        task_limit: int,
    ) -> dict[str, Any]:
        return await self.repository.get_operational_throughput(
            lookback_hours=lookback_hours,
            planner=planner,
            planner_limit=planner_limit,
            task_limit=task_limit,
        )

    async def snapshot(
        self,
        *,
        running: bool,
        paused: bool,
        fetch_counter: int,
    ) -> dict[str, Any]:
        current_time = datetime.now(timezone.utc)
        job_counts, dag_counts = await asyncio.gather(
            self.repository.count_job_states(),
            self.repository.count_dag_states(),
            return_exceptions=True,
        )
        job_error = self._update_counts(job_counts, job=True)
        dag_error = self._update_counts(dag_counts, job=False)

        debug_data: dict[str, Any] = {
            'scheduler_info': {
                'running': running,
                'paused': paused,
                'gateway_instance_id': self.gateway_instance_id,
                'scheduler_lease_owner': self.lease_owner,
                'max_concurrent_dags': self.max_concurrent_dags,
                'known_queues': list(self.known_queues),
                'active_dags_count': len(self.active_dags),
            },
            'timing_info': {
                'current_time': current_time.isoformat(),
                'start_time': self.start_time.isoformat(),
                'uptime_seconds': (current_time - self.start_time).total_seconds(),
                'uptime_human': str(current_time - self.start_time),
            },
            'counters': {
                'fetch_counter': fetch_counter,
                'submission_count': self.submission_service.submission_count,
            },
            'queues': {
                'event_queue_size': self.event_queue.qsize(),
            },
            'execution_planning': {
                'scheduling_engine_available': self.scheduling_engine is not None,
            },
            'sla_monitoring': {
                'warning_top_n': self.sla_warning_top_n,
            },
            'job_state_counts': self._job_state_counts,
            'dag_state_counts': self._dag_state_counts,
        }
        selection_diagnostics = getattr(self.scheduling_engine, 'diagnostics', None)
        if callable(selection_diagnostics):
            try:
                debug_data['selection'] = selection_diagnostics()
            except Exception as error:
                debug_data['selection_error'] = type(error).__name__
        if job_error:
            debug_data['job_state_counts_error'] = job_error
        if dag_error:
            debug_data['dag_state_counts_error'] = dag_error

        if self.active_dags:
            debug_data['active_dags'] = {
                dag_id: {
                    'dag_id': dag_id,
                    'status': self._dag_status(dag_info),
                }
                for dag_id, dag_info in self.active_dags.items()
            }

        try:
            debug_data['frontier_summary'] = self.frontier.summary(detail=True)
        except Exception as error:
            debug_data['frontier_summary_error'] = str(error)

        debug_data['frontier'] = self._frontier_info()
        return debug_data

    def _update_counts(self, result: Any, *, job: bool) -> str | None:
        if isinstance(result, Exception):
            return str(result)
        if job:
            self._job_state_counts = result
        else:
            self._dag_state_counts = result
        return None

    @staticmethod
    def _dag_status(dag_info: Any) -> Any:
        try:
            return dag_info.status
        except Exception:
            return 'unknown'

    def _frontier_info(self) -> dict[str, Any]:
        info: dict[str, Any] = {'available': self.frontier is not None}
        if self.frontier is None:
            return info
        try:
            info['summary'] = self.frontier.summary(detail=True, top_n=10)
        except TypeError:
            try:
                info['summary'] = self.frontier.summary(detail=True)
            except Exception as error:
                info['summary_error'] = str(error)
        except Exception as error:
            info['summary_error'] = str(error)
        info['batch_size'] = self.frontier_batch_size
        info['lease_ttl_seconds'] = self.lease_ttl_seconds
        return info
