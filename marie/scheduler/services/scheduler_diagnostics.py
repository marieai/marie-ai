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
