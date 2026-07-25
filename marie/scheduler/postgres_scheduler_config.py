from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from marie.excepts import BadConfigSource
from marie.query_planner.model import QueryPlannersConf
from marie.scheduler.models import HeartbeatConfig

_HARD_SLA_POLICIES = {
    'track_only',
    'escalate_only',
}


@dataclass(frozen=True, slots=True)
class PostgreSQLSchedulerConfig:
    """Validated runtime settings for the PostgreSQL scheduler."""

    queue_names: frozenset[str]
    scheduler_mode: str
    max_workers: int
    submission_queue_size: int
    job_event_worker_count: int
    job_event_queue_size: int
    sla_priority_interval_seconds: int
    max_concurrent_dags: int
    dag_resolution_retry_limit: int
    dag_resolution_retry_delay: float
    dag_resolution_retry_backoff: bool
    dag_resolution_retry_max_delay: float
    maintenance_interval: int
    query_planners: QueryPlannersConf
    dag_cache_size: int
    heartbeat: HeartbeatConfig
    hard_sla_policy: str
    invalid_hard_sla_policy: str | None
    sla_warning_top_n: int
    priority_refresh_interval: int
    priority_refresh_interval_seconds: float
    priority_refresh_timeout_seconds: float
    priority_refresh_hydrate_limit: int
    frontier_batch_size: int
    lease_ttl_seconds: int
    run_ttl_seconds: int
    run_lease_renewal_interval_seconds: float
    gateway_instance_id: str | None

    @classmethod
    def from_dict(cls, config: Mapping[str, Any]) -> PostgreSQLSchedulerConfig:
        if 'queue_names' not in config:
            raise BadConfigSource('Missing required config: queue_names')

        queue_names = config.get('queue_names')
        if not isinstance(queue_names, (list, tuple, set, frozenset)):
            raise BadConfigSource('queue_names must be a non-empty collection')
        if not all(isinstance(name, str) and name for name in queue_names):
            raise BadConfigSource('queue_names must contain non-empty strings')
        normalized_queues = frozenset(queue_names)
        if not normalized_queues:
            raise BadConfigSource('Queue names are required for JobScheduler')

        distributed_scheduler = config.get('distributed_scheduler', True)
        if distributed_scheduler is not True:
            raise BadConfigSource(
                'distributed_scheduler=false is no longer supported; '
                'durable DB leasing is required.'
            )

        dag_config = config.get('dag_manager', {})
        if not isinstance(dag_config, Mapping):
            raise BadConfigSource('dag_manager must be a mapping')
        query_planners_config = config.get('query_planners', {})
        if not isinstance(query_planners_config, Mapping):
            raise BadConfigSource('query_planners must be a mapping')
        heartbeat_config = config.get('heartbeat', {})
        if not isinstance(heartbeat_config, Mapping):
            raise BadConfigSource('heartbeat must be a mapping')

        hard_sla_policy = str(config.get('hard_sla_policy', 'track_only')).lower()
        if hard_sla_policy == 'expire_unfinished':
            raise BadConfigSource(
                'hard_sla_policy=expire_unfinished is not implemented'
            )
        invalid_hard_sla_policy = None
        if hard_sla_policy not in _HARD_SLA_POLICIES:
            invalid_hard_sla_policy = hard_sla_policy
            hard_sla_policy = 'track_only'

        try:
            run_ttl_seconds = int(config.get('run_ttl_seconds', 60))
            renewal_interval = config.get('run_lease_renewal_interval_seconds')
            if renewal_interval is None:
                renewal_interval = min(20.0, run_ttl_seconds / 3.0)

            settings = cls(
                queue_names=normalized_queues,
                scheduler_mode=str(config.get('scheduler_mode', 'parallel')),
                max_workers=int(config.get('max_workers', 5)),
                submission_queue_size=int(config.get('submission_queue_size', 1000)),
                job_event_worker_count=int(config.get('job_event_worker_count', 8)),
                job_event_queue_size=int(config.get('job_event_queue_size', 1024)),
                sla_priority_interval_seconds=max(
                    1, int(config.get('sla_priority_interval_seconds', 15 * 60))
                ),
                max_concurrent_dags=int(dag_config.get('max_concurrent_dags', 16)),
                dag_resolution_retry_limit=int(
                    dag_config.get('dag_resolution_retry_limit', 3)
                ),
                dag_resolution_retry_delay=float(
                    dag_config.get('dag_resolution_retry_delay', 1.0)
                ),
                dag_resolution_retry_backoff=bool(
                    dag_config.get('dag_resolution_retry_backoff', True)
                ),
                dag_resolution_retry_max_delay=float(
                    dag_config.get('dag_resolution_retry_max_delay', 30.0)
                ),
                maintenance_interval=int(config.get('maintenance_interval', 60)),
                query_planners=QueryPlannersConf.from_dict(dict(query_planners_config)),
                dag_cache_size=int(dag_config.get('dag_cache_size', 5000)),
                heartbeat=HeartbeatConfig.from_dict(dict(heartbeat_config)),
                hard_sla_policy=hard_sla_policy,
                invalid_hard_sla_policy=invalid_hard_sla_policy,
                sla_warning_top_n=int(config.get('sla_warning_top_n', 5)),
                priority_refresh_interval=int(
                    config.get('priority_refresh_interval', 10)
                ),
                priority_refresh_interval_seconds=float(
                    config.get('priority_refresh_interval_seconds', 5.0)
                ),
                priority_refresh_timeout_seconds=max(
                    0.1,
                    float(config.get('priority_refresh_timeout_seconds', 30.0)),
                ),
                priority_refresh_hydrate_limit=int(
                    config.get('priority_refresh_hydrate_limit', 100)
                ),
                frontier_batch_size=int(dag_config.get('frontier_batch_size', 1000)),
                lease_ttl_seconds=int(config.get('lease_ttl_seconds', 5)),
                run_ttl_seconds=run_ttl_seconds,
                run_lease_renewal_interval_seconds=float(renewal_interval),
                gateway_instance_id=(
                    str(config['gateway_instance_id'])
                    if config.get('gateway_instance_id')
                    else None
                ),
            )
        except (TypeError, ValueError) as error:
            raise BadConfigSource(
                f'Invalid scheduler configuration: {error}'
            ) from error

        settings._validate_ranges()
        return settings

    def _validate_ranges(self) -> None:
        if self.max_workers < 0:
            raise BadConfigSource('max_workers must be zero or greater')
        if self.submission_queue_size <= 0:
            raise BadConfigSource('submission_queue_size must be greater than zero')
        if self.job_event_worker_count <= 0:
            raise BadConfigSource('job_event_worker_count must be greater than zero')
        if self.job_event_queue_size < self.job_event_worker_count:
            raise BadConfigSource(
                'job_event_queue_size must be at least job_event_worker_count'
            )
        if self.max_concurrent_dags <= 0:
            raise BadConfigSource(
                'dag_manager.max_concurrent_dags must be greater than zero'
            )
        if self.priority_refresh_interval <= 0:
            raise BadConfigSource('priority_refresh_interval must be greater than zero')
        if self.priority_refresh_interval_seconds <= 0:
            raise BadConfigSource(
                'priority_refresh_interval_seconds must be greater than zero'
            )
        if self.frontier_batch_size <= 0:
            raise BadConfigSource(
                'dag_manager.frontier_batch_size must be greater than zero'
            )
        if self.lease_ttl_seconds <= 0 or self.run_ttl_seconds <= 0:
            raise BadConfigSource('lease and run TTL values must be greater than zero')
        if self.run_lease_renewal_interval_seconds <= 0:
            raise BadConfigSource(
                'run_lease_renewal_interval_seconds must be greater than zero'
            )
        if self.run_lease_renewal_interval_seconds > self.run_ttl_seconds / 3:
            raise BadConfigSource(
                'run_lease_renewal_interval_seconds must not exceed one-third '
                'of run_ttl_seconds'
            )
