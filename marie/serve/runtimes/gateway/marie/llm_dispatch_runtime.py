from __future__ import annotations

import asyncio
import os
from dataclasses import replace
from typing import Any, Optional

from marie.engine.llm_queue.config import LlmQueueRuntimeConfig, resolve_fabric_id
from marie.engine.llm_queue.scheduler_config import (
    DatabaseSchedulerConfigSource,
    LlmQueueSchedulerConfig,
    SchedulerConfigSource,
    StaticSchedulerConfigSource,
)

from marie.excepts import RuntimeFailToStart
from marie.logging_core.logger import MarieLogger
from marie.serve.runtimes.gateway.marie.llm_scheduler_config import (
    PostgresSchedulerConfigRepository,
)


class GatewayLlmDispatchRuntime:
    """Gateway-owned lifecycle wrapper for the LLM dispatch runtime."""

    def __init__(
        self,
        *,
        logger: Optional[MarieLogger] = None,
        config: Optional[dict[str, Any]] = None,
        queue_config: Optional[LlmQueueRuntimeConfig] = None,
        scheduler_config_source: Optional[SchedulerConfigSource] = None,
        fabric_group_id: Optional[str] = None,
    ) -> None:
        self.logger = logger or MarieLogger("GatewayLlmDispatchRuntime")
        self.runtime_config = config or {}
        self.config = queue_config or LlmQueueRuntimeConfig.from_env()
        self._fabric_group_id = fabric_group_id
        self._configured_fabric_group_id = self.config.fabric_group_id
        effective_fabric = fabric_group_id or _scheduler_fabric_group_id(
            self.config, self.runtime_config
        )
        self.config = replace(self.config, fabric_group_id=effective_fabric)
        self._scheduler_config_source = scheduler_config_source
        self._queue_client = None
        self._dispatcher = None
        self._last_error: Optional[str] = None
        self._started_pool_ids: list[str] = []
        self._started_scheduler_policy: Optional[str] = None

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    @property
    def mode(self) -> str:
        return "queued-dispatch" if self.enabled else "direct-batch"

    def health(self) -> dict[str, object]:
        if self._dispatcher is not None:
            health = dict(self._dispatcher.health())
            health.setdefault("last_error", self._last_error)
            health.setdefault("mode", self.mode)
            pool_ids = _pool_ids_from_health(health) or list(self._started_pool_ids)
            if pool_ids:
                health.setdefault("pool_ids", pool_ids)
                health.setdefault("pool_count", len(pool_ids))
            if self._started_scheduler_policy:
                health.setdefault("scheduler_policy", self._started_scheduler_policy)
            return health

        return {
            "enabled": self.config.enabled,
            "mode": self.mode,
            "pool_id": self.config.pool_id,
            "pool_ids": [self.config.pool_id],
            "pool_count": 1,
            "queue_configured": bool(self.config.queue_url),
            "running": False,
            "last_error": self._last_error,
        }

    async def start(self) -> None:
        if not self.enabled:
            self.logger.info("LLM dispatch runtime disabled")
            return
        if self._dispatcher is not None:
            self.logger.info("LLM dispatch runtime already started")
            return

        try:
            scheduler = self.runtime_config.get('scheduler') or {}
            effective_fabric = resolve_fabric_id(
                self._fabric_group_id,
                self.runtime_config.get('fabric_group_id'),
                scheduler.get('fabric_group_id'),
                self._configured_fabric_group_id,
                os.getenv('LLM_QUEUE_FABRIC_GROUP_ID'),
            )
            self.config = replace(self.config, fabric_group_id=effective_fabric)
            if self._scheduler_config_source is None:
                self._scheduler_config_source = _build_scheduler_config_source(
                    config=self.runtime_config,
                    fabric_group_id=effective_fabric,
                    default_total_concurrent_dispatch=self.config.max_batch_items,
                    logger=self.logger,
                )
            await self._start_queued_dispatch()
        except Exception:
            self._last_error = 'dispatch_start_failed'
            raise RuntimeFailToStart(
                'LLM dispatch policy, credentials, fabric, or store is unavailable'
            ) from None

    async def _start_queued_dispatch(self) -> None:
        from marie.engine.llm_queue.endpoint import (
            RegisteredEndpoint,
            RegisteredEndpointGroup,
            RegisteredReplica,
        )
        from marie.engine.llm_queue.request_dispatcher import (
            DispatchLane,
            RequestDispatcher,
        )
        from marie.engine.llm_queue.store import RequestStore, StoreLimits

        if not self.config.queue_url:
            raise ValueError('Dispatch requires queue storage')
        from marie.serve.runtimes.gateway.marie.dispatch_policy import (
            persisted_dispatch_policy,
            validate_dispatch_policy,
        )

        configured = dict(self.runtime_config.get('llm_dispatch') or {})
        source = configured.pop('policy_source', 'static')
        if source == 'database':
            if not isinstance(
                self._scheduler_config_source, DatabaseSchedulerConfigSource
            ):
                raise ValueError(
                    'Database policy source requires scheduler persistence'
                )
            if (
                self._scheduler_config_source.fabric_group_id
                != self.config.fabric_group_id
            ):
                raise ValueError('Database policy fabric identity mismatch')
            repository = self._scheduler_config_source.repository
            if hasattr(repository, 'load_runtime_dispatch_policy'):
                policy = await asyncio.to_thread(
                    repository.load_runtime_dispatch_policy,
                    self.config.fabric_group_id,
                )
            else:
                data = await asyncio.to_thread(
                    repository.load_scheduler_config,
                    self.config.fabric_group_id,
                )
                policy = persisted_dispatch_policy(data)
        elif source == 'static':
            policy = validate_dispatch_policy(configured)
        else:
            raise ValueError('Unknown dispatch policy source')

        def resolve_policy(policy: dict) -> dict:
            endpoints = []
            for record in policy['endpoints']:
                values = dict(record)
                credential_env = values.pop('credential_env', None)
                credential = os.getenv(credential_env) if credential_env else None
                if credential_env and not credential:
                    raise ValueError('Registered endpoint credential is unavailable')
                endpoints.append(RegisteredEndpoint(**values, api_key=credential))
            endpoint_groups = []
            for record in policy.get('endpoint_groups', []):
                replicas = []
                for raw_replica in record['replicas']:
                    values = dict(raw_replica)
                    credential_env = values.get('credential_env')
                    credential = os.getenv(credential_env) if credential_env else None
                    if credential_env and not credential:
                        raise ValueError('Registered replica credential is unavailable')
                    replicas.append(RegisteredReplica(**values, api_key=credential))
                endpoint_groups.append(
                    RegisteredEndpointGroup(
                        group_id=record['group_id'],
                        revision=record['revision'],
                        replicas=tuple(replicas),
                    )
                )
            return dict(
                endpoints=endpoints,
                endpoint_groups=endpoint_groups,
                lanes=[DispatchLane(**values) for values in policy['lanes']],
                limits=StoreLimits(**policy['limits']),
                policy=policy['policy'],
                total_concurrent_dispatch=policy['total_concurrent_dispatch'],
                policy_generation=policy.get('policy_generation'),
                policy_digest=policy.get('policy_digest'),
            )

        def load_policy() -> dict:
            repository = self._scheduler_config_source.repository
            if hasattr(repository, 'load_runtime_dispatch_policy'):
                return resolve_policy(
                    repository.load_runtime_dispatch_policy(self.config.fabric_group_id)
                )
            data = repository.load_scheduler_config(self.config.fabric_group_id)
            return resolve_policy(persisted_dispatch_policy(data))

        resolved = resolve_policy(policy)
        endpoints, lanes = resolved['endpoints'], resolved['lanes']
        store = None
        dispatcher = None
        try:
            store = await asyncio.to_thread(
                RequestStore,
                self.config.queue_url,
                fabric_id=self.config.fabric_group_id,
                limits=StoreLimits(**policy['limits']),
            )
            dispatcher = RequestDispatcher(
                store=store,
                endpoints=endpoints or None,
                endpoint_groups=resolved['endpoint_groups'] or None,
                lanes=lanes,
                policy=resolved['policy'],
                total_concurrent_dispatch=resolved['total_concurrent_dispatch'],
                policy_loader=load_policy if source == 'database' else None,
                policy_generation=resolved['policy_generation'],
                policy_digest=resolved['policy_digest'],
            )
            await dispatcher.start()
        except Exception:
            if dispatcher is not None:
                try:
                    await dispatcher.stop()
                except Exception:
                    self.logger.error(
                        'Failed stopping partially started LLM dispatch runtime'
                    )
            if store is not None:
                try:
                    await asyncio.to_thread(store.close)
                except Exception:
                    self.logger.error(
                        'Failed closing store after LLM dispatch startup error'
                    )
            raise
        self._queue_client = store
        self._dispatcher = dispatcher
        self._started_pool_ids = [lane.pool_id for lane in lanes if lane.enabled]
        self._started_scheduler_policy = resolved['policy']
        self._last_error = None

    async def stop(self) -> None:
        dispatcher = self._dispatcher
        store = self._queue_client
        if dispatcher is not None:
            await dispatcher.stop()
        self._dispatcher = None
        self._queue_client = None
        self._started_pool_ids = []
        self._started_scheduler_policy = None
        if store is not None:
            await asyncio.to_thread(store.close)


def _build_scheduler_config_source(
    *,
    config: dict[str, Any],
    fabric_group_id: str,
    default_total_concurrent_dispatch: int,
    logger: Optional[MarieLogger] = None,
) -> SchedulerConfigSource:
    scheduler_repository_config = _scheduler_repository_config(config)
    if scheduler_repository_config:
        return DatabaseSchedulerConfigSource(
            repository=PostgresSchedulerConfigRepository(
                scheduler_repository_config,
                logger=logger,
            ),
            fabric_group_id=fabric_group_id,
            default_total_concurrent_dispatch=default_total_concurrent_dispatch,
        )
    return StaticSchedulerConfigSource(LlmQueueSchedulerConfig())


def _pool_ids_from_health(health: dict[str, object]) -> list[str]:
    lanes = health.get("lanes")
    if not isinstance(lanes, list):
        return []
    pool_ids: list[str] = []
    for lane in lanes:
        if not isinstance(lane, dict):
            continue
        pool_id = lane.get("pool_id")
        if isinstance(pool_id, str) and pool_id:
            pool_ids.append(pool_id)
    return pool_ids


def _scheduler_fabric_group_id(
    queue_config: LlmQueueRuntimeConfig, runtime_config: dict[str, Any]
) -> str:
    scheduler_config = runtime_config.get("scheduler")
    if isinstance(scheduler_config, dict) and scheduler_config.get("fabric_group_id"):
        return str(scheduler_config["fabric_group_id"])
    if runtime_config.get("fabric_group_id"):
        return str(runtime_config["fabric_group_id"])
    return queue_config.fabric_group_id or "default"


def _scheduler_repository_config(config: dict[str, Any]) -> Optional[dict[str, Any]]:
    scheduler_config = config.get("scheduler")
    if not isinstance(scheduler_config, dict):
        return None

    storage_config = scheduler_config.get("storage")
    if isinstance(storage_config, dict) and isinstance(
        storage_config.get("psql"), dict
    ):
        return dict(storage_config["psql"])

    psql_config = scheduler_config.get("psql")
    if isinstance(psql_config, dict):
        return dict(psql_config)

    return None
