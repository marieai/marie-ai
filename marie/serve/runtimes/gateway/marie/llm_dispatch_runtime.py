from __future__ import annotations

import asyncio
import os
from dataclasses import replace
from functools import partial
from typing import Any, Callable, Optional

from marie.engine.llm_queue.config import LlmQueueConfig, resolve_fabric_id
from marie.engine.llm_queue.queue_io import StoreListQueueClient
from marie.engine.llm_queue.scheduler import DrrLaneConfig
from marie.engine.llm_queue.scheduler_config import (
    DatabaseSchedulerConfigSource,
    LlmQueueSchedulerConfig,
    SchedulerConfigSource,
    StaticSchedulerConfigSource,
    ensure_default_pool,
)
from marie.engine.openai_compat import (
    build_async_openai_client,
    resolve_openai_base_url_from_env,
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
        queue_config: Optional[LlmQueueConfig] = None,
        queue_client_factory: Optional[Callable[[str], Any]] = None,
        openai_client_factory: Optional[Callable[[str, Optional[str]], Any]] = None,
        dispatcher_factory: Optional[Callable[..., Any]] = None,
        scheduler_config_source: Optional[SchedulerConfigSource] = None,
        fabric_group_id: Optional[str] = None,
    ) -> None:
        self.logger = logger or MarieLogger("GatewayLlmDispatchRuntime")
        self.runtime_config = config or {}
        self.config = queue_config or LlmQueueConfig.from_env(
            queue_contract_version=self.runtime_config.get('queue_contract_version')
        )
        if self.config.queue_contract_version == 'v3':
            scheduler = self.runtime_config.get('scheduler') or {}
            effective_fabric = resolve_fabric_id(
                fabric_group_id,
                self.runtime_config.get('fabric_group_id'),
                scheduler.get('fabric_group_id'),
                self.config.fabric_group_id,
                os.getenv('LLM_QUEUE_FABRIC_GROUP_ID'),
            )
        else:
            effective_fabric = fabric_group_id or _scheduler_fabric_group_id(
                self.config, self.runtime_config
            )
        self.config = replace(self.config, fabric_group_id=effective_fabric)
        self._queue_client_factory = queue_client_factory or StoreListQueueClient
        self._openai_client_factory = openai_client_factory or build_async_openai_client
        self._dispatcher_factory = dispatcher_factory or _build_dispatcher
        self._scheduler_config_source = (
            scheduler_config_source
            or _build_scheduler_config_source(
                config=self.runtime_config,
                fabric_group_id=self.config.fabric_group_id,
                default_total_concurrent_dispatch=self.config.max_batch_items,
                logger=self.logger,
            )
        )
        self._queue_client = None
        self._dispatcher = None
        self._last_error: Optional[str] = None
        self._started_pool_ids: list[str] = []
        self._started_scheduler_policy: Optional[str] = None

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    def health(self) -> dict[str, object]:
        if self._dispatcher is not None:
            health = dict(self._dispatcher.health())
            health.setdefault("last_error", self._last_error)
            pool_ids = _pool_ids_from_health(health) or list(self._started_pool_ids)
            if pool_ids:
                health.setdefault("pool_ids", pool_ids)
                health.setdefault("pool_count", len(pool_ids))
            if self._started_scheduler_policy:
                health.setdefault("scheduler_policy", self._started_scheduler_policy)
            return health

        return {
            "enabled": self.config.enabled,
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

        if self.config.queue_contract_version == 'v3':
            await self._start_v3()
            return

        queue_client = None
        dispatcher = None
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeFailToStart(
                    "LLM dispatch runtime is enabled but OPENAI_API_KEY is not configured."
                )
            if not self.config.queue_url:
                raise RuntimeFailToStart(
                    "LLM dispatch runtime is enabled but LLM_QUEUE_URL (or LLM_QUEUE_VALKEY_URL) is not configured."
                )

            queue_client = self._queue_client_factory(self.config.queue_url)
            scheduler_config = ensure_default_pool(self._scheduler_config_source.load())
            pool_ids = _runtime_pool_ids(scheduler_config, self.config.pool_id)
            for pool_id in pool_ids:
                queue_client.request_queue_depth(pool_id)

            openai_base_url = resolve_openai_base_url_from_env()
            if isinstance(self._scheduler_config_source, DatabaseSchedulerConfigSource):
                from marie.serve.runtimes.gateway.marie.dispatch_policy import (
                    validate_legacy_lane_endpoints,
                )

                validate_legacy_lane_endpoints(scheduler_config, openai_base_url)

            def client_factory_for_base_url(base_url: Optional[str]):
                return partial(self._openai_client_factory, api_key, base_url)

            dispatcher = self._dispatcher_factory(
                queue_client=queue_client,
                client=None,
                client_factory=client_factory_for_base_url(openai_base_url),
                client_factory_for_base_url=client_factory_for_base_url,
                config=self.config,
                scheduler_config=scheduler_config,
                logger=self.logger,
                backend_address=openai_base_url,
            )
            dispatcher.start()

            self._queue_client = queue_client
            self._dispatcher = dispatcher
            self._started_pool_ids = pool_ids
            self._started_scheduler_policy = scheduler_config.policy
            self._last_error = None
            self.logger.info(
                _format_started_runtime_log(
                    scheduler_config=scheduler_config,
                    fallback_pool_id=self.config.pool_id,
                )
            )
        except Exception as exc:
            self._last_error = 'startup_failed'
            if dispatcher is not None:
                try:
                    dispatcher.stop()
                except Exception:
                    self.logger.exception(
                        "Failed stopping partially started LLM dispatch runtime"
                    )
            if queue_client is not None:
                try:
                    queue_client.close()
                except Exception:
                    self.logger.exception(
                        "Failed closing queue client after LLM dispatch startup error"
                    )
            if isinstance(exc, RuntimeFailToStart):
                raise
            raise RuntimeFailToStart(
                'LLM dispatch startup policy or store unavailable'
            ) from None

    async def _start_v3(self) -> None:
        from marie.engine.llm_queue.endpoint import RegisteredEndpoint
        from marie.engine.llm_queue.request_dispatcher import (
            DispatchLane,
            RequestDispatcher,
        )
        from marie.engine.llm_queue.store import RequestStore, StoreLimits

        if not self.config.queue_url:
            raise RuntimeFailToStart('V3 dispatch requires LLM_QUEUE_URL')
        from marie.serve.runtimes.gateway.marie.dispatch_policy import (
            persisted_dispatch_policy,
            validate_dispatch_policy,
        )

        configured = dict(self.runtime_config.get('llm_dispatch') or {})
        source = configured.pop('policy_source', 'static')
        try:
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
                data = await asyncio.to_thread(
                    self._scheduler_config_source.repository.load_scheduler_config,
                    self.config.fabric_group_id,
                )
                policy = persisted_dispatch_policy(data)
            elif source == 'static':
                policy = validate_dispatch_policy(configured)
            else:
                raise ValueError('Unknown dispatch policy source')
        except Exception:
            self._last_error = 'v3_start_failed'
            raise RuntimeFailToStart(
                'V3 dispatch policy is unavailable or invalid'
            ) from None

        def resolve_policy(policy: dict) -> dict:
            endpoints = []
            for record in policy['endpoints']:
                values = dict(record)
                credential_env = values.pop('credential_env', None)
                credential = os.getenv(credential_env) if credential_env else None
                if credential_env and not credential:
                    raise ValueError('Registered endpoint credential is unavailable')
                endpoints.append(RegisteredEndpoint(**values, api_key=credential))
            return dict(
                endpoints=endpoints,
                lanes=[DispatchLane(**values) for values in policy['lanes']],
                limits=StoreLimits(**policy['limits']),
                policy=policy['policy'],
                total_concurrent_dispatch=policy['total_concurrent_dispatch'],
            )

        def load_policy() -> dict:
            data = self._scheduler_config_source.repository.load_scheduler_config(
                self.config.fabric_group_id
            )
            return resolve_policy(persisted_dispatch_policy(data))

        resolved = resolve_policy(policy)
        endpoints, lanes = resolved['endpoints'], resolved['lanes']
        store = None
        try:
            store = await asyncio.to_thread(
                RequestStore,
                self.config.queue_url,
                fabric_id=self.config.fabric_group_id,
                version='v3',
                limits=StoreLimits(**policy['limits']),
            )
            dispatcher = RequestDispatcher(
                store=store,
                endpoints=endpoints,
                lanes=lanes,
                policy=resolved['policy'],
                total_concurrent_dispatch=resolved['total_concurrent_dispatch'],
                policy_loader=load_policy if source == 'database' else None,
            )
            await dispatcher.start()
        except (ValueError, RuntimeError):
            if store is not None:
                await asyncio.to_thread(store.close)
            self._last_error = 'v3_start_failed'
            raise RuntimeFailToStart(
                'V3 dispatch configuration or store is unavailable'
            ) from None
        self._queue_client = store
        self._dispatcher = dispatcher
        self._started_pool_ids = [lane.pool_id for lane in lanes if lane.enabled]
        self._started_scheduler_policy = resolved['policy']
        self._last_error = None

    async def stop(self) -> None:
        if self.config.queue_contract_version == 'v3':
            dispatcher = self._dispatcher
            if dispatcher is not None:
                await dispatcher.stop()
            self._dispatcher = None
            self._queue_client = None
            return
        dispatcher = self._dispatcher
        queue_client = self._queue_client
        pool_ids = self._started_pool_ids or [self.config.pool_id]
        scheduler_policy = self._started_scheduler_policy or "fifo"

        self._dispatcher = None
        self._queue_client = None
        self._started_pool_ids = []
        self._started_scheduler_policy = None

        if dispatcher is not None:
            try:
                dispatcher.stop()
            except Exception:
                self.logger.exception("Failed stopping LLM dispatch runtime")

        if queue_client is not None:
            try:
                queue_client.close()
            except Exception:
                self.logger.exception("Failed closing LLM dispatch queue client")

        if self.enabled:
            self.logger.info(
                "Stopped LLM %s dispatch runtime for %s",
                scheduler_policy.upper(),
                _describe_pool_scope(pool_ids),
            )


def _build_dispatcher(
    *,
    queue_client,
    client,
    client_factory=None,
    client_factory_for_base_url: Optional[
        Callable[[Optional[str]], Callable[[], Any]]
    ] = None,
    config,
    scheduler_config: Optional[LlmQueueSchedulerConfig] = None,
    logger,
    backend_address: Optional[str] = None,
):
    from marie.engine.llm_queue.adapters.openai_compatible import (
        OpenAICompatibleExecutionAdapter,
    )
    from marie.engine.llm_queue.dispatcher import (
        DrrQueuedBatchDispatcher,
        QueuedBatchDispatcher,
    )

    adapter = OpenAICompatibleExecutionAdapter(
        client=client,
        client_factory=client_factory,
        logger=logger,
        default_timeout=None,
        backend_address=backend_address,
    )
    scheduler_config = ensure_default_pool(
        scheduler_config or LlmQueueSchedulerConfig()
    )
    if scheduler_config.is_drr:
        execution_adapters_by_pool = {}
        for lane in scheduler_config.lanes:
            if not lane.endpoint_url:
                continue
            if client_factory_for_base_url is None:
                continue
            execution_adapters_by_pool[lane.pool_id] = OpenAICompatibleExecutionAdapter(
                client_factory=client_factory_for_base_url(lane.endpoint_url),
                logger=logger,
                default_timeout=None,
                backend_address=lane.endpoint_url,
            )
        return DrrQueuedBatchDispatcher(
            queue_client=queue_client,
            execution_adapter=adapter,
            config=config,
            logger=logger,
            lanes=list(scheduler_config.lanes),
            total_concurrent_dispatch=scheduler_config.total_concurrent_dispatch
            or config.max_batch_items,
            execution_adapters_by_pool=execution_adapters_by_pool,
        )

    return QueuedBatchDispatcher(
        queue_client=queue_client,
        execution_adapter=adapter,
        config=config,
        logger=logger,
    )


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


def _runtime_pool_ids(
    scheduler_config: LlmQueueSchedulerConfig,
    fallback_pool_id: str,
) -> list[str]:
    if not scheduler_config.is_drr:
        return [fallback_pool_id]
    return [lane.pool_id for lane in scheduler_config.lanes if lane.enabled]


def _describe_pool_scope(pool_ids: list[str]) -> str:
    if len(pool_ids) == 1:
        return f"pool '{pool_ids[0]}'"
    return f"{len(pool_ids)} pools: {', '.join(pool_ids)}"


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


def _format_started_runtime_log(
    *,
    scheduler_config: LlmQueueSchedulerConfig,
    fallback_pool_id: str,
) -> str:
    if not scheduler_config.is_drr:
        return "\n".join(
            [
                "Started LLM FIFO dispatch runtime",
                f"  pool: {fallback_pool_id}",
            ]
        )

    lines = [
        "Started LLM DRR dispatch runtime",
        f"  pools: {len(_runtime_pool_ids(scheduler_config, fallback_pool_id))}",
        "  routes:",
    ]
    for lane in scheduler_config.lanes:
        if not lane.enabled:
            continue
        lines.append(f"    - {_format_lane_route(lane)}")
    return "\n".join(lines)


def _format_lane_route(lane: DrrLaneConfig) -> str:
    endpoint_source = "explicit" if lane.endpoint_url else "runtime default"
    max_concurrent = (
        str(lane.max_concurrent) if lane.max_concurrent is not None else "unbounded"
    )
    max_burst = (
        str(lane.max_burst_per_visit)
        if lane.max_burst_per_visit is not None
        else "default"
    )
    return (
        f"{lane.pool_id} "
        f"({endpoint_source}; quantum={lane.quantum}, "
        f"protected={lane.min_concurrent}, max={max_concurrent}, burst={max_burst})"
    )


def _scheduler_fabric_group_id(
    queue_config: LlmQueueConfig, runtime_config: dict[str, Any]
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
