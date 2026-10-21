from __future__ import annotations

import asyncio
import os
from dataclasses import replace
from threading import Event
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
from marie.serve.runtimes.gateway.marie.dispatch_policy import DispatchPolicyError
from marie.serve.runtimes.gateway.marie.llm_scheduler_config import (
    PostgresSchedulerConfigRepository,
)

_STARTUP_FAILURES = {
    'initialization': (
        'dispatch_start_failed',
        'LLM dispatch initialization failed. Verify the registered policy and '
        'runtime dependencies before restarting.',
    ),
    'fabric': (
        'dispatch_fabric_invalid',
        'Fabric identity is missing, invalid, or conflicting. Align '
        'LLM_QUEUE_FABRIC_GROUP_ID with the gateway and scheduler fabric settings.',
    ),
    'policy_source': (
        'dispatch_policy_source_unavailable',
        'The dispatch policy source could not be initialized. Check '
        'llm_queue.llm_dispatch.policy_source and the scheduler PostgreSQL configuration.',
    ),
    'store_configuration': (
        'dispatch_store_not_configured',
        'Queue storage is not configured. Set LLM_QUEUE_URL to the Valkey '
        'connection URL available to the gateway.',
    ),
    'policy': (
        'dispatch_policy_unavailable',
        'A valid dispatch policy could not be loaded. Check PostgreSQL '
        'connectivity, schema, permissions, and the fabric dispatch configuration.',
    ),
    'static_policy': (
        'dispatch_policy_invalid',
        'The static dispatch policy is invalid. Validate llm_queue.llm_dispatch '
        'endpoint and pool registrations, scheduling, capacity, and limits.',
    ),
    'credentials': (
        'dispatch_credentials_missing',
        'A registered credential binding is unavailable. Populate the credential '
        'environment variables referenced by the policy in the gateway process.',
    ),
    'store': (
        'dispatch_store_unavailable',
        'Queue storage could not be opened. Check LLM_QUEUE_URL, Valkey '
        'availability, authentication, TLS, and immutable fabric limits.',
    ),
    'dispatcher': (
        'dispatch_runtime_start_failed',
        'The dispatcher could not be initialized. Check registered endpoint '
        'and pool settings, capacity limits, and queue-store availability.',
    ),
}

_POLICY_FAILURES = {
    'dispatch_startup_policy_invalid': (
        'The saved startup policy is invalid. Cancel the pending change in the '
        'LLM Scheduler, correct the settings and click Save again.'
    ),
    'dispatch_startup_policy_conflict': (
        'The scheduled policy or active revision changed. Review the current '
        'configuration and click Save again.'
    ),
    'dispatch_startup_owner_busy': (
        'Another dispatcher owns this fabric. Stop that dispatcher before '
        'starting the scheduled policy.'
    ),
    'dispatch_startup_owner_lost': (
        'Startup lost fenced ownership or queue-store connectivity. Restore '
        'storage availability and restart the gateway with the saved settings.'
    ),
    'dispatch_startup_work_pending': (
        'Existing work must drain before startup activation. Cancel the pending '
        'change, restart the active configuration to drain work, then click Save '
        'again and restart the gateway.'
    ),
    'dispatch_policy_not_configured': (
        'No registered dispatch policy is configured. The default fabric seed '
        'does not provide endpoints or pools. Provision the selected policy source with '
        'registered endpoints, pools, and positive dispatch capacity. For bootstrap '
        'only, set LLM_QUEUE_ENABLED=false and keep queued producers stopped.'
    ),
    'dispatch_policy_invalid': (
        'The registered policy is invalid. Validate metadata.llm_dispatch, '
        'endpoint registrations, pool bindings, and policy schema versions.'
    ),
    'dispatch_capacity_invalid': (
        'Dispatch scheduling or capacity is invalid. Set total_concurrent_dispatch '
        'to a positive value within the fabric limits and select fifo or drr.'
    ),
    'dispatch_pools_missing': (
        'No dispatch pools are configured. Provision llm_queue_pool rows with '
        'registered endpoint bindings and capacity before enabling queued dispatch.'
    ),
    'dispatch_fabric_not_configured': (
        'No enabled fabric configuration exists for this gateway. Provision or '
        'enable the matching llm_queue_fabric_config row and its dispatch policy.'
    ),
    'dispatch_fabric_invalid': _STARTUP_FAILURES['fabric'][1],
    'dispatch_store_client_missing': (
        'The Valkey Python client is not installed in the gateway environment. '
        'Synchronize the application dependencies, including valkey, in the '
        'Python environment used to launch the gateway, then restart.'
    ),
    'dispatch_policy_generation_invalid': (
        'The active policy generation is missing or invalid. Inspect the stored '
        'generation and digest, then restore or activate a validated policy revision.'
    ),
}

_STARTUP_GUIDE = (
    'docs/docs/getting-started/deployment/scheduler-database-upgrades.md'
    '#provision-llm-dispatch-on-a-fresh-database'
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
        self._startup_phase = 'initialization'
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

        self._startup_phase = 'fabric'
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
            self._startup_phase = 'policy_source'
            if self._scheduler_config_source is None:
                self._scheduler_config_source = _build_scheduler_config_source(
                    config=self.runtime_config,
                    fabric_group_id=effective_fabric,
                    default_total_concurrent_dispatch=self.config.max_batch_items,
                    logger=self.logger,
                )
            self._startup_phase = 'initialization'
            await self._start_queued_dispatch()
        except Exception as error:
            self._last_error = 'dispatch_start_failed'
            code, action = _STARTUP_FAILURES[self._startup_phase]
            if (
                isinstance(error, DispatchPolicyError)
                and error.code in _POLICY_FAILURES
            ):
                code, action = error.code, _POLICY_FAILURES[error.code]
            elif (
                self._startup_phase == 'store'
                and isinstance(error, ModuleNotFoundError)
                and error.name == 'valkey'
            ):
                code = 'dispatch_store_client_missing'
                action = _POLICY_FAILURES[code]
            message = (
                f'LLM dispatch startup failed [{code}]: {action} See {_STARTUP_GUIDE}.'
            )
            self.logger.error(message)
            raise RuntimeFailToStart(message) from None

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

        self._startup_phase = 'store_configuration'
        if not self.config.queue_url:
            raise ValueError('Dispatch requires queue storage')
        from marie.serve.runtimes.gateway.marie.dispatch_policy import (
            persisted_dispatch_policy,
            validate_dispatch_policy,
        )

        self._startup_phase = 'policy_source'
        configured = dict(self.runtime_config.get('llm_dispatch') or {})
        source = configured.pop('policy_source', 'static')
        startup_request = None
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
                raise DispatchPolicyError(
                    'dispatch_fabric_invalid',
                    'Database policy fabric identity mismatch',
                )
            repository = self._scheduler_config_source.repository
            self._startup_phase = 'policy'
            if hasattr(repository, 'load_startup_dispatch_policy'):
                startup_request = await asyncio.to_thread(
                    repository.load_startup_dispatch_policy,
                    self.config.fabric_group_id,
                    initialize=True,
                )
            if startup_request is not None:
                policy = startup_request.dispatch
            elif hasattr(repository, 'load_runtime_dispatch_policy'):
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
            self._startup_phase = 'static_policy'
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
                loaded = repository.load_runtime_dispatch_policy(
                    self.config.fabric_group_id
                )
                # Unactivated rows are drafts; retain the bootstrap policy until activation.
                return (
                    resolved
                    if loaded.get('policy_generation') is None
                    else resolve_policy(loaded)
                )
            data = repository.load_scheduler_config(self.config.fabric_group_id)
            persisted_dispatch_policy(data)
            return resolved

        self._startup_phase = 'credentials'
        resolved = resolve_policy(policy)
        endpoints, lanes = resolved['endpoints'], resolved['lanes']
        store = None
        dispatcher = None
        startup_owner = None
        startup_heartbeat = None
        try:
            self._startup_phase = 'store'
            store = await asyncio.to_thread(
                RequestStore,
                self.config.queue_url,
                fabric_id=self.config.fabric_group_id,
                limits=StoreLimits(**policy['limits']),
            )
            self._startup_phase = 'dispatcher'
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
            if startup_request is not None:
                startup_owner = await asyncio.to_thread(
                    store.acquire_owner,
                    dispatcher.dispatcher_id,
                    lease_ms=dispatcher.owner_lease_ms,
                )
                if startup_owner is None:
                    raise DispatchPolicyError(
                        'dispatch_startup_owner_busy', 'Fabric owner is unavailable'
                    )

                async def renew_startup_owner() -> None:
                    while True:
                        await asyncio.sleep(dispatcher.owner_lease_ms / 3000)
                        await asyncio.to_thread(
                            store.renew_owner,
                            startup_owner,
                            lease_ms=dispatcher.owner_lease_ms,
                        )

                startup_heartbeat = asyncio.create_task(renew_startup_owner())
                startup_cancelled = Event()
                cursor = 0
                while True:
                    cursor, pools = await asyncio.to_thread(
                        store.scan_routes, cursor=cursor, limit=100
                    )
                    for pool in pools:
                        await asyncio.to_thread(
                            store.disable_route, startup_owner, pool
                        )
                    if cursor == 0:
                        break

                def check_store() -> None:
                    if startup_cancelled.is_set():
                        raise DispatchPolicyError(
                            'dispatch_startup_policy_conflict', 'Startup cancelled'
                        )
                    if startup_heartbeat.done():
                        raise DispatchPolicyError(
                            'dispatch_startup_owner_lost', 'Startup heartbeat stopped'
                        )
                    store.renew_owner(startup_owner, lease_ms=dispatcher.owner_lease_ms)
                    reply = store.reset_idle_bindings(startup_owner)
                    if reply.disposition != 'reset':
                        raise DispatchPolicyError(
                            'dispatch_startup_work_pending', 'Fabric is not drained'
                        )

                activation_task = asyncio.create_task(
                    asyncio.to_thread(
                        repository.activate_startup_policy,
                        self.config.fabric_group_id,
                        startup_request,
                        check_store,
                    )
                )
                try:
                    activation = await asyncio.shield(activation_task)
                except asyncio.CancelledError:
                    startup_cancelled.set()
                    # A cancelled await cannot stop the database thread. Keep its
                    # ownership alive until it has committed or rolled back.
                    while not activation_task.done():
                        try:
                            await asyncio.shield(activation_task)
                        except asyncio.CancelledError:
                            continue
                        except Exception:
                            break
                    if not activation_task.cancelled():
                        activation_task.exception()
                    raise
                dispatcher.policy_generation = activation.generation
                dispatcher.policy_digest = activation.policy_digest
                self.logger.info(
                    f'LLM routing policy activated: fabric={self.config.fabric_group_id} '
                    f'generation={activation.generation}'
                )
                await dispatcher.start(owner=startup_owner)
                startup_owner = None
            else:
                await dispatcher.start()
        except (Exception, asyncio.CancelledError):
            if startup_heartbeat is not None:
                startup_heartbeat.cancel()
                await asyncio.gather(startup_heartbeat, return_exceptions=True)
            if dispatcher is not None:
                try:
                    await dispatcher.stop()
                except Exception:
                    self.logger.error(
                        'Failed stopping partially started LLM dispatch runtime'
                    )
            if startup_owner is not None:
                try:
                    await asyncio.to_thread(store.release_owner, startup_owner)
                except Exception:
                    self.logger.error('Failed releasing startup dispatch ownership')
            if store is not None:
                try:
                    await asyncio.to_thread(store.close)
                except Exception:
                    self.logger.error(
                        'Failed closing store after LLM dispatch startup error'
                    )
            raise
        finally:
            if startup_heartbeat is not None:
                startup_heartbeat.cancel()
                await asyncio.gather(startup_heartbeat, return_exceptions=True)
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
