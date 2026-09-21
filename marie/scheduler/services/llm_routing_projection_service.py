from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from marie.engine.llm_queue.store import (
    AdmissionConflict,
    RequestStore,
    RoutingManifestProjection,
    StoreUnavailable,
)

from marie.logging_core.logger import MarieLogger


@dataclass(frozen=True, slots=True)
class ProjectionBatchResult:
    attempted: int
    projected: int
    existing: int
    acknowledged: int
    failed: int


class LlmRoutingProjectionService:
    def __init__(
        self,
        *,
        repository: Any,
        store: RequestStore,
        logger: MarieLogger,
        retry_seconds: int = 5,
    ) -> None:
        self.repository = repository
        self.store = store
        self.logger = logger
        self.retry_seconds = retry_seconds

    async def run_once(self, limit: int = 100) -> ProjectionBatchResult:
        events = await self.repository.claim_pending_routing_outbox(limit=limit)
        projected = 0
        existing = 0
        acknowledged = 0
        failed = 0
        for event in events:
            event_id = str(event['event_id'])
            route_digest = str(event['route_digest'])
            try:
                projection = self._projection(event, route_digest)
                reply = await asyncio.to_thread(
                    self.store.project_routing_manifest, projection
                )
                if reply.disposition == 'projected':
                    projected += 1
                elif reply.disposition == 'existing':
                    existing += 1
                else:
                    raise RuntimeError('routing_projection_rejected')
                if await self.repository.acknowledge_routing_projection(
                    event_id=event_id,
                    route_digest=route_digest,
                ):
                    acknowledged += 1
            except AdmissionConflict:
                failed += 1
                await self._record_failure(
                    event_id, route_digest, 'routing_binding_conflict'
                )
            except StoreUnavailable:
                failed += 1
                await self._record_failure(
                    event_id, route_digest, 'routing_store_unavailable'
                )
            except (KeyError, TypeError, ValueError):
                failed += 1
                await self._record_failure(
                    event_id, route_digest, 'routing_projection_invalid'
                )
            except Exception:
                failed += 1
                await self._record_failure(
                    event_id, route_digest, 'routing_projection_failed'
                )
        return ProjectionBatchResult(
            attempted=len(events),
            projected=projected,
            existing=existing,
            acknowledged=acknowledged,
            failed=failed,
        )

    async def run_forever(self, idle_seconds: float = 0.25) -> None:
        while True:
            try:
                result = await self.run_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                self.logger.exception('LLM routing projection batch unavailable')
                await asyncio.sleep(idle_seconds)
                continue
            if result.attempted == 0:
                await asyncio.sleep(idle_seconds)

    async def _record_failure(
        self, event_id: str, route_digest: str, category: str
    ) -> None:
        await self.repository.record_routing_projection_failure(
            event_id=event_id,
            route_digest=route_digest,
            category=category,
            retry_seconds=self.retry_seconds,
        )
        self.logger.warning(
            'LLM routing projection failed event_id=%s category=%s',
            event_id,
            category,
        )

    @staticmethod
    def _projection(
        event: dict[str, Any], route_digest: str
    ) -> RoutingManifestProjection:
        payload = event['payload']
        if not isinstance(payload, dict) or set(payload) != {
            'job_id',
            'work_unit_id',
            'fabric_group_id',
            'policy_generation',
            'policy_digest',
            'rule_digest',
            'normalized_fact_digest',
            'effective_page_count',
            'pool_id',
            'logical_endpoint_group_id',
            'endpoint_revision',
            'estimator_version',
            'routing_source',
        }:
            raise ValueError('routing_projection_invalid')
        return RoutingManifestProjection(
            job_id=payload['job_id'],
            work_unit_id=payload['work_unit_id'],
            fabric_group_id=payload['fabric_group_id'],
            policy_generation=payload['policy_generation'],
            route_digest=route_digest,
            pool_id=payload['pool_id'],
            endpoint_group_id=payload['logical_endpoint_group_id'],
            endpoint_revision=payload['endpoint_revision'],
        )
