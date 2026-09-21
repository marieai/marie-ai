"""Fabric-scoped LLM dispatch operator routes."""

import asyncio
from typing import Any, Callable, Literal
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from marie.engine.llm_queue import registry
from marie.engine.llm_queue.admission_policy import (
    AdmissionMatchError,
    AdmissionPolicyError,
)
from pydantic import BaseModel, ConfigDict, Field

from marie.auth.api_key_manager import APIKeyManager
from marie.auth.auth_bearer import TokenBearer
from marie.scheduler.llm_routing import admission_routing_metrics


class RoutingPreviewRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    facts: dict[str, int | str]


class RoutingResourceRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    resource_type: Literal['pool', 'policy_generation', 'endpoint_group', 'replica']
    resource_id: str = Field(min_length=1, max_length=128, pattern=r'^[A-Za-z0-9_-]+$')
    revision: str | None = Field(
        default=None, min_length=1, max_length=128, pattern=r'^[A-Za-z0-9_-]+$'
    )


def add_runtime_routes(
    app: FastAPI,
    effective_fabric: Callable[[], str],
    policy_repository: Callable[[], Any | None] | None = None,
) -> None:
    async def authorize(
        fabric_group_id: str = Query(min_length=1, max_length=128),
        tenant_id: str | None = None,
        token: str = Depends(TokenBearer()),
    ) -> str:
        if (
            fabric_group_id != effective_fabric()
            or not APIKeyManager.can_observe_runtime(token, fabric_group_id, tenant_id)
        ):
            raise HTTPException(
                status_code=403, detail='runtime_observability_forbidden'
            )
        return fabric_group_id

    async def authorize_routing_admin(
        fabric_group_id: str = Query(min_length=1, max_length=128),
        tenant_id: str | None = None,
        token: str = Depends(TokenBearer()),
    ) -> tuple[str, str]:
        actor = APIKeyManager.key_name(token)
        if (
            fabric_group_id != effective_fabric()
            or not APIKeyManager.can_admin_runtime_routing(
                token, fabric_group_id, tenant_id
            )
            or actor is None
        ):
            raise HTTPException(
                status_code=403, detail='runtime_routing_admin_forbidden'
            )
        return fabric_group_id, actor

    def repository() -> Any:
        selected = policy_repository() if policy_repository is not None else None
        if selected is None:
            raise HTTPException(
                status_code=503, detail='routing_policy_repository_unavailable'
            )
        return selected

    async def read(fabric_group_id: str, limit: int, *, debug: bool = False):
        try:
            snapshot = await registry.read_runtime_snapshot(
                fabric_group_id=fabric_group_id, limit=limit
            )
        except registry.SnapshotUnavailable:
            return JSONResponse(
                status_code=503,
                content={
                    'status': 'error',
                    'category': 'runtime_unavailable',
                    'correlation_id': uuid4().hex,
                },
            )
        selected_repository = (
            policy_repository() if policy_repository is not None else None
        )
        if selected_repository is not None and hasattr(
            selected_repository, 'load_routing_diagnostics'
        ):
            try:
                diagnostics = await asyncio.to_thread(
                    selected_repository.load_routing_diagnostics,
                    fabric_group_id,
                    limit,
                )
            except Exception:
                return JSONResponse(
                    status_code=503,
                    content={
                        'status': 'error',
                        'category': 'routing_diagnostics_unavailable',
                        'correlation_id': uuid4().hex,
                    },
                )
            policy = {**snapshot.get('policy', {}), **diagnostics.get('policy', {})}
            desired = policy.get('desired_digest')
            observed = policy.get('observed_digest')
            policy['synchronized'] = bool(
                desired
                and observed
                and desired == observed
                and policy.get('desired_generation')
                == policy.get('observed_generation')
            )
            snapshot['policy'] = policy
            snapshot['routing'] = diagnostics.get('routing', {})
            local_routing = admission_routing_metrics.snapshot(fabric_group_id)
            if local_routing['available']:
                snapshot['routing']['rejected'] = local_routing['rejected']
                snapshot['routing']['shadow'] = {
                    'available': True,
                    **local_routing['shadow'],
                }
            runtime_pools = {
                row.get('pool_id'): row
                for row in snapshot.get('pools', [])
                if isinstance(row, dict) and isinstance(row.get('pool_id'), str)
            }
            database_pools = {
                row.get('pool_id'): row
                for row in diagnostics.get('pools', [])
                if isinstance(row, dict) and isinstance(row.get('pool_id'), str)
            }
            merged_pools = []
            for pool_id in sorted(runtime_pools.keys() | database_pools.keys()):
                runtime_pool = runtime_pools.get(pool_id, {})
                database_pool = database_pools.get(pool_id, {})
                merged = {**runtime_pool, **database_pool}
                postgres_refs = int(database_pool.get('drain_references') or 0)
                valkey_refs = int(runtime_pool.get('drain_references') or 0)
                merged['drain_references'] = {
                    'postgres': postgres_refs,
                    'valkey': valkey_refs,
                    'total': postgres_refs + valkey_refs,
                }
                merged_pools.append(merged)
            snapshot['pools'] = merged_pools[:limit]
            snapshot['pool_count'] = diagnostics.get(
                'pool_count', len(snapshot['pools'])
            )
            snapshot['pools_truncated'] = bool(
                diagnostics.get('pools_truncated')
                or len(runtime_pools.keys() | database_pools.keys()) > limit
            )
        result = (
            {'llm_dispatch': snapshot, 'scheduler_details_available': False}
            if debug
            else snapshot
        )
        return {'status': 'OK', 'result': result}

    @app.get('/api/llm-dispatch/runtime')
    async def runtime(
        fabric_group_id: str = Depends(authorize),
        limit: int = Query(default=50, ge=1, le=250),
    ):
        return await read(fabric_group_id, limit)

    @app.get('/api/debug')
    async def debug(fabric_group_id: str = Depends(authorize)):
        return await read(fabric_group_id, 50, debug=True)

    @app.post('/api/llm-dispatch/policy/preview')
    async def preview_policy(
        request: RoutingPreviewRequest,
        authorization: tuple[str, str] = Depends(authorize_routing_admin),
    ) -> dict[str, object]:
        fabric_group_id, _ = authorization
        try:
            policy = await asyncio.to_thread(
                repository().load_active_admission_policy, fabric_group_id
            )
            decision = policy.match(request.facts)
        except (AdmissionMatchError, AdmissionPolicyError) as exc:
            raise HTTPException(status_code=400, detail=exc.category) from None
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(
                status_code=503, detail='routing_policy_unavailable'
            ) from None
        return {
            'status': 'OK',
            'result': {
                'pool_id': decision.pool_id,
                'priority': decision.priority,
                'policy_generation': decision.policy_generation,
                'policy_digest': decision.policy_digest,
                'rule_digest': decision.rule_digest,
                'matched_facts': list(decision.matched_facts),
            },
        }

    @app.post('/api/llm-dispatch/policy/activate')
    async def activate_policy(
        authorization: tuple[str, str] = Depends(authorize_routing_admin),
    ) -> dict[str, object]:
        fabric_group_id, actor = authorization
        try:
            activated = await asyncio.to_thread(
                repository().activate_admission_policy,
                fabric_group_id,
                actor,
            )
        except AdmissionPolicyError as exc:
            raise HTTPException(status_code=400, detail=exc.category) from None
        except HTTPException:
            raise
        except ValueError:
            raise HTTPException(
                status_code=400, detail='routing_policy_invalid'
            ) from None
        except Exception:
            raise HTTPException(
                status_code=503, detail='routing_policy_unavailable'
            ) from None
        return {
            'status': 'OK',
            'result': {
                'fabric_group_id': activated.fabric_group_id,
                'generation': activated.generation,
                'policy_digest': activated.policy_digest,
                'activated_by': activated.activated_by,
            },
        }

    @app.post('/api/llm-dispatch/resources/check')
    async def check_routing_resource(
        request: RoutingResourceRequest,
        authorization: tuple[str, str] = Depends(authorize_routing_admin),
        limit: int = Query(default=25, ge=1, le=250),
    ) -> dict[str, object]:
        fabric_group_id, _ = authorization
        try:
            postgres, valkey = await asyncio.gather(
                asyncio.to_thread(
                    repository().routing_resource_references,
                    fabric_group_id=fabric_group_id,
                    resource_type=request.resource_type,
                    resource_id=request.resource_id,
                    revision=request.revision,
                    limit=limit,
                ),
                asyncio.to_thread(
                    registry.routing_resource_references,
                    fabric_group_id=fabric_group_id,
                    resource_type=request.resource_type,
                    resource_id=request.resource_id,
                    revision=request.revision,
                    limit=limit,
                ),
            )
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(
                status_code=503, detail='routing_reference_state_unavailable'
            ) from None
        references = {
            **postgres,
            **valkey,
            'truncated': bool(postgres.get('truncated') or valkey.get('truncated')),
        }
        references['total'] = int(references.get('postgres', 0)) + int(
            references.get('valkey', 0)
        )
        if references['total'] == 0 and references['truncated']:
            raise HTTPException(
                status_code=503, detail='routing_reference_state_incomplete'
            )
        if references['total']:
            raise HTTPException(
                status_code=409,
                detail={
                    'category': 'routing_resource_in_use',
                    'references': references,
                },
            )
        return {'status': 'OK', 'result': {'references': references}}
