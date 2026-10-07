"""Fabric-scoped LLM dispatch operator routes."""

import asyncio
from collections.abc import Awaitable
from typing import Any, Callable, Literal
from uuid import UUID, uuid4

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from marie.engine.llm_queue import registry
from marie.engine.llm_queue.admission_policy import (
    AdmissionMatchError,
    AdmissionPolicyError,
)
from pydantic import BaseModel, ConfigDict, Field, field_validator

from marie.auth.api_key_manager import APIKeyManager
from marie.auth.auth_bearer import TokenBearer
from marie.scheduler.llm_routing import admission_routing_metrics


class RoutingPreviewRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    facts: dict[str, int | str]


class PolicyActivationRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    generation: int | None = Field(default=None, ge=1, le=2**53 - 1, strict=True)
    expected_generation: int | None = Field(
        default=None, ge=1, le=2**53 - 1, strict=True
    )


class RoutingResourceRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    resource_type: Literal['pool', 'policy_generation', 'endpoint_group', 'replica']
    resource_id: str = Field(min_length=1, max_length=128, pattern=r'^[A-Za-z0-9_-]+$')
    revision: str | None = Field(
        default=None, min_length=1, max_length=128, pattern=r'^[A-Za-z0-9_-]+$'
    )


class RoutingOverrideRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    pool_id: str = Field(min_length=1, max_length=128, pattern=r'^[A-Za-z0-9_.-]+$')
    reason: str = Field(min_length=1, max_length=512)
    submission: dict[str, Any]

    @field_validator('reason')
    @classmethod
    def validate_reason(cls, value: str) -> str:
        reason = value.strip()
        if not reason:
            raise ValueError('routing override reason cannot be blank')
        return reason


class RoutingDiagnosticsUnavailable(RuntimeError):
    pass


def _merge_runtime_diagnostics(
    snapshot: dict[str, Any],
    diagnostics: dict[str, Any],
    fabric_group_id: str,
    limit: int,
) -> dict[str, Any]:
    policy = {**snapshot.get('policy', {}), **diagnostics.get('policy', {})}
    desired = policy.get('desired_digest')
    observed = policy.get('observed_digest')
    policy['synchronized'] = bool(
        desired
        and observed
        and desired == observed
        and policy.get('desired_generation') == policy.get('observed_generation')
    )
    snapshot['policy'] = policy
    if diagnostics.get('scheduler_config') is not None:
        snapshot['scheduler_config'] = diagnostics['scheduler_config']
    snapshot['routing'] = diagnostics.get('routing', {})
    snapshot['recent_routes'] = diagnostics.get('recent_routes', [])
    snapshot['recent_routes_truncated'] = bool(
        diagnostics.get('recent_routes_truncated')
    )
    local_routing = admission_routing_metrics.snapshot(fabric_group_id)
    if local_routing['available']:
        snapshot['routing']['rejected'] = local_routing['rejected']
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
    snapshot['pool_count'] = diagnostics.get('pool_count', len(snapshot['pools']))
    snapshot['pools_truncated'] = bool(
        diagnostics.get('pools_truncated')
        or len(runtime_pools.keys() | database_pools.keys()) > limit
    )
    return snapshot


async def read_operator_runtime_snapshot(
    *,
    fabric_group_id: str,
    limit: int,
    policy_repository: Any | None,
) -> dict[str, Any]:
    snapshot = await registry.read_runtime_snapshot(
        fabric_group_id=fabric_group_id, limit=limit
    )
    if policy_repository is None or not hasattr(
        policy_repository, 'load_routing_diagnostics'
    ):
        return snapshot
    try:
        diagnostics = await asyncio.to_thread(
            policy_repository.load_routing_diagnostics,
            fabric_group_id,
            limit,
        )
    except Exception as exc:
        raise RoutingDiagnosticsUnavailable from exc
    return _merge_runtime_diagnostics(snapshot, diagnostics, fabric_group_id, limit)


def add_runtime_routes(
    app: FastAPI,
    effective_fabric: Callable[[], str],
    policy_repository: Callable[[], Any | None] | None = None,
    routing_override_submitter: (
        Callable[[dict[str, Any], str, str, str, str], Awaitable[dict[str, Any]]] | None
    ) = None,
    gateway_debug: Callable[[], dict[str, object]] | None = None,
    failure_report_reader: (
        Callable[[str, int], Awaitable[dict[str, Any] | None]] | None
    ) = None,
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

    @app.get('/api/operations/jobs/{job_id}/failure-report')
    async def failure_report(
        job_id: UUID,
        history_id: int = Query(ge=1, le=2**63 - 1),
        fabric_group_id: str = Depends(authorize),
    ) -> dict[str, Any]:
        if failure_report_reader is None:
            raise HTTPException(status_code=503, detail='failure_report_unavailable')
        try:
            report = await failure_report_reader(str(job_id), history_id)
        except LookupError:
            raise HTTPException(
                status_code=404, detail='execution_event_not_found'
            ) from None
        except Exception:
            raise HTTPException(
                status_code=503, detail='failure_report_unavailable'
            ) from None
        if report is None:
            raise HTTPException(status_code=404, detail='job_not_found')
        return {
            'status': 'OK',
            'result': {**report, 'fabric_group_id': fabric_group_id},
        }

    def repository() -> Any:
        selected = policy_repository() if policy_repository is not None else None
        if selected is None:
            raise HTTPException(
                status_code=503, detail='routing_policy_repository_unavailable'
            )
        return selected

    async def read(fabric_group_id: str, limit: int, *, debug: bool = False):
        gateway = gateway_debug() if debug and gateway_debug is not None else None
        selected_repository = (
            policy_repository() if policy_repository is not None else None
        )
        try:
            snapshot = await read_operator_runtime_snapshot(
                fabric_group_id=fabric_group_id,
                limit=limit,
                policy_repository=selected_repository,
            )
        except registry.SnapshotUnavailable:
            content: dict[str, Any] = {
                'status': 'error',
                'category': 'runtime_unavailable',
                'correlation_id': uuid4().hex,
            }
            if gateway is not None:
                content['gateway'] = gateway
            return JSONResponse(
                status_code=503,
                content=content,
            )
        except RoutingDiagnosticsUnavailable:
            content = {
                'status': 'error',
                'category': 'routing_diagnostics_unavailable',
                'correlation_id': uuid4().hex,
            }
            if gateway is not None:
                content['gateway'] = gateway
            return JSONResponse(
                status_code=503,
                content=content,
            )
        result = (
            {
                'gateway': gateway,
                'llm_dispatch': snapshot,
                'scheduler_details_available': False,
            }
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
        request: PolicyActivationRequest | None = None,
        authorization: tuple[str, str] = Depends(authorize_routing_admin),
    ) -> dict[str, object]:
        fabric_group_id, actor = authorization
        try:
            if request is None or request.generation is None:
                activated = await asyncio.to_thread(
                    repository().activate_admission_policy,
                    fabric_group_id,
                    actor,
                )
            else:
                activated = await asyncio.to_thread(
                    repository().activate_policy_revision,
                    fabric_group_id,
                    request.generation,
                    actor,
                    expected_generation=request.expected_generation,
                )
        except AdmissionPolicyError as exc:
            raise HTTPException(status_code=400, detail=exc.category) from None
        except HTTPException:
            raise
        except ValueError as exc:
            if str(exc) == 'routing_policy_activation_conflict':
                raise HTTPException(status_code=409, detail=str(exc)) from None
            if str(exc) == 'routing_policy_revision_unavailable':
                raise HTTPException(status_code=400, detail=str(exc)) from None
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

    @app.post('/api/llm-dispatch/routing/override')
    async def submit_routing_override(
        request: RoutingOverrideRequest,
        authorization: tuple[str, str] = Depends(authorize_routing_admin),
    ) -> dict[str, object]:
        fabric_group_id, actor = authorization
        if routing_override_submitter is None:
            raise HTTPException(status_code=503, detail='routing_override_unavailable')
        try:
            policy = await asyncio.to_thread(
                repository().load_active_admission_policy, fabric_group_id
            )
            policy.endpoint_binding(request.pool_id)
            if not any(rule.pool_id == request.pool_id for rule in policy.rules):
                raise AdmissionMatchError('routing_override_pool_invalid')
            result = await routing_override_submitter(
                request.submission,
                fabric_group_id,
                request.pool_id,
                actor,
                request.reason,
            )
        except AdmissionMatchError as exc:
            raise HTTPException(status_code=400, detail=exc.category) from None
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(
                status_code=503, detail='routing_override_submission_failed'
            ) from None
        return {'status': 'OK', 'result': result}

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
