from __future__ import annotations

import hashlib
import json
import re
import threading
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

from marie.engine.llm_queue.admission_policy import (
    AdmissionMatchError,
    AdmissionPolicy,
    AdmissionPolicyError,
    FactValue,
    PoolEndpointBinding,
)
from pydantic import BaseModel, ConfigDict, Field, field_validator

from marie.query_planner.base import Query, QueryPlan
from marie.storage.submission.types import SubmissionDocument

if TYPE_CHECKING:
    from marie.scheduler.models import WorkInfo

EndpointBinding = PoolEndpointBinding

_FORBIDDEN_SELECTORS = frozenset(
    {
        'pool_id',
        'endpoint_id',
        'endpoint_group_id',
        'endpoint_revision',
        'replica_id',
        'admission_priority',
        'policy_generation',
        'llm_queue_pool_id',
        'llm_queue_contract_version',
    }
)
_WORKLOAD_KINDS = frozenset({'document', 'text', 'image', 'agent'})
_WORKLOAD_MODES = frozenset({'interactive', 'batch', 'backfill', 'background'})
_PIPELINE_STAGES = frozenset({'extract', 'validate', 'enrich'})
_REQUEST_SOURCES = frozenset(
    {'gateway-job-api', 'studio-submission', 'workflow', 'operator'}
)
_METRIC_CATEGORY_RE = re.compile(r'^[a-z][a-z0-9_]{0,63}$')


class AdmissionRoutingMetrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._matched: Counter[tuple[str, str]] = Counter()
        self._rejected: Counter[tuple[str, str]] = Counter()
        self._seen_fabrics: set[str] = set()

    def record_match(self, *, fabric_group_id: str, category: str) -> None:
        category = (
            category if _METRIC_CATEGORY_RE.fullmatch(category) else 'routing_error'
        )
        with self._lock:
            self._seen_fabrics.add(fabric_group_id)
            self._matched[(fabric_group_id, category)] += 1

    def record_rejection(self, *, fabric_group_id: str, category: str) -> None:
        category = (
            category if _METRIC_CATEGORY_RE.fullmatch(category) else 'routing_error'
        )
        with self._lock:
            self._seen_fabrics.add(fabric_group_id)
            self._rejected[(fabric_group_id, category)] += 1

    def snapshot(self, fabric_group_id: str) -> dict[str, Any]:
        with self._lock:
            return {
                'available': fabric_group_id in self._seen_fabrics,
                'matched': {
                    category: count
                    for (fabric, category), count in self._matched.items()
                    if fabric == fabric_group_id
                },
                'rejected': {
                    category: count
                    for (fabric, category), count in self._rejected.items()
                    if fabric == fabric_group_id
                },
            }


admission_routing_metrics = AdmissionRoutingMetrics()


class RoutingSubmissionError(ValueError):
    def __init__(self, category: str) -> None:
        self.category = category
        super().__init__(category)


class TrustedRoutingFacts(BaseModel):
    model_config = ConfigDict(frozen=True)

    values: dict[str, int | str]
    digest: str

    @classmethod
    def from_values(cls, values: Mapping[str, FactValue]) -> TrustedRoutingFacts:
        normalized = dict(sorted(values.items()))
        return cls(values=normalized, digest=_digest(normalized))

    def with_stage(self, stage: str) -> TrustedRoutingFacts:
        if stage not in _PIPELINE_STAGES:
            raise RoutingSubmissionError('routing_facts_invalid')
        return self.from_values({**self.values, 'pipeline.stage': stage})


class TrustedRoutingContext(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    base_facts: TrustedRoutingFacts
    policy: AdmissionPolicy = Field(exclude=True)


class TrustedRoutingOverride(BaseModel):
    model_config = ConfigDict(frozen=True)

    pool_id: str = Field(min_length=1, max_length=128, pattern=r'^[A-Za-z0-9_.-]+$')
    actor: str = Field(min_length=1, max_length=128)
    reason: str = Field(min_length=1, max_length=512)

    @field_validator('reason')
    @classmethod
    def validate_reason(cls, value: str) -> str:
        reason = value.strip()
        if not reason:
            raise ValueError('routing override reason cannot be blank')
        return reason


class PlannedLlmRoute(BaseModel):
    model_config = ConfigDict(frozen=True)

    job_id: str
    work_unit_id: str
    fabric_group_id: str
    policy_generation: int
    policy_digest: str
    rule_digest: str
    normalized_fact_digest: str
    effective_page_count: int | None
    pool_id: str
    logical_endpoint_group_id: str
    endpoint_revision: str
    estimator_version: Literal['page-count-v1'] = 'page-count-v1'
    routing_source: Literal['automatic', 'operator-override'] = 'automatic'
    routing_actor: str | None = None
    routing_reason: str | None = None


def reject_external_routing_selectors(value: object) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if isinstance(key, str) and key.casefold() in _FORBIDDEN_SELECTORS:
                raise RoutingSubmissionError('caller_pool_forbidden')
            reject_external_routing_selectors(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            reject_external_routing_selectors(nested)


def normalize_routing_facts(
    *,
    document: SubmissionDocument | None,
    requested_pages: Sequence[int] | None,
    workload_kind: str,
    workload_mode: str,
    pipeline_stage: str | None,
    request_source: str,
) -> TrustedRoutingFacts:
    if (
        workload_kind not in _WORKLOAD_KINDS
        or workload_mode not in _WORKLOAD_MODES
        or request_source not in _REQUEST_SOURCES
        or (pipeline_stage is not None and pipeline_stage not in _PIPELINE_STAGES)
    ):
        raise RoutingSubmissionError('routing_facts_invalid')

    values: dict[str, FactValue] = {
        'workload.kind': workload_kind,
        'workload.mode': workload_mode,
        'request.source': request_source,
    }
    if pipeline_stage is not None:
        values['pipeline.stage'] = pipeline_stage

    if workload_kind == 'document':
        if document is None or document.page_count is None:
            raise RoutingSubmissionError('routing_facts_missing')
        total = document.page_count
        if type(total) is not int or total < 1:
            raise RoutingSubmissionError('routing_facts_invalid')
        values['document.total_page_count'] = total
        effective = total
        if requested_pages is not None:
            if not requested_pages:
                raise RoutingSubmissionError('routing_facts_invalid')
            pages: set[int] = set()
            for page in requested_pages:
                if type(page) is not int or page < 0 or page >= total:
                    raise RoutingSubmissionError('routing_facts_invalid')
                pages.add(page)
            effective = len(pages)
            if effective < 1:
                raise RoutingSubmissionError('routing_facts_invalid')
            values['document.requested_page_count'] = effective
        values['document.effective_page_count'] = effective
    elif requested_pages is not None:
        raise RoutingSubmissionError('routing_facts_invalid')

    return TrustedRoutingFacts.from_values(values)


def normalize_pipeline_stage(node: Query) -> str:
    params = node.definition.params if isinstance(node.definition.params, dict) else {}
    explicit = params.get('pipeline_stage')
    if explicit is not None:
        if explicit not in _PIPELINE_STAGES:
            raise RoutingSubmissionError('routing_facts_invalid')
        return explicit
    source = f'{node.query_str} {node.definition.endpoint}'.casefold()
    if 'validat' in source or 'review' in source:
        return 'validate'
    if 'enrich' in source or 'embed' in source or 'index' in source:
        return 'enrich'
    return 'extract'


def plan_llm_routes(
    *,
    root: WorkInfo,
    plan: QueryPlan,
    nodes: Sequence[WorkInfo],
    policy: AdmissionPolicy,
    base_facts: TrustedRoutingFacts,
    override: TrustedRoutingOverride | None = None,
) -> tuple[PlannedLlmRoute, ...]:
    work_by_id = {work.id: work for work in nodes}
    if len(work_by_id) != len(nodes):
        raise RoutingSubmissionError('routing_manifest_missing')
    routes: list[PlannedLlmRoute] = []
    routed_ids: set[str] = set()
    for node in plan.nodes:
        if str(node.definition.method).upper() != 'LLM':
            continue
        work_info = work_by_id.get(node.task_id)
        if work_info is None or work_info.id in routed_ids:
            raise RoutingSubmissionError('routing_manifest_missing')
        facts = base_facts.with_stage(normalize_pipeline_stage(node))
        try:
            decision = policy.match(facts.values) if override is None else None
            pool_id = decision.pool_id if decision is not None else override.pool_id
            binding = policy.endpoint_binding(pool_id)
            rule = next(rule for rule in policy.rules if rule.pool_id == pool_id)
        except (AdmissionMatchError, AdmissionPolicyError) as exc:
            raise RoutingSubmissionError(exc.category) from None
        except StopIteration:
            raise RoutingSubmissionError('routing_override_pool_invalid') from None
        routes.append(
            PlannedLlmRoute(
                job_id=root.id,
                work_unit_id=work_info.id,
                fabric_group_id=policy.fabric_group_id,
                policy_generation=policy.generation,
                policy_digest=policy.policy_digest,
                rule_digest=rule.rule_digest,
                normalized_fact_digest=facts.digest,
                effective_page_count=_optional_int(
                    facts.values.get('document.effective_page_count')
                ),
                pool_id=pool_id,
                logical_endpoint_group_id=binding.endpoint_group_id,
                endpoint_revision=binding.revision,
                routing_source=(
                    'operator-override' if override is not None else 'automatic'
                ),
                routing_actor=override.actor if override is not None else None,
                routing_reason=override.reason if override is not None else None,
            )
        )
        routed_ids.add(work_info.id)
    return tuple(routes)


def resolve_planned_route(
    routes: Sequence[PlannedLlmRoute],
    work_unit_id: str,
    *,
    parent_work_unit_id: str | None = None,
) -> PlannedLlmRoute:
    identity = parent_work_unit_id or work_unit_id
    matches = [route for route in routes if route.work_unit_id == identity]
    if len(matches) != 1:
        raise RoutingSubmissionError('routing_manifest_missing')
    return matches[0]


def _optional_int(value: object) -> int | None:
    return value if type(value) is int else None


def _digest(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(',', ':'),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
