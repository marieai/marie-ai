from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

from marie.engine.llm_queue.admission_policy import (
    AdmissionPolicy,
    FactValue,
    PoolEndpointBinding,
)
from pydantic import BaseModel, ConfigDict, Field

from marie.query_planner.base import Query, QueryPlan
from marie.storage.submission.types import SubmissionDocument

if TYPE_CHECKING:
    from marie.scheduler.models import WorkInfo
    from marie.storage.submission.storage import SubmissionStorage

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


def resolve_submission_document(
    *,
    storage: SubmissionStorage | None,
    document_id: str | None,
    uri: str | None,
    page_count_loader: Any,
    max_bytes: int,
    timeout_seconds: float,
) -> SubmissionDocument:
    document = None
    if storage is not None and document_id:
        document = storage.get_document_by_id(document_id)
        if document is None:
            raise RoutingSubmissionError('routing_facts_missing')
    elif storage is not None and uri:
        document = storage.get_document_by_storage_key(uri)

    if document is not None and uri and document.storage_key != uri:
        raise RoutingSubmissionError('routing_facts_invalid')
    resolved_uri = document.storage_key if document is not None else uri
    if not resolved_uri:
        raise RoutingSubmissionError('routing_facts_missing')
    if document is not None and document.file_size > max_bytes:
        raise RoutingSubmissionError('routing_facts_invalid')

    if document is None:
        document = SubmissionDocument(
            id='',
            submission_id='',
            file_name=resolved_uri.rsplit('/', 1)[-1],
            file_size=0,
            content_type='application/octet-stream',
            storage_key=resolved_uri,
        )
    if document.page_count is None:
        try:
            page_count = page_count_loader(resolved_uri, max_bytes, timeout_seconds)
        except Exception as exc:
            category = (
                'routing_facts_invalid'
                if isinstance(exc, (ValueError, TypeError))
                else 'routing_facts_missing'
            )
            raise RoutingSubmissionError(category) from None
        if type(page_count) is not int or page_count < 1:
            raise RoutingSubmissionError('routing_facts_invalid')
        document.page_count = page_count
        if storage is not None and document.id:
            storage.update_document_page_count(
                document.id, document.storage_key, page_count
            )
    return document


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
        decision = policy.match(facts.values)
        binding = policy.endpoint_binding(decision.pool_id)
        routes.append(
            PlannedLlmRoute(
                job_id=root.id,
                work_unit_id=work_info.id,
                fabric_group_id=policy.fabric_group_id,
                policy_generation=decision.policy_generation,
                policy_digest=decision.policy_digest,
                rule_digest=decision.rule_digest,
                normalized_fact_digest=decision.normalized_fact_digest,
                effective_page_count=_optional_int(
                    facts.values.get('document.effective_page_count')
                ),
                pool_id=decision.pool_id,
                logical_endpoint_group_id=binding.endpoint_group_id,
                endpoint_revision=binding.revision,
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
