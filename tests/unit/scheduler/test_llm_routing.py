from datetime import datetime, timedelta, timezone

import pytest
from marie.engine.llm_queue.admission_policy import AdmissionPolicy

from marie.query_planner.base import (
    LlmQueryDefinition,
    NoopQueryDefinition,
    Query,
    QueryPlan,
    QueryType,
)
from marie.scheduler.llm_routing import (
    RoutingSubmissionError,
    TrustedRoutingContext,
    normalize_routing_facts,
    plan_llm_routes,
    reject_external_routing_selectors,
    resolve_planned_route,
    resolve_submission_document,
)
from marie.scheduler.models import WorkInfo
from marie.scheduler.state import WorkState
from marie.storage.submission.types import SubmissionDocument


def _document(*, page_count: int | None = 20, storage_key: str = 's3://docs/a.tif'):
    return SubmissionDocument(
        id='document-1',
        submission_id='submission-1',
        file_name='a.tif',
        file_size=100,
        content_type='image/tiff',
        storage_key=storage_key,
        page_count=page_count,
    )


def _work(work_id: str = 'job-1') -> WorkInfo:
    now = datetime.now(timezone.utc)
    return WorkInfo(
        id=work_id,
        name='extract',
        data={'metadata': {}},
        state=WorkState.CREATED,
        retry_limit=1,
        retry_delay=1,
        retry_backoff=False,
        start_after=now,
        expire_in_seconds=0,
        keep_until=now + timedelta(days=1),
    )


def _policy() -> AdmissionPolicy:
    return AdmissionPolicy.from_rows(
        'default',
        7,
        [
            {
                'pool_id': 'document-small',
                'enabled': True,
                'metadata': {
                    'admission': {
                        'schema_version': 1,
                        'priority': 10,
                        'accepting': True,
                        'match': {
                            'document.effective_page_count': {'gte': 1, 'lte': 4}
                        },
                    },
                    'llm_dispatch': {
                        'schema_version': 1,
                        'endpoint_id': 'document-models',
                        'revision': 'r3',
                    },
                },
            },
            {
                'pool_id': 'default',
                'enabled': True,
                'metadata': {
                    'admission': {
                        'schema_version': 1,
                        'priority': 1_000_000,
                        'accepting': True,
                        'match': {},
                    },
                    'llm_dispatch': {
                        'schema_version': 1,
                        'endpoint_id': 'document-models',
                        'revision': 'r3',
                    },
                },
            },
        ],
    )


def test_effective_page_count_deduplicates_validated_requested_pages() -> None:
    facts = normalize_routing_facts(
        document=_document(),
        requested_pages=[0, 2, 2, 19],
        workload_kind='document',
        workload_mode='batch',
        pipeline_stage='extract',
        request_source='gateway-job-api',
    )

    assert facts.values['document.total_page_count'] == 20
    assert facts.values['document.requested_page_count'] == 3
    assert facts.values['document.effective_page_count'] == 3


@pytest.mark.parametrize('pages', [[-1], [20], [], [True]])
def test_requested_pages_must_be_nonempty_in_range_integers(pages) -> None:
    with pytest.raises(RoutingSubmissionError, match='routing_facts_invalid'):
        normalize_routing_facts(
            document=_document(),
            requested_pages=pages,
            workload_kind='document',
            workload_mode='batch',
            pipeline_stage='extract',
            request_source='gateway-job-api',
        )


def test_document_workload_requires_authoritative_page_count() -> None:
    with pytest.raises(RoutingSubmissionError, match='routing_facts_missing'):
        normalize_routing_facts(
            document=_document(page_count=None),
            requested_pages=None,
            workload_kind='document',
            workload_mode='batch',
            pipeline_stage='extract',
            request_source='gateway-job-api',
        )


def test_document_id_must_match_submitted_storage_uri() -> None:
    class Storage:
        def get_document_by_id(self, document_id):
            assert document_id == 'document-1'
            return _document(storage_key='s3://docs/registered.tif')

    with pytest.raises(RoutingSubmissionError, match='routing_facts_invalid'):
        resolve_submission_document(
            storage=Storage(),
            document_id='document-1',
            uri='s3://docs/different.tif',
            page_count_loader=lambda *_: 20,
            max_bytes=1_000,
            timeout_seconds=1.0,
        )


def test_missing_registered_count_is_calculated_and_cached() -> None:
    document = _document(page_count=None)

    class Storage:
        cached = None

        def get_document_by_storage_key(self, uri):
            assert uri == document.storage_key
            return document

        def update_document_page_count(self, document_id, storage_key, page_count):
            self.cached = (document_id, storage_key, page_count)

    storage = Storage()
    resolved = resolve_submission_document(
        storage=storage,
        document_id=None,
        uri=document.storage_key,
        page_count_loader=lambda *_: 6,
        max_bytes=1_000,
        timeout_seconds=1.0,
    )

    assert resolved.page_count == 6
    assert storage.cached == ('document-1', document.storage_key, 6)


@pytest.mark.parametrize(
    'key',
    [
        'pool_id',
        'endpoint_id',
        'LLM_QUEUE_POOL_ID',
        'LLM_QUEUE_CONTRACT_VERSION',
        'policy_generation',
        'replica_id',
    ],
)
def test_external_routing_selector_is_rejected_at_any_metadata_depth(key: str) -> None:
    with pytest.raises(RoutingSubmissionError, match='caller_pool_forbidden'):
        reject_external_routing_selectors({'nested': [{'more': {key: 'value'}}]})


def test_plan_routes_only_llm_nodes_using_node_stage_and_stable_id() -> None:
    root = _work()
    facts = normalize_routing_facts(
        document=_document(page_count=3),
        requested_pages=None,
        workload_kind='document',
        workload_mode='batch',
        pipeline_stage=None,
        request_source='gateway-job-api',
    )
    policy = _policy()
    root.routing_context = TrustedRoutingContext(base_facts=facts, policy=policy)
    noop = Query(
        task_id='node-0',
        query_str='START',
        dependencies=[],
        node_type=QueryType.COMPUTE,
        definition=NoopQueryDefinition(),
    )
    llm = Query(
        task_id='node-1',
        query_str='Validate extracted fields',
        dependencies=['node-0'],
        node_type=QueryType.COMPUTE,
        definition=LlmQueryDefinition(
            model_name='mock',
            endpoint='annotator_llm://validate',
            params={'layout': 'mock', 'pipeline_stage': 'validate'},
        ),
    )
    plan = QueryPlan(nodes=[noop, llm])
    nodes = [_work('node-1'), _work('node-0')]

    routes = plan_llm_routes(
        root=root,
        plan=plan,
        nodes=nodes,
        policy=policy,
        base_facts=facts,
    )

    assert len(routes) == 1
    route = routes[0]
    assert route.job_id == 'job-1'
    assert route.work_unit_id == 'node-1'
    assert route.pool_id == 'document-small'
    assert route.logical_endpoint_group_id == 'document-models'
    assert route.endpoint_revision == 'r3'
    assert route.effective_page_count == 3
    assert root.data['metadata'] == {}


def test_refinement_inherits_parent_route_and_unplanned_node_is_rejected() -> None:
    route = plan_llm_routes(
        root=(root := _work()),
        plan=QueryPlan(
            nodes=[
                Query(
                    task_id='node-1',
                    query_str='Extract',
                    dependencies=[],
                    node_type=QueryType.COMPUTE,
                    definition=LlmQueryDefinition(
                        model_name='mock',
                        endpoint='annotator_llm://extract',
                        params={'layout': 'mock'},
                    ),
                )
            ]
        ),
        nodes=[_work('node-1')],
        policy=(policy := _policy()),
        base_facts=(
            facts := normalize_routing_facts(
                document=_document(page_count=2),
                requested_pages=None,
                workload_kind='document',
                workload_mode='batch',
                pipeline_stage=None,
                request_source='gateway-job-api',
            )
        ),
    )[0]
    root.routing_context = TrustedRoutingContext(base_facts=facts, policy=policy)

    assert resolve_planned_route((route,), 'refine-1', parent_work_unit_id='node-1') == route
    with pytest.raises(RoutingSubmissionError, match='routing_manifest_missing'):
        resolve_planned_route((route,), 'unplanned')
