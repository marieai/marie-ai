from __future__ import annotations

from copy import deepcopy

import pytest
from marie.engine.llm_queue.admission_policy import (
    AdmissionMatchError,
    AdmissionPolicy,
    AdmissionPolicyError,
)


def _row(
    pool_id: str,
    priority: int,
    match: dict[str, dict[str, object]],
    *,
    enabled: bool = True,
    accepting: bool = True,
) -> dict[str, object]:
    return {
        'pool_id': pool_id,
        'enabled': enabled,
        'metadata': {
            'label': pool_id,
            'admission': {
                'schema_version': 1,
                'priority': priority,
                'accepting': accepting,
                'match': match,
            },
        },
    }


def _catch_all() -> dict[str, object]:
    return _row('default', 1_000_000, {})


def _policy(*rows: dict[str, object]) -> AdmissionPolicy:
    return AdmissionPolicy.from_rows('default', 7, [*rows, _catch_all()])


def test_match_uses_lowest_priority_and_inclusive_page_bounds() -> None:
    policy = _policy(
        _row(
            'document-small',
            100,
            {'document.effective_page_count': {'gte': 1, 'lte': 4}},
        )
    )

    assert policy.match({'document.effective_page_count': 4}).pool_id == (
        'document-small'
    )
    assert policy.match({'document.effective_page_count': 5}).pool_id == 'default'


def test_match_combines_fields_with_and() -> None:
    policy = _policy(
        _row(
            'interactive-validation',
            10,
            {
                'workload.mode': {'eq': 'interactive'},
                'pipeline.stage': {'in': ['validate', 'enrich']},
            },
        )
    )

    assert (
        policy.match(
            {'workload.mode': 'interactive', 'pipeline.stage': 'validate'}
        ).pool_id
        == 'interactive-validation'
    )
    assert (
        policy.match({'workload.mode': 'batch', 'pipeline.stage': 'validate'}).pool_id
        == 'default'
    )


def test_missing_optional_fact_makes_condition_false() -> None:
    policy = _policy(
        _row('interactive', 10, {'workload.mode': {'eq': 'interactive'}})
    )

    assert policy.match({'workload.kind': 'document'}).pool_id == 'default'


def test_disabled_and_nonaccepting_pools_do_not_match() -> None:
    policy = _policy(
        _row(
            'disabled',
            10,
            {'workload.kind': {'eq': 'document'}},
            enabled=False,
            accepting=False,
        ),
        _row(
            'draining',
            20,
            {'workload.kind': {'eq': 'document'}},
            accepting=False,
        ),
    )

    assert policy.match({'workload.kind': 'document'}).pool_id == 'default'


@pytest.mark.parametrize(
    ('rows', 'category'),
    [
        (
            [
                _row(
                    'small-a',
                    100,
                    {'document.effective_page_count': {'gte': 1, 'lte': 4}},
                ),
                _row(
                    'small-b',
                    100,
                    {'document.effective_page_count': {'gte': 4, 'lte': 8}},
                ),
                _catch_all(),
            ],
            'routing_policy_ambiguous',
        ),
        (
            [
                _row(
                    'small',
                    100,
                    {'document.effective_page_count': {'lte': 4}},
                )
            ],
            'routing_policy_catch_all_required',
        ),
        (
            [
                _row('unknown', 100, {'document.color': {'eq': 'blue'}}),
                _catch_all(),
            ],
            'routing_policy_unknown_fact',
        ),
        (
            [
                _row('bad-string', 100, {'workload.kind': {'gte': 1}}),
                _catch_all(),
            ],
            'routing_policy_operator_type_mismatch',
        ),
        (
            [
                _row('empty-in', 100, {'workload.mode': {'in': []}}),
                _catch_all(),
            ],
            'routing_policy_value_invalid',
        ),
        (
            [
                _row(
                    'eq-in',
                    100,
                    {'pipeline.stage': {'eq': 'extract', 'in': ['extract']}},
                ),
                _catch_all(),
            ],
            'routing_policy_operator_conflict',
        ),
        (
            [
                _row(
                    'reverse-range',
                    100,
                    {'document.effective_page_count': {'gte': 5, 'lte': 4}},
                ),
                _catch_all(),
            ],
            'routing_policy_value_invalid',
        ),
        (
            [
                _row(
                    'unknown-key',
                    100,
                    {'workload.kind': {'eq': 'document', 'contains': 'doc'}},
                ),
                _catch_all(),
            ],
            'routing_policy_unknown_key',
        ),
        (
            [
                _row(
                    'enabled-but-not-accepting',
                    1_000_000,
                    {},
                    accepting=False,
                )
            ],
            'routing_policy_catch_all_required',
        ),
        (
            [
                _row(
                    'disabled-catch-all',
                    1_000_000,
                    {},
                    enabled=False,
                    accepting=True,
                )
            ],
            'routing_policy_pool_disabled',
        ),
    ],
)
def test_invalid_policy_is_rejected(
    rows: list[dict[str, object]], category: str
) -> None:
    with pytest.raises(AdmissionPolicyError, match=category):
        AdmissionPolicy.from_rows('default', 1, rows)


def test_different_priority_rules_may_overlap() -> None:
    policy = AdmissionPolicy.from_rows(
        'default',
        1,
        [
            _row('document', 100, {'workload.kind': {'eq': 'document'}}),
            _row('interactive', 50, {'workload.mode': {'eq': 'interactive'}}),
            _catch_all(),
        ],
    )

    decision = policy.match(
        {'workload.kind': 'document', 'workload.mode': 'interactive'}
    )

    assert decision.pool_id == 'interactive'
    assert decision.matched_facts == ('workload.mode',)


def test_disjoint_same_priority_rules_are_valid() -> None:
    policy = AdmissionPolicy.from_rows(
        'default',
        1,
        [
            _row('small', 100, {'document.effective_page_count': {'lte': 4}}),
            _row('large', 100, {'document.effective_page_count': {'gte': 5}}),
            _catch_all(),
        ],
    )

    assert policy.match({'document.effective_page_count': 4}).pool_id == 'small'
    assert policy.match({'document.effective_page_count': 5}).pool_id == 'large'


def test_policy_and_rule_digests_are_canonical() -> None:
    rows = [
        _row(
            'pipeline',
            100,
            {
                'pipeline.stage': {'in': ['validate', 'extract']},
                'workload.kind': {'eq': 'document'},
            },
        ),
        _catch_all(),
    ]
    reordered = deepcopy(rows)
    match = reordered[0]['metadata']['admission']['match']
    reordered[0]['metadata']['admission']['match'] = {
        'workload.kind': match['workload.kind'],
        'pipeline.stage': {'in': ['extract', 'validate']},
    }

    left = AdmissionPolicy.from_rows('default', 3, rows)
    right = AdmissionPolicy.from_rows('default', 3, list(reversed(reordered)))

    assert left.policy_digest == right.policy_digest
    assert left.to_snapshot() == right.to_snapshot()


def test_fact_digest_is_canonical_and_excludes_unmatched_optional_facts() -> None:
    policy = _policy(
        _row('document', 100, {'workload.kind': {'eq': 'document'}})
    )

    left = policy.match({'workload.kind': 'document', 'request.source': 'workflow'})
    right = policy.match({'request.source': 'workflow', 'workload.kind': 'document'})

    assert left.normalized_fact_digest == right.normalized_fact_digest


@pytest.mark.parametrize(
    'facts',
    [
        {'document.total_page_count': True},
        {'document.total_page_count': 0},
        {'workload.kind': 'unknown-kind'},
        {'request.source': 'x' * 129},
        {'unknown.fact': 'value'},
    ],
)
def test_invalid_runtime_fact_is_rejected(facts: dict[str, object]) -> None:
    policy = _policy()

    with pytest.raises(AdmissionMatchError, match='routing_facts_invalid'):
        policy.match(facts)


def test_duplicate_pool_id_is_rejected() -> None:
    with pytest.raises(AdmissionPolicyError, match='routing_policy_duplicate_pool'):
        AdmissionPolicy.from_rows(
            'default',
            1,
            [_row('same', 10, {'workload.kind': {'eq': 'document'}}), _row('same', 20, {}), _catch_all()],
        )


def test_policy_rejects_more_than_one_thousand_pool_rows() -> None:
    rows = [
        {'pool_id': f'pool-{index}', 'enabled': False, 'metadata': {}}
        for index in range(1000)
    ]
    rows.append(_catch_all())

    with pytest.raises(AdmissionPolicyError, match='routing_policy_pool_limit'):
        AdmissionPolicy.from_rows('default', 1, rows)


def test_runtime_defensively_rejects_same_priority_match() -> None:
    policy = _policy(
        _row('document', 10, {'workload.kind': {'eq': 'document'}}),
        _row('batch', 20, {'workload.mode': {'eq': 'batch'}}),
    )
    duplicate_priority = tuple(
        rule.__class__(
            pool_id=rule.pool_id,
            enabled=rule.enabled,
            priority=10 if rule.pool_id == 'batch' else rule.priority,
            accepting=rule.accepting,
            conditions=rule.conditions,
            rule_digest=rule.rule_digest,
        )
        for rule in policy.rules
    )
    corrupted = policy.__class__(
        fabric_group_id=policy.fabric_group_id,
        generation=policy.generation,
        policy_digest=policy.policy_digest,
        rules=duplicate_priority,
    )

    with pytest.raises(AdmissionMatchError, match='routing_ambiguous'):
        corrupted.match({'workload.kind': 'document', 'workload.mode': 'batch'})
