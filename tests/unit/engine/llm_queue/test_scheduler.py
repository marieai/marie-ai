import pytest
from marie.engine.completion_contract import build_completion_call
from marie.engine.llm_queue.scheduler import (
    DrrLaneConfig,
    DrrLaneScheduler,
    prepared_cost_units,
)
from marie.engine.llm_queue.store import ClaimRecord


def test_prepared_cost_uses_numeric_metadata_and_effective_images() -> None:
    call = build_completion_call(model='model', messages=[])
    assert prepared_cost_units(call, {'estimated_cost_units': 16}) == 16
    assert prepared_cost_units(call, {'image_count': 2, 'chunk_page_count': 17}) == 8
    images = build_completion_call(
        model='model',
        messages=[
            {
                'role': 'user',
                'content': [
                    {'type': 'image_url', 'image_url': {'url': 'data:a'}},
                    {'type': 'image_url', 'image_url': {'url': 'data:b'}},
                ],
            }
        ],
    )
    assert prepared_cost_units(images) == 5


@pytest.mark.parametrize(
    'metadata',
    [
        {'image_count': '2'},
        {'estimated_cost_units': 99},
        {'page_count': -1},
        {'image_count': True},
    ],
)
def test_prepared_cost_rejects_invalid_metadata(metadata: dict) -> None:
    call = build_completion_call(model='model', messages=[])
    with pytest.raises(ValueError):
        prepared_cost_units(call, metadata)


def test_drr_quantum_shapes_dispatch_share_for_equal_cost_work() -> None:
    scheduler = DrrLaneScheduler(
        total_concurrent_dispatch=100,
        lanes=[
            DrrLaneConfig('interactive', quantum=4),
            DrrLaneConfig('backfill', quantum=1),
        ],
    )
    selected = []
    for sequence in range(10):
        pool = scheduler.select_metadata({'interactive': 1, 'backfill': 1})
        assert pool is not None
        selected.append(pool)
        scheduler.claimed(pool, 1, charge_sequence=sequence)
        scheduler.release(pool)
        scheduler.retire_charge(sequence)
    assert selected == ['interactive'] * 4 + ['backfill'] + ['interactive'] * 4 + [
        'backfill'
    ]


@pytest.mark.parametrize('document_backlogged', [True, False])
def test_minimum_slots_protect_backlogged_lanes_and_allow_idle_borrowing(
    document_backlogged: bool,
) -> None:
    scheduler = DrrLaneScheduler(
        total_concurrent_dispatch=3,
        lanes=[
            DrrLaneConfig(
                'interactive', quantum=10, min_concurrent=1, max_concurrent=3
            ),
            DrrLaneConfig('document', quantum=10, min_concurrent=1, max_concurrent=3),
        ],
    )
    costs = {'interactive': 1}
    if document_backlogged:
        costs['document'] = 1
    selected = []
    for sequence in range(3):
        pool = scheduler.select_metadata(costs)
        assert pool is not None
        selected.append(pool)
        scheduler.claimed(pool, 1, charge_sequence=sequence)
    assert selected == [
        'interactive',
        'interactive',
        'document' if document_backlogged else 'interactive',
    ]
    assert scheduler.inflight_count == 3
    assert scheduler.select_metadata(costs) is None


def test_minimum_slots_cannot_exceed_global_capacity() -> None:
    with pytest.raises(ValueError, match='min_concurrent'):
        DrrLaneScheduler(
            total_concurrent_dispatch=1,
            lanes=[
                DrrLaneConfig('interactive', min_concurrent=1),
                DrrLaneConfig('backfill', min_concurrent=1),
            ],
        )


def test_high_cost_metadata_accrues_credit_without_transport_reads() -> None:
    scheduler = DrrLaneScheduler(
        total_concurrent_dispatch=1, lanes=[DrrLaneConfig('document', quantum=4)]
    )
    assert scheduler.select_metadata({'document': 9}) == 'document'
    assert scheduler.lane_metadata('document')['deficit'] == 12
    scheduler.claimed('document', 9, charge_sequence=1)
    assert scheduler.lane_metadata('document')['deficit'] == 3
    assert scheduler.inflight_count == 1


def test_per_lane_maximum_is_enforced_against_durable_reservations() -> None:
    scheduler = DrrLaneScheduler(
        total_concurrent_dispatch=10,
        lanes=[
            DrrLaneConfig('backfill', quantum=10, max_concurrent=1),
        ],
    )
    assert scheduler.select_metadata({'backfill': 1}) == 'backfill'
    scheduler.claimed('backfill', 1, charge_sequence=1)
    assert scheduler.select_metadata({'backfill': 1}) is None
    assert scheduler.lane_metadata('backfill')['skip_counts']['lane_capacity'] == 1
    scheduler.release('backfill')
    assert scheduler.select_metadata({'backfill': 1}) == 'backfill'


def test_reconcile_claims_restores_capacity_and_resets_volatile_round() -> None:
    scheduler = DrrLaneScheduler(
        total_concurrent_dispatch=2,
        lanes=[DrrLaneConfig(pool_id="document-small", quantum=4)],
    )
    scheduler.select_metadata({"document-small": 1})
    claim = ClaimRecord(
        attempt_id="attempt-1",
        claim_id="claim-1",
        execution_sequence=0,
        owner_generation=7,
        pool_id="document-small",
        endpoint_group_id="primary",
        charged_cost=4,
        charge_sequence=11,
        refund_state="not_refunded",
    )

    scheduler.reconcile_charges([claim])

    lane = scheduler.lane_metadata("document-small")
    assert scheduler.inflight_count == 1
    assert lane["inflight"] == 1
    assert lane["deficit"] == 0
    assert lane["committed_charge"] == 4


def test_charge_and_refund_sequences_are_applied_once() -> None:
    scheduler = DrrLaneScheduler(
        total_concurrent_dispatch=2,
        lanes=[DrrLaneConfig(pool_id="document-small", quantum=4)],
    )
    assert scheduler.select_metadata({"document-small": 4}) == "document-small"

    assert scheduler.claimed("document-small", 4, charge_sequence=11)
    assert not scheduler.claimed("document-small", 4, charge_sequence=11)
    assert scheduler.refunded("document-small", 4, charge_sequence=11)
    assert not scheduler.refunded("document-small", 4, charge_sequence=11)

    lane = scheduler.lane_metadata("document-small")
    assert lane["deficit"] == 4
    assert lane["committed_charge"] == 0
    assert scheduler._active_charges == {}
