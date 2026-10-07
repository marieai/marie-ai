from types import SimpleNamespace

from marie.engine.llm_queue.request_dispatcher import RequestDispatcher, _InflightClaim
from marie.engine.llm_queue.store import ClaimRecord, RequestMetadata


def test_inflight_snapshot_retains_claimed_request_metadata(monkeypatch):
    monkeypatch.setattr(
        'marie.engine.llm_queue.request_dispatcher.time.time', lambda: 15.0
    )
    claim = ClaimRecord(
        attempt_id='attempt-1',
        claim_id='claim-1',
        execution_sequence=2,
        owner_generation=3,
        pool_id='document-large',
        endpoint_group_id='local-mock',
        charged_cost=4,
        charge_sequence=5,
        refund_state='not_refunded',
    )
    metadata = RequestMetadata(
        attempt_id='attempt-1',
        producer_id='producer-1',
        pool_id='document-large',
        endpoint_id='local-mock',
        config_revision='current',
        state='ready',
        execution_seq=1,
        expires_at_ms=20_000,
        payload_bytes=56_752,
        cost=4,
        claim_id=None,
        owner_id=None,
        next_eligible=0,
        uncertainty_until=0,
        retain_until=0,
        model='gpt-5.2-mock',
        admitted_at_ms=10_000,
    )
    dispatcher = object.__new__(RequestDispatcher)
    dispatcher.store = SimpleNamespace(keys=SimpleNamespace(fabric_id='default'))
    dispatcher.dispatcher_id = 'dispatcher-1'
    dispatcher._inflight_claims = {
        claim.attempt_id: _InflightClaim(claim=claim, metadata=metadata, popped_at=12.0)
    }

    assert dispatcher.inflight_requests_snapshot() == [
        {
            'request_id': 'attempt-1',
            'attempt_id': 'attempt-1',
            'fabric_group_id': 'default',
            'contract_version': 'v3',
            'lifecycle_stage': 'dispatching',
            'state_source': 'dispatcher',
            'dispatcher_id': 'dispatcher-1',
            'pool_id': 'document-large',
            'endpoint_id': 'local-mock',
            'endpoint_group_id': 'local-mock',
            'config_revision': 'current',
            'model': 'gpt-5.2-mock',
            'payload_bytes': 56_752,
            'cost': 4,
            'admitted_at_ms': 10_000,
            'submitted_at': 10.0,
            'expires_at_ms': 20_000,
            'popped_at': 12.0,
            'state_updated_at': 12.0,
            'queue_wait_age_seconds': 2.0,
            'inflight_age_seconds': 3.0,
            'execution_seq': 2,
            'claim_id': 'claim-1',
            'owner_generation': 3,
        }
    ]
