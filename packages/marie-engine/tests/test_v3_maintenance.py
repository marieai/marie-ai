"""One-attempt affirmative reconciliation against both owned stores."""

import pytest
from test_request_store import request_for, run_request, store


def test_preview_apply_and_idempotency_preserve_sibling(store):
    from marie.engine.llm_queue.maintenance import reconcile_attempt

    req = request_for(store)
    claim, seq = run_request(store, req)
    sibling = request_for(store)
    run_request(store, sibling)
    store.release_owner(store.test_owner)
    args = dict(
        endpoint_id='endpoint',
        attempt_id=req.attempt_id,
        claim_id=claim,
        execution_seq=seq,
        evidence='remote_completed',
        owners_stopped=True,
        endpoint_quiescent=True,
    )
    preview = reconcile_attempt(store, **args)
    assert preview['applied_count'] == 0
    assert store.usage()['reserved_items'] == 2
    result = reconcile_attempt(store, **args, apply=True)
    assert result['applied_count'] == 1
    assert result['before']['reserved_items'] == 2
    assert result['after']['reserved_items'] == 1
    assert reconcile_attempt(store, **args, apply=True)['applied_count'] == 0
    assert store.metadata(sibling.attempt_id).state == 'executing'


@pytest.mark.parametrize(
    'change',
    [
        dict(endpoint_id='wrong'),
        dict(claim_id='wrong'),
        dict(execution_seq=2),
        dict(owners_stopped=False),
        dict(endpoint_quiescent=False),
        dict(evidence='uncertainty_elapsed'),
    ],
)
def test_reconciliation_rejects_wrong_evidence_or_identity(store, change):
    from marie.engine.llm_queue.maintenance import reconcile_attempt

    req = request_for(store)
    claim, seq = run_request(store, req)
    store.release_owner(store.test_owner)
    args = dict(
        endpoint_id='endpoint',
        attempt_id=req.attempt_id,
        claim_id=claim,
        execution_seq=seq,
        evidence='remote_completed',
        owners_stopped=True,
        endpoint_quiescent=True,
        apply=True,
    )
    with pytest.raises(ValueError):
        reconcile_attempt(store, **(args | change))
    assert store.usage()['reserved_items'] == 1


def test_reconciliation_refuses_competing_owner(store):
    from marie.engine.llm_queue.maintenance import reconcile_attempt

    req = request_for(store)
    claim, seq = run_request(store, req)
    with pytest.raises(RuntimeError, match='owner'):
        reconcile_attempt(
            store,
            endpoint_id='endpoint',
            attempt_id=req.attempt_id,
            claim_id=claim,
            execution_seq=seq,
            evidence='remote_cancelled',
            owners_stopped=True,
            endpoint_quiescent=True,
            apply=True,
        )
    assert store.usage()['reserved_items'] == 1


def test_ambiguous_settlement_is_not_retried_and_stale_owner_is_fenced(store):
    from marie.engine.llm_queue.maintenance import reconcile_attempt
    from marie.engine.llm_queue.store import StaleOwner, StoreUnavailable

    req = request_for(store)
    claim, seq = run_request(store, req)
    store.release_owner(store.test_owner)
    original = store.settle_remote
    calls = []

    def lost(*args, **kwargs):
        calls.append(1)
        original(*args, **kwargs)
        raise StoreUnavailable('synthetic lost response')

    store.settle_remote = lost
    args = dict(
        endpoint_id='endpoint',
        attempt_id=req.attempt_id,
        claim_id=claim,
        execution_seq=seq,
        evidence='remote_cancelled',
        owners_stopped=True,
        endpoint_quiescent=True,
        apply=True,
    )
    with pytest.raises(StoreUnavailable):
        reconcile_attempt(store, **args)
    assert len(calls) == 1
    assert store.usage()['reserved_items'] == 0
    store.settle_remote = original
    assert reconcile_attempt(store, **args)['applied_count'] == 0
    with pytest.raises(StaleOwner):
        original(
            store.test_owner,
            req.attempt_id,
            claim_id=claim,
            execution_seq=seq,
            evidence='remote_cancelled',
        )


def test_preview_reports_endpoint_accounting_and_never_reads_content(
    store, monkeypatch
):
    from marie.engine.llm_queue.maintenance import reconcile_attempt

    req = request_for(store)
    claim, seq = run_request(store, req)
    store.release_owner(store.test_owner)

    def forbidden(*args, **kwargs):
        raise AssertionError('Content must not be read by maintenance')

    monkeypatch.setattr(store, 'fetch_payload', forbidden)
    monkeypatch.setattr(store, 'read_result', forbidden)
    report = reconcile_attempt(
        store,
        endpoint_id='endpoint',
        attempt_id=req.attempt_id,
        claim_id=claim,
        execution_seq=seq,
        evidence='remote_completed',
        owners_stopped=True,
        endpoint_quiescent=True,
    )
    assert report['endpoint_before']['reserved_items'] == 1
    assert report['endpoint_after']['reserved_items'] == 1


def test_module_command_previews_then_applies_same_identity(store):
    import json
    import os
    import subprocess
    import sys

    req = request_for(store)
    claim, seq = run_request(store, req)
    store.release_owner(store.test_owner)
    port = store.client.connection_pool.connection_kwargs['port']
    environment = dict(os.environ, LLM_QUEUE_URL=f'redis://127.0.0.1:{port}/0')
    command = [
        sys.executable,
        '-m',
        'marie.engine.llm_queue.maintenance',
        '--fabric',
        store.keys.fabric_id,
        '--endpoint',
        'endpoint',
        '--attempt',
        req.attempt_id,
        '--claim',
        claim,
        '--execution-seq',
        str(seq),
        '--evidence',
        'remote_completed',
        '--owners-stopped',
        '--endpoint-quiescent',
    ]
    preview = subprocess.run(
        command, env=environment, capture_output=True, text=True, timeout=5, check=True
    )
    assert json.loads(preview.stdout)['applied_count'] == 0
    applied = subprocess.run(
        command + ['--apply'],
        env=environment,
        capture_output=True,
        text=True,
        timeout=5,
        check=True,
    )
    assert json.loads(applied.stdout)['applied_count'] == 1
    assert store.usage()['reserved_items'] == 0
