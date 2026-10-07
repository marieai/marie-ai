"""Offline, owner-fenced reconciliation of one affirmatively stopped remote attempt."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Literal
from uuid import uuid4

from marie.engine.llm_queue.queue_keys import validate_identifier
from marie.engine.llm_queue.store import RequestStore, StaleOwner, StoreUnavailable


def reconcile_attempt(
    store: RequestStore,
    *,
    endpoint_id: str,
    attempt_id: str,
    claim_id: str,
    execution_seq: int,
    evidence: Literal['remote_completed', 'remote_cancelled'],
    owners_stopped: bool,
    endpoint_quiescent: bool,
    apply: bool = False,
) -> dict[str, Any]:
    """Preview or settle one original identity without reading payload or result."""
    if not owners_stopped or not endpoint_quiescent:
        raise ValueError(
            'Affirmative owner-stop and endpoint-quiescence evidence required'
        )
    if evidence not in {'remote_completed', 'remote_cancelled'}:
        raise ValueError('Affirmative remote settlement evidence required')
    for identity in (endpoint_id, attempt_id, claim_id):
        validate_identifier(identity)
    if type(execution_seq) is not int or execution_seq < 1:
        raise ValueError('A positive original execution sequence is required')
    owner = store.acquire_owner(
        'maintenance-' + uuid4().hex, lease_ms=min(10_000, store.limits.max_lease_ms)
    )
    if owner is None:
        raise RuntimeError('An owner is still active; stop without mutation')
    try:
        metadata = store.metadata(attempt_id)
        if (
            metadata is None
            or metadata.endpoint_id != endpoint_id
            or metadata.claim_id != claim_id
            or metadata.execution_seq != execution_seq
            or metadata.state
            not in {
                'executing',
                'outcome_unknown',
                'abandoned',
                'succeeded',
                'failed',
                'cancelled',
                'expired',
            }
        ):
            raise ValueError('Original attempt identity or state does not match')
        before = store.usage()
        endpoint_before = store.endpoint_status(endpoint_id)
        disposition = 'preview'
        if apply:
            disposition = store.settle_remote(
                owner,
                attempt_id,
                claim_id=claim_id,
                execution_seq=execution_seq,
                evidence=evidence,
            ).disposition
        after = store.usage()
        endpoint_after = store.endpoint_status(endpoint_id)
        return {
            'schema_version': 1,
            'fabric_id': store.keys.fabric_id,
            'endpoint_id': endpoint_id,
            'attempt_id': attempt_id,
            'claim_id': claim_id,
            'execution_seq': execution_seq,
            'evidence': evidence,
            'disposition': disposition,
            'applied_count': int(disposition == 'settled'),
            'before': before,
            'after': after,
            'endpoint_before': {
                field: int(endpoint_before[field] or 0)
                for field in ('reserved_items', 'reserved_bytes')
            },
            'endpoint_after': {
                field: int(endpoint_after[field] or 0)
                for field in ('reserved_items', 'reserved_bytes')
            },
        }
    finally:
        try:
            store.release_owner(owner)
        except (StaleOwner, StoreUnavailable):
            pass


def main() -> int:
    """Read queue credentials from LLM_QUEUE_URL, never command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('fabric', 'endpoint', 'attempt', 'claim'):
        parser.add_argument('--' + field, required=True, type=validate_identifier)
    parser.add_argument('--execution-seq', type=int, required=True)
    parser.add_argument(
        '--evidence', choices=('remote_completed', 'remote_cancelled'), required=True
    )
    parser.add_argument('--owners-stopped', action='store_true', required=True)
    parser.add_argument('--endpoint-quiescent', action='store_true', required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    if args.execution_seq < 1:
        parser.error('execution-seq must be positive')
    url = os.environ.get('LLM_QUEUE_URL')
    if not url:
        parser.error(
            'LLM_QUEUE_URL must be supplied through the existing secret environment'
        )
    store = None
    try:
        store = RequestStore.for_producer(url, fabric_id=args.fabric)
        report = reconcile_attempt(
            store,
            endpoint_id=args.endpoint,
            attempt_id=args.attempt,
            claim_id=args.claim,
            execution_seq=args.execution_seq,
            evidence=args.evidence,
            owners_stopped=args.owners_stopped,
            endpoint_quiescent=args.endpoint_quiescent,
            apply=args.apply,
        )
        print(json.dumps(report, sort_keys=True))
        return 0
    except (ValueError, RuntimeError, StoreUnavailable, StaleOwner):
        print(
            json.dumps(
                {
                    'schema_version': 1,
                    'disposition': 'unconfirmed',
                    'applied_count': None,
                }
            )
        )
        return 1
    finally:
        if store is not None:
            store.close()


if __name__ == '__main__':
    raise SystemExit(main())
