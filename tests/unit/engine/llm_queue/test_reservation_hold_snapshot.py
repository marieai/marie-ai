from marie.engine.llm_queue.registry import _clean, _clean_state_counts


def test_hold_diagnostics_survive_snapshot_sanitization():
    assert _clean_state_counts(
        {'abandoned': 2, 'failed': 1, 'cancelled': 1, 'expired': 0}
    ) == {
        'ready': 0,
        'claimed': 0,
        'running': 0,
        'unknown': 0,
        'abandoned': 2,
        'failed': 1,
        'cancelled': 1,
    }
    assert _clean({'held_reservations_sampled': 2, 'state_counts_truncated': True}) == {
        'held_reservations_sampled': 2,
        'state_counts_truncated': True,
    }
