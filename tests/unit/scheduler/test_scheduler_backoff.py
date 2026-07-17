import pytest

from marie.scheduler import util

MIN_POLL_PERIOD = 0.25
MAX_POLL_PERIOD = 8.0


def idle_backoff(wait_time: float, idle_streak: int) -> float:
    return util.adjust_backoff(
        wait_time,
        idle_streak,
        scheduled=False,
        min_poll_period=MIN_POLL_PERIOD,
        max_poll_period=MAX_POLL_PERIOD,
    )


def test_idle_backoff_grows_to_the_maximum(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(util.random, "uniform", lambda _low, _high: 1.0)

    wait_time = MIN_POLL_PERIOD
    observed = []
    for idle_streak in range(1, 8):
        wait_time = idle_backoff(wait_time, idle_streak)
        observed.append(wait_time)

    assert observed == pytest.approx([0.4, 0.68, 1.224, 2.3256, 4.6512, 8.0, 8.0])


def test_idle_backoff_applies_jitter_before_clamping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(util.random, "uniform", lambda _low, _high: 0.9)

    assert idle_backoff(1.0, 0) == pytest.approx(1.35)
    assert idle_backoff(0.0, 0) == MIN_POLL_PERIOD


def test_idle_backoff_never_exceeds_the_maximum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(util.random, "uniform", lambda _low, _high: 1.1)

    assert idle_backoff(MAX_POLL_PERIOD, 100) == MAX_POLL_PERIOD


@pytest.mark.parametrize(
    ("wait_time", "expected"),
    [(0.0, 0.25), (0.25, 0.25), (8.0, 4.0), (100.0, 8.0)],
)
def test_scheduled_backoff_halves_within_bounds(
    wait_time: float, expected: float
) -> None:
    assert (
        util.adjust_backoff(
            wait_time,
            idle_streak=0,
            scheduled=True,
            min_poll_period=MIN_POLL_PERIOD,
            max_poll_period=MAX_POLL_PERIOD,
        )
        == expected
    )
