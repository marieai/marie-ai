import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from marie.serve.runtimes.worker.request_handling import WorkerRequestHandler
from marie.state.semaphore_store import SemaphoreReleaseResult


class RenewalReleaseRaceSemaphore:
    def __init__(self) -> None:
        self.holder_exists = True
        self.renew_started = threading.Event()
        self.allow_renew = threading.Event()
        self.release_called = threading.Event()
        self.reserve_calls = 0
        self.renew_calls = 0
        self.validate_calls = 0

    def validate_holder(self, *_args, **_kwargs) -> bool:
        self.validate_calls += 1
        return self.holder_exists

    def renew(self, *_args, **_kwargs) -> bool:
        self.renew_calls += 1
        self.renew_started.set()
        assert self.allow_renew.wait(timeout=2.0)
        return self.holder_exists

    def reserve(self, **_kwargs) -> bool:
        self.reserve_calls += 1
        self.holder_exists = True
        return True

    def release_owned_result(self, *_args, **_kwargs) -> SemaphoreReleaseResult:
        self.holder_exists = False
        self.release_called.set()
        return SemaphoreReleaseResult(True, "released", 1)


class ContendedReleaseSemaphore:
    def __init__(self) -> None:
        self.release_calls = 0
        self.renew_calls = 0

    def release_owned_result(self, *_args, **_kwargs) -> SemaphoreReleaseResult:
        self.release_calls += 1
        if self.release_calls == 1:
            return SemaphoreReleaseResult(
                False,
                "counter_contention",
                12,
                retryable=True,
            )
        return SemaphoreReleaseResult(True, "released", 2)

    def renew(self, *_args, **_kwargs) -> bool:
        self.renew_calls += 1
        return True


def test_terminal_release_cannot_be_resurrected_by_concurrent_renewal() -> None:
    semaphore = RenewalReleaseRaceSemaphore()
    handler = object.__new__(WorkerRequestHandler)
    handler.logger = MagicMock()
    handler._semaphore = semaphore
    handler._node = "worker-1"
    handler._sem_default_ttl = 30
    handler._sem_renew_fraction = 0.4
    handler._sem_ticket_lock = threading.Lock()
    handler._active_sem_tickets = {
        "job-1": {
            "slot": "mock_executor_a",
            "ttl": 30,
            "last": 0.0,
            "owner": "job-1",
            "run_attempt_id": "attempt-1",
        }
    }

    renew_thread = threading.Thread(target=handler._sem_renew_all_if_due)
    renew_thread.start()
    assert semaphore.renew_started.wait(timeout=1.0)

    release_results: list[bool] = []
    release_started = threading.Event()

    def release_ticket() -> None:
        release_started.set()
        release_results.append(bool(handler._sem_untrack("job-1", release=True)))

    release_thread = threading.Thread(target=release_ticket)
    release_thread.start()
    assert release_started.wait(timeout=1.0)

    assert not semaphore.release_called.wait(timeout=0.2)

    semaphore.allow_renew.set()
    renew_thread.join(timeout=2.0)
    release_thread.join(timeout=2.0)

    assert not renew_thread.is_alive()
    assert not release_thread.is_alive()
    assert release_results == [True]
    assert semaphore.reserve_calls == 0
    assert semaphore.holder_exists is False
    assert "job-1" not in handler._active_sem_tickets


def test_contended_terminal_release_is_retried_without_renewal() -> None:
    semaphore = ContendedReleaseSemaphore()
    handler = object.__new__(WorkerRequestHandler)
    handler.logger = MagicMock()
    handler._semaphore = semaphore
    handler._deployment = "mock_executor_a"
    handler._sem_default_ttl = 30
    handler._sem_renew_fraction = 0.4
    handler._sem_ticket_lock = threading.Lock()
    handler._active_sem_tickets = {
        "job-1": {
            "slot": "mock_executor_a",
            "ttl": 30,
            "last": 0.0,
            "owner": "job-1",
            "run_attempt_id": "attempt-1",
        }
    }

    first_result = handler._sem_untrack("job-1", release=True)

    assert not first_result
    assert first_result.reason == "counter_contention"
    assert handler._active_sem_tickets["job-1"]["release_pending"] is True

    handler._sem_renew_all_if_due()

    assert semaphore.release_calls == 2
    assert semaphore.renew_calls == 0
    assert "job-1" not in handler._active_sem_tickets


def test_worker_does_not_recreate_missing_attempt_ticket() -> None:
    semaphore = RenewalReleaseRaceSemaphore()
    semaphore.holder_exists = False
    handler = object.__new__(WorkerRequestHandler)
    handler.logger = MagicMock()
    handler._semaphore = semaphore
    handler._node = "worker-1"
    handler._sem_default_ttl = 30
    handler._sem_ticket_lock = threading.Lock()
    handler._active_sem_tickets = {}

    tracked = handler._sem_track(
        "job-1",
        "mock_executor_a",
        run_attempt_id="attempt-1",
    )

    assert tracked is False
    assert semaphore.reserve_calls == 0
    assert semaphore.renew_calls == 0
    assert semaphore.validate_calls == 1
    assert handler._active_sem_tickets == {}


def test_worker_adopts_existing_attempt_ticket_without_renewing() -> None:
    semaphore = RenewalReleaseRaceSemaphore()
    handler = object.__new__(WorkerRequestHandler)
    handler.logger = MagicMock()
    handler._semaphore = semaphore
    handler._node = "worker-1"
    handler._sem_default_ttl = 30
    handler._sem_ticket_lock = threading.Lock()
    handler._active_sem_tickets = {}

    tracked = handler._sem_track(
        "job-1",
        "mock_executor_a",
        run_attempt_id="attempt-1",
    )

    assert tracked is True
    assert semaphore.validate_calls == 1
    assert semaphore.renew_calls == 0
    assert semaphore.reserve_calls == 0
    assert handler._active_sem_tickets["job-1"]["run_attempt_id"] == "attempt-1"


def test_legacy_worker_request_can_reserve_missing_ticket() -> None:
    semaphore = RenewalReleaseRaceSemaphore()
    semaphore.holder_exists = False
    semaphore.allow_renew.set()
    handler = object.__new__(WorkerRequestHandler)
    handler.logger = MagicMock()
    handler._semaphore = semaphore
    handler._node = "worker-1"
    handler._sem_default_ttl = 30
    handler._sem_ticket_lock = threading.Lock()
    handler._active_sem_tickets = {}

    tracked = handler._sem_track(
        "job-1",
        "mock_executor_a",
        run_attempt_id=None,
    )

    assert tracked is True
    assert semaphore.reserve_calls == 1
    assert handler._active_sem_tickets["job-1"]["run_attempt_id"] is None


@pytest.mark.asyncio
async def test_scheduler_managed_request_requires_complete_run_identity() -> None:
    handler = object.__new__(WorkerRequestHandler)
    handler.logger = MagicMock()
    handler.args = SimpleNamespace(name="mock_executor_a/rep-0")
    handler._deployment = "mock_executor_a"
    handler._sem_default_ttl = 30
    handler._job_info_client = None
    handler._set_deployment_status = MagicMock()
    handler._sem_track = MagicMock()
    handler.is_dry_run = MagicMock(return_value=False)

    with pytest.raises(RuntimeError, match="Incomplete durable run identity"):
        await handler._record_started_job(
            "job-1",
            [],
            {"dag_id": "dag-1", "run_owner": "scheduler-1"},
        )

    handler._sem_track.assert_not_called()
