import threading
from unittest.mock import MagicMock

from marie.serve.runtimes.worker.request_handling import WorkerRequestHandler


class RenewalReleaseRaceSemaphore:
    def __init__(self) -> None:
        self.holder_exists = True
        self.renew_started = threading.Event()
        self.allow_renew = threading.Event()
        self.release_called = threading.Event()
        self.reserve_calls = 0

    def renew(self, *_args, **_kwargs) -> bool:
        self.renew_started.set()
        assert self.allow_renew.wait(timeout=2.0)
        return self.holder_exists

    def reserve(self, **_kwargs) -> bool:
        self.reserve_calls += 1
        self.holder_exists = True
        return True

    def release_owned(self, *_args, **_kwargs) -> bool:
        self.holder_exists = False
        self.release_called.set()
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
        }
    }

    renew_thread = threading.Thread(target=handler._sem_renew_all_if_due)
    renew_thread.start()
    assert semaphore.renew_started.wait(timeout=1.0)

    release_results: list[bool] = []
    release_started = threading.Event()

    def release_ticket() -> None:
        release_started.set()
        release_results.append(handler._sem_untrack("job-1", release=True))

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
