import threading

import pytest

from marie.job.job_callback_executor import JobCallbackExecutor


@pytest.mark.parametrize(
    ("wait", "release_before_return"),
    [(True, True), (False, False), (True, False)],
    ids=["wait", "no-wait", "join-timeout"],
)
def test_shutdown_drains_a_full_queue(
    monkeypatch, wait: bool, release_before_return: bool
) -> None:
    executor = JobCallbackExecutor(max_queue_size=1, max_workers=1)
    dispatch_started = threading.Event()
    release_dispatch = threading.Event()
    shutdown_returned = threading.Event()
    received: list[int] = []
    pool_submit = executor._executor.submit

    def paused_submit(fn, *args):
        dispatch_started.set()
        assert release_dispatch.wait(5)
        return pool_submit(fn, *args)

    monkeypatch.setattr(executor._executor, "submit", paused_submit)

    def shutdown() -> None:
        executor.shutdown(wait=wait, timeout=0.5)
        shutdown_returned.set()

    shutdown_thread = threading.Thread(target=shutdown, daemon=True)
    try:
        executor.submit(received.append, 1)
        assert dispatch_started.wait(1)
        executor.submit(received.append, 2)
        assert executor._queue.full()

        shutdown_thread.start()
        assert executor._shutdown_event.wait(1)
        executor.submit(received.append, 3)
        if not release_before_return:
            assert shutdown_returned.wait(1)

        release_dispatch.set()
        assert shutdown_returned.wait(2)
        executor._thread.join(1)
        assert not executor._thread.is_alive()
        executor.shutdown(wait=True)
        assert received == [1, 2]
        assert executor._queue.unfinished_tasks == 0
    finally:
        release_dispatch.set()
        if shutdown_thread.ident is not None:
            shutdown_thread.join(2)
        executor.shutdown(wait=False)
        executor._thread.join(2)
        executor.shutdown(wait=True, timeout=0.1)


def test_shutdown_waits_for_an_admitted_submission(monkeypatch) -> None:
    executor = JobCallbackExecutor(max_queue_size=1, max_workers=1)
    submission_started = threading.Event()
    release_submission = threading.Event()
    received: list[int] = []
    queue_put = executor._queue.put

    def paused_put(item, *args, **kwargs):
        submission_started.set()
        assert release_submission.wait(5)
        return queue_put(item, *args, **kwargs)

    monkeypatch.setattr(executor._queue, "put", paused_put)
    submitter = threading.Thread(
        target=executor.submit, args=(received.append, 1), daemon=True
    )
    try:
        submitter.start()
        assert submission_started.wait(1)
        executor.shutdown(wait=False)
        release_submission.set()
        submitter.join(1)
        assert not submitter.is_alive()
        executor.shutdown(wait=True, timeout=1)
        assert not executor._thread.is_alive()
        assert received == [1]
        assert executor._queue.unfinished_tasks == 0
    finally:
        release_submission.set()
        submitter.join(2)
        executor.shutdown(wait=False)
        executor._thread.join(2)
        executor.shutdown(wait=True, timeout=0.1)


def test_empty_shutdown_is_repeatable() -> None:
    executor = JobCallbackExecutor(max_queue_size=1, max_workers=1)
    try:
        executor.shutdown(wait=True, timeout=1)
        executor.shutdown(wait=True, timeout=1)
        assert not executor._thread.is_alive()
        assert executor._queue.unfinished_tasks == 0
    finally:
        executor.shutdown(wait=False)
