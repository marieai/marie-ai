import asyncio
import contextvars
import queue
import threading
from typing import Any, Coroutine, TypeVar

T = TypeVar("T")


class AsyncLoopRunner:
    """Run coroutines on one long-lived event loop."""

    def __init__(self, name: str) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ready = threading.Event()
        self._closed = False
        self._thread = threading.Thread(target=self._run_loop, name=name, daemon=True)
        self._thread.start()
        self._ready.wait()

    def run(self, coroutine: Coroutine[Any, Any, T]) -> T:
        if self._closed or self._loop is None:
            coroutine.close()
            raise RuntimeError("Async loop runner is closed")
        return asyncio.run_coroutine_threadsafe(coroutine, self._loop).result()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if threading.current_thread() is not self._thread:
            self._thread.join()

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        self._ready.set()
        try:
            loop.run_forever()
        finally:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()


def run_coroutine_in_current_loop(coroutine):
    """
    Runs `coroutine` to completion, even if we're inside a running loop.
    - Outside any loop: uses asyncio.run()
    - Inside a loop: spins up a fresh loop in a background thread,
      runs `coroutine`, shuts down async generators, then drains
      any *other* pending tasks before closing.
    """
    try:
        # If no loop is running here, just run normally.
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)

    # Capture caller's OTel context so the child thread inherits trace_id.
    ctx = contextvars.copy_context()
    result_q = queue.Queue()

    def _thread_target():
        def _run():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)

            async def _runner():
                result = await coroutine
                # 2) clean up any async generators
                await new_loop.shutdown_asyncgens()

                # 3) drain *other* pending tasks, excluding this one
                current = asyncio.current_task()
                pending = [
                    t for t in asyncio.all_tasks() if not t.done() and t is not current
                ]
                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)

                return result

            try:
                res = new_loop.run_until_complete(_runner())
                result_q.put((True, res))
            except Exception as exc:
                result_q.put((False, exc))
            finally:
                new_loop.close()

        ctx.run(_run)

    t = threading.Thread(target=_thread_target)
    t.start()
    t.join()

    ok, payload = result_q.get()
    if not ok:
        raise payload
    return payload
