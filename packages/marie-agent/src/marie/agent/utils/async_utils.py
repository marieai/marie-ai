"""Async helpers that do not depend on the Marie server runtime."""

import asyncio
import inspect
import threading
from collections.abc import Callable, Coroutine
from typing import Any


def get_or_reuse_loop() -> asyncio.AbstractEventLoop:
    """Return the current usable event loop or create one for this thread."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


def run_async(
    func: Callable[..., Coroutine[Any, Any, Any]] | Coroutine[Any, Any, Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Run a coroutine from synchronous code, including inside a running loop."""
    if inspect.iscoroutine(func):
        if args or kwargs:
            raise ValueError(
                "Cannot pass arguments when func is already a coroutine object."
            )
        coroutine = func
    elif callable(func):
        coroutine = func(*args, **kwargs)
    else:
        raise TypeError("func must be a coroutine function or coroutine object")

    class RunThread(threading.Thread):
        result: Any = None
        exception: BaseException | None = None

        def run(self) -> None:
            try:
                self.result = asyncio.run(coroutine)
            except BaseException as exc:
                self.exception = exc

    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        running_loop = None

    if running_loop and running_loop.is_running():
        thread = RunThread()
        thread.start()
        thread.join()
        if thread.exception:
            raise thread.exception
        return thread.result
    return asyncio.run(coroutine)


asyncio_run = run_async

__all__ = ["asyncio_run", "get_or_reuse_loop", "run_async"]
