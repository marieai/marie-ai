import asyncio
from typing import Coroutine

from marie.helper import get_or_reuse_loop

background_tasks = set()


def run_background_task(coroutine: Coroutine) -> asyncio.Task:
    """Schedule a task reliably to the event loop.

    This API is used when you don't want to cache the reference of `asyncio.Task`.
    For example,

    ```
    get_event_loop().create_task(coroutine(*args))
    ```

    The above code doesn't guarantee to schedule the coroutine to the event loops

    When using create_task in a  "fire and forget" way, we should keep the references
    alive for the reliable execution. This API is used to fire and forget
    asynchronous execution.

    https://docs.python.org/3/library/asyncio-task.html#creating-tasks
    """
    task = get_or_reuse_loop().create_task(coroutine)
    # Add task to the set. This creates a strong reference.
    background_tasks.add(task)

    # To prevent keeping references to finished tasks forever,
    # make each task remove its own reference from the set after
    # completion:
    task.add_done_callback(background_tasks.discard)
    return task
