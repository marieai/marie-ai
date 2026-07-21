from __future__ import annotations

import asyncio
from collections.abc import Coroutine, Mapping
from typing import Any

from marie.logging_core.logger import MarieLogger


class SchedulerRuntime:
    """Own scheduler background tasks from creation through cancellation."""

    def __init__(self, logger: MarieLogger) -> None:
        self._logger = logger
        self._tasks: dict[str, asyncio.Task[Any]] = {}
        self._event_tasks: set[asyncio.Task[Any]] = set()

    def create_task(
        self,
        coroutine: Coroutine[Any, Any, Any],
        *,
        name: str,
    ) -> asyncio.Task[Any]:
        existing = self._tasks.get(name)
        if existing is not None and not existing.done():
            coroutine.close()
            raise RuntimeError(f'Scheduler task is already running: {name}')

        task = asyncio.create_task(coroutine, name=name)
        self._tasks[name] = task
        return task

    def track_event_task(self, task: asyncio.Task[Any]) -> None:
        self._event_tasks.add(task)

    def discard_event_task(self, task: asyncio.Task[Any]) -> None:
        self._event_tasks.discard(task)

    def tasks(self, *, prefix: str | None = None) -> list[asyncio.Task[Any]]:
        named = [
            task
            for name, task in self._tasks.items()
            if prefix is None or name.startswith(prefix)
        ]
        if prefix is None:
            named.extend(self._event_tasks)
        return list(dict.fromkeys(named))

    async def stop(
        self,
        service_stops: Mapping[str, Coroutine[Any, Any, Any]],
        *,
        timeout: float,
    ) -> None:
        background_tasks = self.tasks()
        for task in background_tasks:
            if not task.done():
                task.cancel()

        service_tasks = [
            asyncio.create_task(coroutine, name=name)
            for name, coroutine in service_stops.items()
        ]
        shutdown_tasks = background_tasks + service_tasks
        if shutdown_tasks:
            _, pending = await asyncio.wait(
                shutdown_tasks,
                timeout=max(0.0, timeout),
            )
            if pending:
                task_names = sorted(task.get_name() for task in pending)
                self._logger.warning(
                    'Scheduler shutdown timed out; cancelling tasks: %s',
                    ', '.join(task_names),
                )
                for task in pending:
                    task.cancel()

            results = await asyncio.gather(*shutdown_tasks, return_exceptions=True)
            for task, result in zip(shutdown_tasks, results):
                if isinstance(result, Exception):
                    self._logger.error(
                        'Task %s failed during scheduler shutdown: %s',
                        task.get_name(),
                        result,
                    )

        self._tasks.clear()
        self._event_tasks.clear()
