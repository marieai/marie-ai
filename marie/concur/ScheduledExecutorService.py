import asyncio
import functools
import logging
import signal
import threading
from abc import ABC, abstractmethod
from contextlib import suppress
from enum import Enum
from typing import Any, Callable, List

from marie.helper import iscoroutinefunction, run_async


class TimeUnit(Enum):
    MILLISECONDS = 0


class ScheduledExecutorService:
    """
    Scheduler execution service
    This is based on Java Executor Service
    """

    @staticmethod
    def new_scheduled_thread_pool() -> Any:
        pass

    @staticmethod
    def new_scheduled_asyncio_pool() -> "ScheduledAsyncioExecutorService":
        return ScheduledAsyncioExecutorService()

    @abstractmethod
    def schedule(
        self,
        func: Callable,
        initial_delay: int,
        delay: int,
        unit: TimeUnit,
        *args,
        **kwargs,
    ):
        pass

    @abstractmethod
    def schedule_with_fixed_delay(
        self,
        func: Callable,
        initial_delay: int,
        delay: int,
        unit: TimeUnit,
        *args,
        **kwargs,
    ) -> Any:
        """
        Creates and executes a periodic action that becomes enabled first after the given initial delay,
        and subsequently with the given period.

        Args:
            func:  the task to execute
            initial_delay:  the time to delay first execution
            delay: the period between successive executions
            unit:  the time unit of the initial_delay and period parameters
        Return:
            a ScheduledTask representing pending completion of the task
        """
        pass

    @abstractmethod
    def schedule_at_fixed_rate(
        self, func: Callable, delay: int, unit: TimeUnit, *args, **kwargs
    ):
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Initiates an orderly shutdown in which previously submitted tasks are executed, but no new tasks will be accepted.
        Invocation has no additional effect if already shut down."""
        pass

    @abstractmethod
    def shutdown_now(self) -> List[Callable]:
        """Attempts to stop all actively executing tasks, halts the processing of waiting tasks,
        and returns a list of the tasks that were awaiting execution."""
        pass


async def repeat(interval: float, func: Callable, *args, **kwargs):
    """Run the func every interval seconds
    If func has not finished before the interval then the func will run immediately.

    This method accounts for time drifts when scheduled task run for extended period of time.
    """
    while True:
        await asyncio.gather(func(*args, **kwargs), asyncio.sleep(interval))


class ScheduledTask:
    def __init__(
        self, interval: float, func: Callable, task_name: str, *args, **kwargs
    ):
        self.interval = interval
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.is_started = False
        self._task = None
        self.task_name = task_name

    @property
    def task(self):
        return self._task

    def start(self):
        if not self.is_started:
            self.is_started = True
            self._task = asyncio.create_task(self._run(), name=self.task_name)

    async def stop(self):
        if self.is_started:
            self.is_started = False
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task

    async def _run(self):
        """Run the func every interval seconds
        If func has not finished before the interval then the func will run immediately.

        This method accounts for time drifts when scheduled task run for extended period of time.
        """

        # RuntimeWarning: coroutine cancel was never awaited
        if False:
            tasks = list(
                map(
                    asyncio.create_task,
                    [
                        self.func(*self.args, **self.kwargs),
                        asyncio.sleep(self.interval),
                    ],
                )
            )
            print(tasks)

            while True:
                try:
                    await asyncio.gather(*tasks)
                finally:
                    for t in tasks:
                        if not t.done():
                            t.cancel()

        while True:
            await asyncio.gather(
                self.func(*self.args, **self.kwargs), asyncio.sleep(self.interval)
            )

        # return repeat(self.interval, self.func, *self.args, **self.kwargs)


async def shutdown(sig, loop):
    print("caught {0}".format(sig.name))
    tasks = [
        task
        for task in asyncio.Task.all_tasks()
        if task is not asyncio.tasks.Task.current_task()
    ]
    list(map(lambda task: task.cancel(), tasks))
    results = await asyncio.gather(*tasks, return_exceptions=True)
    print("finished awaiting cancelled tasks, results: {0}".format(results))


class ScheduledAsyncioExecutorService(ScheduledExecutorService):
    """This scheduler will create a new Thread with new event loop"""

    def __init__(self, any_event_loop=False, *args, **kwargs) -> None:
        self.logger = logging.Logger
        self.tasks = []

        try:
            self._loop = asyncio.get_event_loop()
            if self._loop.is_closed():
                raise RuntimeError
        except RuntimeError:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

        self.is_cancel = asyncio.Event()
        try:
            for signame in {"SIGINT", "SIGTERM"}:
                # self._loop.add_signal_handler(
                #     getattr(signal, signame),
                #     lambda *args, **kwargs: self.is_cancel.set(),
                # )
                self._loop.add_signal_handler(
                    getattr(signal, signame),
                    functools.partial(
                        asyncio.ensure_future, shutdown(signal.SIGTERM, self._loop)
                    ),
                )
        except (ValueError, RuntimeError) as exc:
            self.logger.warning(
                f" The Scheduler {self.__class__.__name__} will not be able to handle termination signals.  {repr(exc)}"
            )

        self.run_forever()

    def run_forever(self):
        print("Running forever")
        # https://stackoverflow.com/questions/31623194/asyncio-two-loops-for-different-i-o-tasks
        # This will not work as asyncio does not allow nested event loops
        # RuntimeError: This event loop is already running
        kwargs = {"any_event_loop": True}
        # self._loop.run_until_complete(self._loop_body())

    async def _loop_body(self):
        try:
            await asyncio.gather(self.async_run_forever(), self._wait_for_cancel())
        except asyncio.CancelledError:
            self.logger.warning("received terminate ctrl message from main process")

    @abstractmethod
    async def async_cancel(self):
        """An async method to cancel async_run_forever."""
        ...

    @abstractmethod
    async def async_run_forever(self):
        """The async method to run until it is stopped."""
        ...

    async def _wait_for_cancel(self):
        # threads are not using asyncio.Event, but threading.Event
        if isinstance(self.is_cancel, asyncio.Event):
            await self.is_cancel.wait()
        else:
            while not self.is_cancel.is_set():
                await asyncio.sleep(0.1)

        await self.async_cancel()

    def schedule_with_fixed_delay(
        self,
        func: Callable,
        initial_delay: float,
        interval: float,
        unit: TimeUnit,
        *args,
        **kwargs,
    ) -> ScheduledTask:

        raise NotImplementedError

    def schedule_at_fixed_rate(
        self, func: Callable, interval: float, unit: TimeUnit, *args, **kwargs
    ) -> ScheduledTask:
        if not iscoroutinefunction(func):
            raise RuntimeError
        # task = asyncio.ensure_future(repeat(interval, func, *args, **kwargs))
        # return task

        scheduled_task = ScheduledTask(
            interval, func, f"scheduled-task-{len(self.tasks)}", *args, **kwargs
        )

        scheduled_task.start()

        self.tasks.append(scheduled_task)
        return scheduled_task

    def schedule(
        self,
        func: Callable,
        initial_delay: int,
        delay: int,
        unit: TimeUnit,
        *args,
        **kwargs,
    ):
        raise NotImplementedError

    def shutdown_now(self) -> List[Callable]:
        # await asyncio.gather(*tasks)
        raise NotImplementedError

    def shutdown(self) -> None:
        try:
            if not self._loop.is_closed():
                self._loop.close()
        except RuntimeError:
            # no loop
            pass
