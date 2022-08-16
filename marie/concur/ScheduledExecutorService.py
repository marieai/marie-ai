import asyncio
import threading
from abc import abstractmethod, ABC
from enum import Enum
from typing import Callable, Any, List


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
        self, command: Callable, initial_delay: int, delay: int, unit: TimeUnit
    ):
        pass

    @abstractmethod
    def schedule_with_fixed_delay(
        self, function: Callable, initial_delay: int, delay: int, unit: TimeUnit
    ) -> Any:
        """
        Creates and executes a periodic action that becomes enabled first after the given initial delay,
        and subsequently with the given period.

        Args:
            function:  the task to execute
            initial_delay:  the time to delay first execution
            delay: the period between successive executions
            unit:  the time unit of the initial_delay and period parameters
        Return:
            a ScheduledTask representing pending completion of the task
        """
        pass

    @abstractmethod
    def schedule_at_fixed_rate(
        self, command: Callable, initial_delay: int, delay: int, unit: TimeUnit
    ):
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Initiates an orderly shutdown in which previously submitted tasks are executed, but no new tasks will be accepted.
        Invocation has no additional effect if already shut down. """
        pass

    @abstractmethod
    def shutdown_now(self) -> List[Callable]:
        """Attempts to stop all actively executing tasks, halts the processing of waiting tasks,
        and returns a list of the tasks that were awaiting execution."""
        pass


class ScheduledAsyncioExecutorService(ScheduledExecutorService):
    """This scheduler will create a new Thread with new event loop"""

    def __init__(self, any_event_loop=False, *args, **kwargs) -> None:

        try:
            self._loop = asyncio.get_event_loop()
            if self._loop.is_closed():
                raise RuntimeError
        except RuntimeError:
            self._loop = None

        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        self._thread = threading.Thread(target=self.worker)
        self._thread.start()

    def schedule_with_fixed_delay(
        self, function: Callable, initial_delay: int, delay: int, unit: TimeUnit
    ) -> Any:
        pass

    def worker(self):
        print("worker")
        return

    def schedule_at_fixed_rate(
        self, command: Callable, initial_delay: int, delay: int, unit: TimeUnit
    ):
        pass

    def schedule(
        self, command: Callable, initial_delay: int, delay: int, unit: TimeUnit
    ):
        pass

    def shutdown_now(self) -> List[Callable]:
        # await asyncio.gather(*tasks)
        pass

    def shutdown(self) -> None:
        try:
            self._loop.close()
        except RuntimeError:
            # no loop
            pass
