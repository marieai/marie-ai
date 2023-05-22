import abc
from typing import Any


class Scheduler(abc.ABC):
    """Abstract base class for a scheduler. This component is responsible for interfacing with
    an external system such as cron to ensure scheduled repeated execution according.
    """

    def start_schedule(self) -> Any:
        """
        Starts the scheduler.
        """
        started_state = 1

        return started_state

    def stop_schedule(
        self,
    ) -> Any:
        """
        Stops the scheduler.
        :return:
        """

        stopped_state = 0
        return stopped_state

    @abc.abstractmethod
    def debug_info(self) -> str:
        """Returns debug information about the scheduler."""
