import torch

from marie.helper import colored, get_readable_time
from marie.logging_core.logger import MarieLogger


class TimeContextCuda:
    """Timing a code snippet with a context manager."""

    def __init__(
        self,
        task_name: str,
        logger: "MarieLogger" = None,
        enabled=True,
        callback: callable = None,
    ):
        """
        Create the context manager to timing a code snippet with CUDA.

        :param task_name: The context/message.
        :param logger: Use existing logger or use naive :func:`print`.

        Example:
        .. highlight:: python
        .. code-block:: python

            with TimeContextCuda('loop'):
                do_busy()

        """
        self.enabled = enabled
        self.callback = callback or (lambda x: x)
        self.task_name = task_name
        self._logger = logger
        self.duration = 0
        self.start_cuda = torch.cuda.Event(enable_timing=True)
        self.end_cuda = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        if not self.enabled:
            return self

        torch.cuda.synchronize()
        self.start_cuda.record()

        self._enter_msg()
        return self

    def _enter_msg(self):
        if self._logger:
            self._logger.info(self.task_name + "...")
        else:
            print(self.task_name, end=" ...\t", flush=True)

    def __exit__(self, typ, value, traceback):
        if not self.enabled:
            return self
        self.end_cuda.record()
        torch.cuda.synchronize()
        self.duration = self.start_cuda.elapsed_time(self.end_cuda) / 1000

        self.readable_duration = get_readable_time(seconds=self.duration)
        self.callback(self.duration)
        self._exit_msg()

    def _exit_msg(self):
        if self._logger:
            self._logger.info(
                f"{self.task_name} takes {self.readable_duration} ({self.duration:.3f}s)"
            )
        else:
            print(
                colored(
                    f"{self.task_name} takes {self.readable_duration} ({self.duration:.3f}s)"
                ),
                flush=True,
            )
