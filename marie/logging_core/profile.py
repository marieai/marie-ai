import time
import typing
from functools import wraps
from typing import Callable, Optional, Union

if typing.TYPE_CHECKING:
    from marie.logging_core.logger import MarieLogger

import torch
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    Task,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Column
from rich.text import Text

from marie.constants import __windows__
from marie.helper import colored, get_readable_size, get_readable_time, get_rich_console


def used_memory(unit: int = 1024 * 1024 * 1024) -> float:
    """
    Get the memory usage of the current process.

    :param unit: Unit of the memory, default in Gigabytes.
    :return: Memory usage of the current process.
    """
    if __windows__:
        # TODO: windows doesn't include `resource` module
        return 0

    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / unit


def used_memory_readable() -> str:
    """
    Get the memory usage of the current process in a human-readable format.

    :return: Memory usage of the current process.
    """
    return get_readable_size(used_memory(1))


def profiling(func):
    """
    Create the Decorator to mark a function for profiling. The time and memory usage will be recorded and printed.

    Example:
    .. highlight:: python
    .. code-block:: python

        @profiling
        def foo():
            print(1)

    :param func: function to be profiled
    :return: arguments wrapper
    """
    from marie.logging_core.predefined import default_logger

    @wraps(func)
    def arg_wrapper(*args, **kwargs):
        start_t = time.perf_counter()
        start_mem = used_memory(unit=1)
        r = func(*args, **kwargs)
        elapsed = time.perf_counter() - start_t
        end_mem = used_memory(unit=1)
        # level_prefix = ''.join('-' for v in inspect.stack() if v and v.index is not None and v.index >= 0)
        level_prefix = ""
        mem_status = f"memory Δ {get_readable_size(end_mem - start_mem)} {get_readable_size(start_mem)} -> {get_readable_size(end_mem)}"
        default_logger.info(
            f"{level_prefix} {func.__qualname__} time: {elapsed}s {mem_status}"
        )
        return r

    return arg_wrapper


class ProgressBar(Progress):
    """
    A progress bar made with rich.

    Example:
        .. highlight:: python
        .. code-block:: python

            with ProgressBar(100, 'loop') as p_bar:
                for i in range(100):
                    do_busy()
                    p_bar.update()
    """

    def __init__(
        self,
        description: str = "Working...",
        total_length: Optional[float] = None,
        message_on_done: Optional[Union[str, Callable[..., str]]] = None,
        columns: Optional[Union[str, ProgressColumn]] = None,
        disable: bool = False,
        console: Optional[Console] = None,
        **kwargs,
    ):
        """Init a custom progress bar based on rich. This is the default progress bar of jina if you want to  customize
        it you should probably just use a rich `Progress` and add your custom column and task

        :param description: description of your task ex : 'Working...'
        :param total_length: the number of steps
        :param message_on_done: The message that you want to print at the end of your task. It can either be a string to
        be formatted with task (ex '{task.completed}') task or a function which take task as input (ex : lambda task : f'{task.completed}'
        :param columns: If you want to customize the column of the progress bar. Note that you should probably directly use
        rich Progress object than overwriting these columns parameters.
        :param total_length: disable the progress bar



        .. # noqa: DAR202
        .. # noqa: DAR101
        .. # noqa: DAR003

        """

        def _default_message_on_done(task):
            return f"{task.completed} steps done in {get_readable_time(seconds=task.finished_time)}"

        columns = columns or [
            SpinnerColumn(),
            _OnDoneColumn(f"DONE", description, "progress.description"),
            BarColumn(complete_style="green", finished_style="yellow"),
            TimeElapsedColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TextColumn("ETA:", style="progress.remaining"),
            TimeRemainingColumn(),
            _OnDoneColumn(
                message_on_done if message_on_done else _default_message_on_done
            ),
        ]

        if not console:
            console = get_rich_console()

        super().__init__(*columns, console=console, disable=disable, **kwargs)

        self.task_id = self.add_task(
            "Working...", total=total_length if total_length else 100.0
        )

    def update(
        self,
        task_id: Optional[TaskID] = None,
        advance: float = 1,
        *args,
        **kwargs,
    ):
        """Update the progress bar

        :param task_id: the task to update
        :param advance: Add a value to main task.completed


        .. # noqa: DAR202
        .. # noqa: DAR101
        .. # noqa: DAR003
        """
        if not task_id:
            super().update(self.task_id, advance=advance, *args, **kwargs)
        else:
            super().update(task_id, advance=advance, *args, **kwargs)


class _OnDoneColumn(ProgressColumn):
    """Renders custom on done for jina progress bar."""

    def __init__(
        self,
        text_on_done_format: Union[str, Callable],
        text_init_format: str = "",
        style: Optional[str] = None,
        table_column: Optional[Column] = None,
    ):
        """
        Create a ProgressBar column with a final message

        Example:
        .. highlight:: python
        .. code-block:: python

            def on_done(task):
                return f'{task.completed} steps done in {task.finished_time:.0f} seconds'


            column = _OnDoneColumn(text_on_done_format=on_done)  # functional

            column = _OnDoneColumn(
                text_on_done_format='{task.completed} steps done in {task.finished_time:.0f} seconds'
            )  # formatting


        :param text_on_done_format: message_on_done
        :param text_init_format: string to be formatted with task or a function which take task as input
        :param style: rich style for the Text
        :param table_column: rich table column
        """
        super().__init__(table_column)
        self.text_on_done_format = text_on_done_format
        self.text_init_format = text_init_format
        self.style = style

    def render(self, task: "Task") -> Text:
        if task.finished_time:
            if callable(self.text_on_done_format):
                return Text(self.text_on_done_format(task), style=self.style)
            else:
                return Text(
                    self.text_on_done_format.format(task=task), style=self.style
                )
        else:
            return Text(self.text_init_format.format(task=task), style=self.style)


class TimeContext:
    """Timing a code snippet with a context manager."""

    def __init__(self, task_name: str, logger: "MarieLogger" = None):
        """
        Create the context manager to timing a code snippet.

        :param task_name: The context/message.
        :param logger: Use existing logger or use naive :func:`print`.

        Example:
        .. highlight:: python
        .. code-block:: python

            with TimeContext('loop'):
                do_busy()

        """
        self.task_name = task_name
        self._logger = logger
        self.duration = 0

    def __enter__(self):
        self.start = time.perf_counter()
        self._enter_msg()
        return self

    def _enter_msg(self):
        if self._logger:
            self._logger.info(self.task_name + "...")
        else:
            print(self.task_name, end=" ...\t", flush=True)

    def __exit__(self, typ, value, traceback):
        self.duration = self.now()

        self.readable_duration = get_readable_time(seconds=self.duration)

        self._exit_msg()

    def now(self) -> float:
        """
        Get the passed time from start to now.

        :return: passed time
        """
        return time.perf_counter() - self.start

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
