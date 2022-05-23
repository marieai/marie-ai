import asyncio
import inspect
import os
import threading
import warnings
from typing import Dict, TYPE_CHECKING, Optional, Tuple, Union, Callable

from rich.console import Console


from marie import __windows__

# based on jina


def get_internal_ip():
    """
    Return the private IP address of the gateway for connecting from other machine in the same network.

    :return: Private IP address.
    """
    import socket

    ip = "127.0.0.1"
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            # doesn't even have to be reachable
            s.connect(("10.255.255.255", 1))
            ip = s.getsockname()[0]
    except Exception:
        pass
    return ip


def get_public_ip(timeout: float = 0.3):
    """
    Return the public IP address of the gateway for connecting from other machine in the public network.

    :param timeout: the seconds to wait until return None.

    :return: Public IP address.

    .. warn::
        Set `timeout` to a large number will block the Flow.

    """
    import urllib.request

    results = []

    def _get_ip(url):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as fp:
                _ip = fp.read().decode().strip()
                results.append(_ip)

        except:
            pass  # intentionally ignored, public ip is not showed

    ip_server_list = [
        "https://api.ipify.org",
        "https://ident.me",
        "https://checkip.amazonaws.com/",
    ]

    threads = []

    for idx, ip in enumerate(ip_server_list):
        t = threading.Thread(target=_get_ip, args=(ip,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join(timeout)

    for r in results:
        if r:
            return r


def convert_tuple_to_list(d: Dict):
    """
    Convert all the tuple type values from a dict to list.

    :param d: Dict type of data.
    """
    for k, v in d.items():
        if isinstance(v, tuple):
            d[k] = list(v)
        elif isinstance(v, dict):
            convert_tuple_to_list(v)


def is_jupyter() -> bool:  # pragma: no cover
    """
    Check if we're running in a Jupyter notebook, using magic command `get_ipython` that only available in Jupyter.

    :return: True if run in a Jupyter notebook else False.
    """
    try:
        get_ipython  # noqa: F821
    except NameError:
        return False
    shell = get_ipython().__class__.__name__  # noqa: F821
    if shell == "ZMQInteractiveShell":
        return True  # Jupyter notebook or qtconsole
    elif shell == "Shell":
        return True  # Google colab
    elif shell == "TerminalInteractiveShell":
        return False  # Terminal running IPython
    else:
        return False  # Other type (?)


def iscoroutinefunction(func: Callable):
    return inspect.iscoroutinefunction(func)


def run_async(func, *args, **kwargs):
    """Generalized asyncio.run for jupyter notebook.

    When running inside jupyter, an eventloop is already exist, can't be stopped, can't be killed.
    Directly calling asyncio.run will fail, as This function cannot be called when another asyncio event loop
    is running in the same thread.

    .. see_also:
        https://stackoverflow.com/questions/55409641/asyncio-run-cannot-be-called-from-a-running-event-loop

    call `run_async(my_function, any_event_loop=True, *args, **kwargs)` to enable run with any eventloop

    :param func: function to run
    :param args: parameters
    :param kwargs: key-value parameters
    :return: asyncio.run(func)
    """

    any_event_loop = kwargs.pop("any_event_loop", False)

    class _RunThread(threading.Thread):
        """Create a running thread when in Jupyter notebook."""

        def run(self):
            """Run given `func` asynchronously."""
            self.result = asyncio.run(func(*args, **kwargs))

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # eventloop already exist
        # running inside Jupyter
        if any_event_loop or is_jupyter():
            thread = _RunThread()
            thread.start()
            thread.join()
            try:
                return thread.result
            except AttributeError:
                from marie.excepts import BadClient

                raise BadClient("something wrong when running the eventloop, result can not be retrieved")
        else:

            raise RuntimeError(
                "you have an eventloop running but not using Jupyter/ipython, "
                "this may mean you are using Jina with other integration? if so, then you "
                "may want to use Client/Flow(asyncio=True). If not, then "
                "please report this issue here: https://github.com/jina-ai/jina"
            )
    else:
        return get_or_reuse_loop().run_until_complete(func(*args, **kwargs))


if TYPE_CHECKING:
    from fastapi import FastAPI


def extend_rest_interface(app: "FastAPI") -> "FastAPI":
    """Extend Marie built-in FastAPI instance with customized APIs, routing, etc.

    :param app: the built-in FastAPI instance given by Marie
    :return: the extended FastAPI instance

    .. highlight:: python
    .. code-block:: python

        def extend_rest_interface(app: 'FastAPI'):
            @app.get('/extension1')
            async def root():
                return {"message": "Hello World"}

            return app
    """
    return app


def get_full_version() -> Optional[Tuple[Dict, Dict]]:
    info = {"marie": "-1.-1.-1"}
    return info


def _update_policy():
    if __windows__:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    elif "MARIE_DISABLE_UVLOOP" in os.environ:
        return
    else:
        try:
            import uvloop

            if not isinstance(asyncio.get_event_loop_policy(), uvloop.EventLoopPolicy):
                asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        except ModuleNotFoundError:
            warnings.warn('Install `uvloop` via `pip install "marie[uvloop]"` for better performance.')


def _close_loop():
    try:
        loop = asyncio.get_event_loop()
        if not loop.is_closed():
            loop.close()
    except RuntimeError:
        # there is no loop, so nothing to do here
        pass


# workaround for asyncio loop and fork issue: https://github.com/python/cpython/issues/66197
# we close the loop after forking to avoid reusing the parents process loop
# a new loop should be created in the child process
os.register_at_fork(after_in_child=_close_loop)


def get_or_reuse_loop():
    """
    Get a new eventloop or reuse the current opened eventloop.

    :return: A new eventloop or reuse the current opened eventloop.
    """
    _update_policy()
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError
    except RuntimeError:
        # no event loop
        # create a new loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


def typename(obj):
    """
    Get the typename of object.

    :param obj: Target object.
    :return: Typename of the obj.
    """
    if not isinstance(obj, type):
        obj = obj.__class__
    try:
        return f"{obj.__module__}.{obj.__name__}"
    except AttributeError:
        return str(obj)


class CatchAllCleanupContextManager:
    """
    This context manager guarantees, that the :method:``__exit__`` of the
    sub context is called, even when there is an Exception in the
    :method:``__enter__``.

    :param sub_context: The context, that should be taken care of.
    """

    def __init__(self, sub_context):
        self.sub_context = sub_context

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.sub_context.__exit__(exc_type, exc_val, exc_tb)


_ATTRIBUTES = {
    "bold": 1,
    "dark": 2,
    "underline": 4,
    "blink": 5,
    "reverse": 7,
    "concealed": 8,
}

_HIGHLIGHTS = {
    "on_grey": 40,
    "on_red": 41,
    "on_green": 42,
    "on_yellow": 43,
    "on_blue": 44,
    "on_magenta": 45,
    "on_cyan": 46,
    "on_white": 47,
}

_COLORS = {
    "black": 30,
    "red": 31,
    "green": 32,
    "yellow": 33,
    "blue": 34,
    "magenta": 35,
    "cyan": 36,
    "white": 37,
}

_RESET = "\033[0m"

if __windows__:
    os.system("color")


def colored(
    text: str,
    color: Optional[str] = None,
    on_color: Optional[str] = None,
    attrs: Optional[Union[str, list]] = None,
) -> str:
    """
    Give the text with color.

    :param text: The target text.
    :param color: The color of text. Chosen from the following.
        {
            'grey': 30,
            'red': 31,
            'green': 32,
            'yellow': 33,
            'blue': 34,
            'magenta': 35,
            'cyan': 36,
            'white': 37
        }
    :param on_color: The on_color of text. Chosen from the following.
        {
            'on_grey': 40,
            'on_red': 41,
            'on_green': 42,
            'on_yellow': 43,
            'on_blue': 44,
            'on_magenta': 45,
            'on_cyan': 46,
            'on_white': 47
        }
    :param attrs: Attributes of color. Chosen from the following.
        {
           'bold': 1,
           'dark': 2,
           'underline': 4,
           'blink': 5,
           'reverse': 7,
           'concealed': 8
        }
    :return: Colored text.
    """
    if "MARIE_LOG_NO_COLOR" not in os.environ:
        fmt_str = "\033[%dm%s"
        if color:
            text = fmt_str % (_COLORS[color], text)
        if on_color:
            text = fmt_str % (_HIGHLIGHTS[on_color], text)

        if attrs:
            if isinstance(attrs, str):
                attrs = [attrs]
            if isinstance(attrs, list):
                for attr in attrs:
                    text = fmt_str % (_ATTRIBUTES[attr], text)
        text += _RESET
    return text


def colored_rich(
    text: str,
    color: Optional[str] = None,
    on_color: Optional[str] = None,
    attrs: Optional[Union[str, list]] = None,
) -> str:
    """
    Give the text with color. You should only use it when printing with rich print. Othersiwe please see the colored
    function

    :param text: The target text
    :param color: The color of text
    :param on_color: The on color of text: ex on yellow
    :param attrs: Attributes of color

    :return: Colored text.
    """
    if "MARIE_LOG_NO_COLOR" not in os.environ:
        if color:
            text = _wrap_text_in_rich_bracket(text, color)
        if on_color:
            text = _wrap_text_in_rich_bracket(text, on_color)

        if attrs:
            if isinstance(attrs, str):
                attrs = [attrs]
            if isinstance(attrs, list):
                for attr in attrs:
                    text = _wrap_text_in_rich_bracket(text, attr)
    return text


def _wrap_text_in_rich_bracket(text: str, wrapper: str):
    return f"[{wrapper}]{text}[/{wrapper}]"


def get_rich_console():
    """
    Function to get jina rich default console.
    :return: rich console
    """
    return Console(
        force_terminal=True if "PYCHARM_HOSTED" in os.environ else None,
        color_system=None if "MARIE_LOG_NO_COLOR" in os.environ else "auto",
    )


def get_readable_time(*args, **kwargs):
    """
    Get the datetime in human readable format (e.g. 115 days and 17 hours and 46 minutes and 40 seconds).

    For example:
        .. highlight:: python
        .. code-block:: python
            get_readable_time(seconds=1000)

    :param args: arguments for datetime.timedelta
    :param kwargs: key word arguments for datetime.timedelta
    :return: Datetime in human readable format.
    """
    import datetime

    secs = float(datetime.timedelta(*args, **kwargs).total_seconds())
    units = [("day", 86400), ("hour", 3600), ("minute", 60), ("second", 1)]
    parts = []
    for unit, mul in units:
        if secs / mul >= 1 or mul == 1:
            if mul > 1:
                n = int(math.floor(secs / mul))
                secs -= n * mul
            else:
                n = int(secs)
            parts.append(f"{n} {unit}" + ("" if n == 1 else "s"))
    return " and ".join(parts)


def get_readable_size(num_bytes: Union[int, float]) -> str:
    """
    Transform the bytes into readable value with different units (e.g. 1 KB, 20 MB, 30.1 GB).

    :param num_bytes: Number of bytes.
    :return: Human readable string representation.
    """
    num_bytes = int(num_bytes)
    if num_bytes < 1024:
        return f"{num_bytes} Bytes"
    elif num_bytes < 1024**2:
        return f"{num_bytes / 1024:.1f} KB"
    elif num_bytes < 1024**3:
        return f"{num_bytes / (1024 ** 2):.1f} MB"
    else:
        return f"{num_bytes / (1024 ** 3):.1f} GB"
