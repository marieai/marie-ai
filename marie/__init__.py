"""
Top-level module of Marie.

The primary function of this module is to import all of the public Marie
interfaces into a single place. The interfaces themselves are located in
sub-modules, as described below.

"""

import os as _os
import platform as _platform
import signal as _signal
import sys as _sys
import warnings as _warnings

import docarray as _docarray

from marie.utils.types import strtobool

if _sys.version_info < (3, 10, 0):
    raise OSError(f"Marie requires Python >= 3.10, but yours is {_sys.version_info}")

if strtobool(_os.environ.get("MARIE_SUPPRESS_WARNINGS", "false")):
    # attempt to suppress all warnings from dependencies

    _warnings.simplefilter(action="ignore", category=FutureWarning)
    _warnings.simplefilter(action="ignore", category=UserWarning)
    _warnings.simplefilter(action="ignore", category=DeprecationWarning)

    # # Compiled functions can't take variable number of arguments or use keyword-only arguments with defaults
    try:

        def warn(*args, **kwargs):
            pass

        # _warnings.warn = warn

        # Work around for https://github.com/pytorch/pytorch/issues/29637
        # We will owerwrite the formatting function for warnings to make it not print anything
    except Exception as ex:
        pass
else:

    def _warning_on_one_line(message, category, filename, lineno, *args, **kwargs):
        return "\033[1;33m%s: %s\033[0m \033[1;30m(raised from %s:%s)\033[0m\n" % (
            category.__name__,
            message,
            filename,
            lineno,
        )

    def _ignore_google_warnings():
        import warnings

        warnings.filterwarnings(
            'ignore',
            category=DeprecationWarning,
            message='Deprecated call to `pkg_resources.declare_namespace(\'google\')`.',
            append=True,
        )

    _warnings.formatwarning = _warning_on_one_line
    _warnings.simplefilter('always', DeprecationWarning, append=True)
    _ignore_google_warnings()


# fix fork error on MacOS but seems no effect? must do EXPORT manually before jina start
_os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

# JINA_MP_START_METHOD has higher priority than os-patch
_start_method = _os.environ.get("JINA_MP_START_METHOD", None)
if _start_method and _start_method.lower() in {"fork", "spawn", "forkserver"}:
    from multiprocessing import set_start_method as _set_start_method

    try:
        _set_start_method(_start_method.lower())
        _warnings.warn(
            f"multiprocessing start method is set to `{_start_method.lower()}`"
        )
    except Exception as e:
        _warnings.warn(
            f"failed to set multiprocessing start_method to `{_start_method.lower()}`: {e!r}"
        )
elif _sys.version_info >= (3, 8, 0) and _platform.system() == "Darwin":
    # DO SOME OS-WISE PATCHES

    # temporary fix for python 3.8 on macos where the default start is set to "spawn"
    # https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
    from multiprocessing import set_start_method as _set_start_method

    try:
        _set_start_method('fork')
        _warnings.warn(f'multiprocessing start method is set to `fork`')
    except Exception as e:
        _warnings.warn(f'failed to set multiprocessing start_method to `fork`: {e!r}')

# do not change this line manually
# this is managed by git tag and updated on every release
# NOTE: this represents the NEXT release version
__version__ = '3.0.30'

# do not change this line manually
# this is managed by proto/build-proto.sh and updated on every execution
__proto_version__ = '0.1.28'

try:
    __docarray_version__ = _docarray.__version__
except AttributeError as e:
    raise RuntimeError(
        "`docarray` dependency is not installed correctly, please reinstall with `pip install -U --force-reinstall docarray`"
    )

try:
    _signal.signal(_signal.SIGINT, _signal.default_int_handler)
except Exception as exc:
    _warnings.warn(f"failed to set default signal handler: {exc!r}`")


def _set_nofile(nofile_atleast=4096):
    """
    Set nofile soft limit to at least 4096, useful for running matlplotlib/seaborn on
    parallel executing plot generators vs. Ubuntu default ulimit -n 1024 or OS X El Captian 256
    temporary setting extinguishing with Python session.

    :param nofile_atleast: nofile soft limit
    :return: nofile soft limit and nofile hard limit
    """

    try:
        import resource as res
    except ImportError:  # Windows
        res = None

    if res is None:
        return (None,) * 2

    soft, ohard = res.getrlimit(res.RLIMIT_NOFILE)
    hard = ohard

    if soft < nofile_atleast:
        soft = nofile_atleast
        if hard < soft:
            hard = soft

        try:
            res.setrlimit(res.RLIMIT_NOFILE, (soft, hard))
        except (ValueError, res.error):
            try:
                hard = soft
                print(f"trouble with max limit, retrying with soft,hard {soft},{hard}")
                res.setrlimit(res.RLIMIT_NOFILE, (soft, hard))
            except Exception:
                print("failed to set ulimit, giving up")
                soft, hard = res.getrlimit(res.RLIMIT_NOFILE)

    return soft, hard


_set_nofile()

# ONLY FIRST CLASS CITIZENS ARE ALLOWED HERE, namely Document, Executor Flow

# Document
from marie._docarray import Document, DocumentArray

# Client
from marie.clients import Client
from marie.orchestrate.deployments import Deployment
from marie.orchestrate.flow.asyncio import AsyncFlow

# Flow
from marie.orchestrate.flow.base import Flow

# Executor
from marie.serve.executors import BaseExecutor as Executor
from marie.serve.executors.decorators import (
    dynamic_batching,
    monitor,
    requests,
    safely_encoded,
)

# Custom Gateway
from marie.serve.runtimes.gateway.gateway import Gateway
