import datetime as _datetime
import os as _os
import platform as _platform
import signal as _signal
import sys as _sys
import warnings as _warnings


if _sys.version_info < (3, 7, 0):
    raise OSError(f"Marie requires Python >= 3.7, but yours is {_sys.version_info}")

__windows__ = _sys.platform == "win32"
__args_executor_init__ = {"metas", "requests", "runtime_args"}
__root_dir__ = _os.path.dirname(_os.path.abspath(__file__))
__resources_path__ = _os.path.join(_os.path.dirname(_sys.modules["marie"].__file__), "resources")

__default_host__ = _os.environ.get("MARIE_DEFAULT_HOST", "127.0.0.1" if __windows__ else "0.0.0.0")
__default_endpoint__ = "/default"
__args_executor_func__ = {
    "docs",
    "parameters",
    "docs_matrix",
}
__args_executor_init__ = {"metas", "requests", "runtime_args"}
__uptime__ = _datetime.datetime.now().isoformat()

# ONLY FIRST CLASS CITIZENS ARE ALLOWED HERE, namely Document, Executor Flow

# Executor
from marie.serve.executors import BaseExecutor as Executor
from marie.serve.executors.decorators import monitor, requests

from marie.docarray.document_array import DocumentArray as DocumentArray
from marie.docarray.document import Document as Document
