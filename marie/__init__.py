import datetime as _datetime
import os as _os
import platform as _platform
import re
import signal as _signal
import sys as _sys
import warnings as _warnings
import docarray as _docarray


if _sys.version_info < (3, 7, 0):
    raise OSError(f"Marie requires Python >= 3.7, but yours is {_sys.version_info}")

__windows__ = _sys.platform == "win32"
__args_executor_init__ = {"metas", "requests", "runtime_args"}
__root_dir__ = _os.path.dirname(_os.path.abspath(__file__))
# __resources_path__ = _os.path.join(_os.path.dirname(_sys.modules["marie"].__file__), "resources")
__resources_path__ = _os.path.join(_os.path.abspath(_os.path.join(__root_dir__, "..")), "resources")

__default_host__ = _os.environ.get("MARIE_DEFAULT_HOST", "127.0.0.1" if __windows__ else "0.0.0.0")
__default_port_monitoring__ = 9090
__docker_host__ = 'host.docker.internal'
__default_executor__ = "BaseExecutor"
__default_endpoint__ = "/default"
__ready_msg__ = 'ready and listening'
__stop_msg__ = 'terminated'
__unset_msg__ = "(unset)"

__args_executor_func__ = {
    "docs",
    "parameters",
    "docs_matrix",
}
__args_executor_init__ = {"metas", "requests", "runtime_args"}

try:
    __docarray_version__ = _docarray.__version__
except AttributeError as e:
    raise OSError(
        "`docarray` dependency is not installed correctly, please reinstall with `pip install -U --force-reinstall docarray`"
    )

__uptime__ = _datetime.datetime.now().isoformat()


# 1. clean this tuple,
# 2. grep -rohEI --exclude-dir=jina/hub --exclude-dir=tests --include \*.py "\'MARIE_.*?\'" jina  | sort -u | sed "s/$/,/g"
# 3. copy all lines EXCEPT the first (which is the grep command in the last line)
__marie_env__ = (
    "MARIE_DEFAULT_HOST",
    "MARIE_DEFAULT_TIMEOUT_CTRL",
    "MARIE_DEFAULT_WORKSPACE_BASE",
    "MARIE_DEPLOYMENT_NAME",
    "MARIE_DISABLE_UVLOOP",
    "MARIE_CHECK_VERSION",
)


new_env_regex_str = r"\${{\sENV\.[a-zA-Z0-9_]*\s}}|\${{\senv\.[a-zA-Z0-9_]*\s}}"
new_env_var_regex = re.compile(new_env_regex_str)  # matches expressions of form '${{ ENV.var }}' or '${{ env.var }}'

env_var_deprecated_regex_str = r"\$[a-zA-Z0-9_]*"
env_var_deprecated_regex = re.compile(r"\$[a-zA-Z0-9_]*")  # matches expressions of form '$var'

env_var_regex_str = env_var_deprecated_regex_str + "|" + new_env_regex_str
env_var_regex = re.compile(env_var_regex_str)  # matches either of the above


# ONLY FIRST CLASS CITIZENS ARE ALLOWED HERE, namely Document, Executor Flow

# Executor
from marie.serve.executors import BaseExecutor as Executor
from marie.serve.executors.decorators import monitor, requests

from docarray import DocumentArray as DocumentArray
from docarray.document import Document as Document
