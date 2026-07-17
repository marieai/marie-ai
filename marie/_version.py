"""Version metadata for the marie-ai distribution."""

import docarray

__version__ = "5.0.0"
__proto_version__ = "0.1.28"

try:
    __docarray_version__ = docarray.__version__
except AttributeError as exc:
    raise RuntimeError(
        "The docarray dependency is not installed correctly; reinstall it with "
        "`uv sync --reinstall-package docarray`."
    ) from exc
