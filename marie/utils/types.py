from __future__ import annotations

from distutils import util


def strtobool(val: [bool | str]) -> bool:
    """
    Convert bool string or bool to a bool type
    """
    if isinstance(val, bool):
        return val

    return util.strtobool(val)
