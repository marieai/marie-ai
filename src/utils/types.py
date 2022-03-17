from __future__ import annotations

import distutils


def strtobool(val: [bool | str]) -> bool:
    """
    Convert bool string or bool to a bool type
    """
    if isinstance(val, bool):
        return val
    return distutils.util.strtobool(val)
