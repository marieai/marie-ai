from __future__ import annotations

from types import TracebackType
from typing import Tuple, Type, Union

ExcInfo = Union[
    Tuple[Type[BaseException], BaseException, TracebackType], Tuple[None, None, None]
]


def strtobool(val: Union[bool, str]) -> bool:
    """
    Convert a bool string or bool to a bool type.
    """
    if isinstance(val, bool):
        return val

    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return True
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return False
    else:
        raise ValueError(f"Invalid truth value: {val}")


def to_bool(val, default: bool = False) -> bool:
    """Convert a value to bool, returning *default* for ``None`` or empty strings."""
    if isinstance(val, bool):
        return val
    if val is None:
        return default
    if isinstance(val, str):
        val = val.strip().lower()
        if not val:
            return default
        return val in ("1", "true", "yes", "on")
    return bool(val)
