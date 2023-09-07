from __future__ import annotations

from distutils import util
from types import TracebackType
from typing import Tuple, Type, Union

ExcInfo = Union[
    Tuple[Type[BaseException], BaseException, TracebackType], Tuple[None, None, None]
]


def strtobool(val: [bool | str]) -> bool:
    """
    Convert bool string or bool to a bool type
    """
    if isinstance(val, bool):
        return val

    return bool(util.strtobool(val))
