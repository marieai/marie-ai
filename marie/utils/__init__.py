import datetime
import errno
import os
from enum import Enum
from typing import Mapping, TypeVar, Tuple, TypeAlias, Callable, Any

from .utils import FileSystem
from .. import check

K = TypeVar("K")
T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")

EPOCH = datetime.datetime.utcfromtimestamp(0)

PICKLE_PROTOCOL = 4

PrintFn: TypeAlias = Callable[[Any], None]


def mkdir_p(path: str) -> str:
    try:
        os.makedirs(path)
        return path
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            return path
        else:
            raise


def datetime_as_float(dt: datetime.datetime) -> float:
    check.inst_param(dt, "dt", datetime.datetime)
    return float((dt - EPOCH).total_seconds())


def ensure_single_item(ddict: Mapping[T, U]) -> Tuple[T, U]:
    check.mapping_param(ddict, "ddict")
    check.param_invariant(len(ddict) == 1, "ddict", "Expected dict with single item")
    return list(ddict.items())[0]


def is_enum_value(value: object) -> bool:
    return False if value is None else issubclass(value.__class__, Enum)
