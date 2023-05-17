import asyncio
import os
import re
import time
import warnings
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from signal import Signals
from typing import (
    TYPE_CHECKING,
    AbstractSet,
    Any,
    Callable,
    Dict,
    Iterator,
    Mapping,
    NamedTuple,
    NoReturn,
    Optional,
    Sequence,
    TypeVar,
    Union,
    cast,
)

import pendulum
from typing_extensions import Self

from .events import DagsterEvent

# test utils from separate light weight file since are exported top level
from .instance_for_test import (
    cleanup_test_instance as cleanup_test_instance,
    environ as environ,
    instance_for_test as instance_for_test,
)

if TYPE_CHECKING:
    from pendulum.datetime import DateTime

T = TypeVar("T")
T_NamedTuple = TypeVar("T_NamedTuple", bound=NamedTuple)


def assert_namedtuple_lists_equal(
    t1_list: Sequence[T_NamedTuple],
    t2_list: Sequence[T_NamedTuple],
    exclude_fields: Optional[Sequence[str]] = None,
) -> None:
    for t1, t2 in zip(t1_list, t2_list):
        assert_namedtuples_equal(t1, t2, exclude_fields)


def assert_namedtuples_equal(
    t1: T_NamedTuple, t2: T_NamedTuple, exclude_fields: Optional[Sequence[str]] = None
) -> None:
    exclude_fields = exclude_fields or []
    for field in type(t1)._fields:
        if field not in exclude_fields:
            assert getattr(t1, field) == getattr(t2, field)


def step_output_event_filter(pipe_iterator: Iterator[DagsterEvent]):
    for step_event in pipe_iterator:
        if step_event.is_successful_output:
            yield step_event
