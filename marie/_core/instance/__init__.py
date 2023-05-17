from __future__ import annotations

import logging
import logging.config
import os
import sys
import time
import weakref
from abc import abstractmethod
from collections import defaultdict
from enum import Enum
from tempfile import TemporaryDirectory
from types import TracebackType
from typing import (
    TYPE_CHECKING,
    AbstractSet,
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
    cast,
    TypeVar,
)

import yaml
from typing_extensions import Protocol, Self, TypeAlias, runtime_checkable

from marie import check
from marie._annotations import public
from marie._core.errors import DagsterInvariantViolationError


class InstanceType(Enum):
    PERSISTENT = "PERSISTENT"
    EPHEMERAL = "EPHEMERAL"


T_DagsterInstance = TypeVar(
    "T_DagsterInstance", bound="DagsterInstance", default="DagsterInstance"
)


class MayHaveInstanceWeakref(Generic[T_DagsterInstance]):
    """Mixin for classes that can have a weakref back to a Dagster instance."""

    _instance_weakref: Optional[weakref.ReferenceType[T_DagsterInstance]]

    def __init__(self):
        self._instance_weakref = None

    @property
    def has_instance(self) -> bool:
        return hasattr(self, "_instance_weakref") and (
            self._instance_weakref is not None
        )

    @property
    def _instance(self) -> T_DagsterInstance:
        instance = (
            self._instance_weakref()
            # Backcompat with custom subclasses that don't call super().__init__()
            # in their own __init__ implementations
            if (
                hasattr(self, "_instance_weakref")
                and self._instance_weakref is not None
            )
            else None
        )
        if instance is None:
            raise DagsterInvariantViolationError(
                "Attempted to resolve undefined DagsterInstance weakref."
            )
        else:
            return instance

    def register_instance(self, instance: T_DagsterInstance) -> None:
        check.invariant(
            # Backcompat with custom subclasses that don't call super().__init__()
            # in their own __init__ implementations
            (not hasattr(self, "_instance_weakref") or self._instance_weakref is None),
            "Must only call initialize once",
        )

        # Store a weakref to avoid a circular reference / enable GC
        self._instance_weakref = weakref.ref(instance)


class DagsterInstance:
    @public
    @staticmethod
    def get() -> "DagsterInstance":
        raise NotImplementedError()  # pragma: no cover
