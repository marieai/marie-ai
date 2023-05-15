"""Serialization & deserialization for Dagster objects.

Why have custom serialization?

* Default json serialization doesn't work well on namedtuples, which we use extensively to create
  immutable value types. Namedtuples serialize like tuples as flat lists.
* Explicit whitelisting should help ensure we are only persisting or communicating across a
  serialization boundary the types we expect to.

Why not pickle?

* This isn't meant to replace pickle in the conditions that pickle is reasonable to use
  (in memory, not human readable, etc) just handle the json case effectively.
"""
import collections.abc
import warnings
from abc import ABC, abstractmethod
from enum import Enum
from functools import partial
from inspect import Parameter, signature
from typing import (
    AbstractSet,
    Any,
    Callable,
    Dict,
    FrozenSet,
    Generic,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
    cast,
    overload,
    TypeVar,
)

from typing_extensions import Final, Self, TypeAlias


T = TypeVar("T")
U = TypeVar("U")
T_Type = TypeVar("T_Type", bound=Type[object])
T_Scalar = TypeVar("T_Scalar", bound=Union[str, int, float, bool, None])


@overload
def whitelist_for_serdes(__cls: T_Type) -> T_Type:
    ...


###################################################################################################
# Types
###################################################################################################
JsonSerializableValue: TypeAlias = Union[
    Sequence["JsonSerializableValue"],
    Mapping[str, "JsonSerializableValue"],
    str,
    int,
    float,
    bool,
    None,
]

PackableValue: TypeAlias = Union[
    Sequence["PackableValue"],
    Mapping[str, "PackableValue"],
    str,
    int,
    float,
    bool,
    None,
    NamedTuple,
    Set["PackableValue"],
    FrozenSet["PackableValue"],
    Enum,
]


class WhitelistMap(NamedTuple):
    pass


def deserialize_value(val: str) -> Any:
    from marie.utils.json import deserialize_value

    """Deserialize a JSON string to an object.
    """
    return deserialize_value(val)


def pack_value(val: PackableValue) -> JsonSerializableValue:
    return val


def serialize_value(
    val: PackableValue,
    **json_kwargs: object,
) -> str:
    from marie.utils.json import to_json

    """Serialize an object to a JSON string.

    Objects are first converted to a JSON-serializable form with `pack_value`.
    """
    packed_value = pack_value(val)
    return to_json(packed_value, **json_kwargs)


def whitelist_for_serdes():
    return
