import datetime
import sys
import threading
from abc import abstractmethod
from contextlib import AbstractContextManager
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)


class CodeLocation(AbstractContextManager):
    """A CodeLocation represents a target containing user code which has a set of Dagster
    definition objects. A given location will contain some number of uniquely named
    RepositoryDefinitions, which therein contains job, op, and other definitions.

    Dagster tools are typically "host" processes, meaning they load a CodeLocation and
    communicate with it over an IPC/RPC layer. Currently this IPC layer is implemented by
    invoking the dagster CLI in a target python interpreter (e.g. a virtual environment) in either
      a) the current node
      b) a container

    In the near future, we may also make this communication channel able over an RPC layer, in
    which case the information needed to load a CodeLocation will be a url that abides by
    some RPC contract.

    We also allow for InProcessCodeLocation which actually loads the user-defined artifacts
    into process with the host tool. This is mostly for test scenarios.
    """

    pass
