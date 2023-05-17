import hashlib
from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    AbstractSet,
    Iterator,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Union,
    cast,
)

from typing_extensions import TypeAlias

import marie.check as check
from marie._core.definitions import JobDefinition, NodeHandle
from marie._core.errors import (
    DagsterExecutionLoadInputError,
    DagsterInvariantViolationError,
    DagsterTypeLoadingError,
    user_code_error_boundary,
)

from .objects import TypeCheckData
from .outputs import StepOutputHandle, UnresolvedStepOutputHandle
from ..context.system import StepExecutionContext
from ...definitions.job_definition import InputDefinition
from ...system_config.objects import ResolvedRunConfig


class UnresolvedCollectStepInput:
    pass


class UnresolvedMappedStepInput:
    pass


StepInputUnion: TypeAlias = Union[
    "StepInput", "UnresolvedMappedStepInput", "UnresolvedCollectStepInput"
]


class StepInputData(
    NamedTuple(
        "_StepInputData", [("input_name", str), ("type_check_data", TypeCheckData)]
    )
):
    """Serializable payload of information for the result of processing a step input."""

    def __new__(cls, input_name: str, type_check_data: TypeCheckData):
        return super(StepInputData, cls).__new__(
            cls,
            input_name=check.str_param(input_name, "input_name"),
            type_check_data=check.inst_param(
                type_check_data, "type_check_data", TypeCheckData
            ),
        )


class StepInput(
    NamedTuple(
        "_StepInput",
        [("name", str), ("dagster_type_key", str), ("source", "StepInputSource")],
    )
):
    """Holds information for how to prepare an input for an ExecutionStep."""

    def __new__(cls, name: str, dagster_type_key: str, source: "StepInputSource"):
        return super(StepInput, cls).__new__(
            cls,
            name=check.str_param(name, "name"),
            dagster_type_key=check.str_param(dagster_type_key, "dagster_type_key"),
            source=check.inst_param(source, "source", StepInputSource),
        )

    @property
    def dependency_keys(self) -> AbstractSet[str]:
        return self.source.step_key_dependencies

    def get_step_output_handle_dependencies(self) -> Sequence[StepOutputHandle]:
        return self.source.step_output_handle_dependencies


def join_and_hash(*args: Optional[str]) -> Optional[str]:
    lst = [check.opt_str_param(elem, "elem") for elem in args]
    if None in lst:
        return None

    str_lst = cast(List[str], lst)
    unhashed = "".join(sorted(str_lst))
    return hashlib.sha1(unhashed.encode("utf-8")).hexdigest()


class StepInputSource(ABC):
    """How to load the data for a step input."""

    @property
    def step_key_dependencies(self) -> Set[str]:
        return set()

    @property
    def step_output_handle_dependencies(self) -> Sequence[StepOutputHandle]:
        return []

    @abstractmethod
    def load_input_object(
        self, step_context: "StepExecutionContext", input_def: InputDefinition
    ) -> Iterator[object]:
        ...

    def required_resource_keys(self, _job_def: JobDefinition) -> AbstractSet[str]:
        return set()

    @abstractmethod
    def compute_version(
        self,
        step_versions: Mapping[str, Optional[str]],
        job_def: JobDefinition,
        resolved_run_config: ResolvedRunConfig,
    ) -> Optional[str]:
        """See resolve_step_versions in resolve_versions.py for explanation of step_versions."""
        raise NotImplementedError()
