from typing import Dict, NamedTuple, Sequence, Union, FrozenSet, Optional

from marie import check


class ExecutionPlan(
    NamedTuple(
        "_ExecutionPlan",
        [
            ("step_dict", Dict[StepHandleUnion, IExecutionStep]),
            ("executable_map", Dict[str, Union[StepHandle, ResolvedFromDynamicStepHandle]]),
            (
                "resolvable_map",
                Dict[FrozenSet[str], Sequence[Union[StepHandle, UnresolvedStepHandle]]],
            ),
            ("step_handles_to_execute", Sequence[StepHandleUnion]),
            ("known_state", KnownExecutionState),
            ("artifacts_persisted", bool),
            ("step_dict_by_key", Dict[str, IExecutionStep]),
            ("executor_name", Optional[str]),
            ("repository_load_data", Optional[RepositoryLoadData]),
        ],
    )
):
    def __new__(
        cls,
        step_dict: Dict[StepHandleUnion, IExecutionStep],
        executable_map: Dict[str, Union[StepHandle, ResolvedFromDynamicStepHandle]],
        resolvable_map: Dict[FrozenSet[str], Sequence[Union[StepHandle, UnresolvedStepHandle]]],
        step_handles_to_execute: Sequence[StepHandleUnion],
        known_state: KnownExecutionState,
        artifacts_persisted: bool = False,
        step_dict_by_key: Optional[Dict[str, IExecutionStep]] = None,
        executor_name: Optional[str] = None,
        repository_load_data: Optional[RepositoryLoadData] = None,
    ):
        return super(ExecutionPlan, cls).__new__(
            cls,
            step_dict=check.dict_param(
                step_dict,
                "step_dict",
                key_type=StepHandleTypes,
                value_type=(
                    ExecutionStep,
                    UnresolvedMappedExecutionStep,
                    UnresolvedCollectExecutionStep,
                ),
            ),
            executable_map=executable_map,
            resolvable_map=resolvable_map,
            step_handles_to_execute=check.sequence_param(
                step_handles_to_execute,
                "step_handles_to_execute",
                of_type=(StepHandle, UnresolvedStepHandle, ResolvedFromDynamicStepHandle),
            ),
            known_state=check.inst_param(known_state, "known_state", KnownExecutionState),
            artifacts_persisted=check.bool_param(artifacts_persisted, "artifacts_persisted"),
            step_dict_by_key={step.key: step for step in step_dict.values()}
            if step_dict_by_key is None
            else check.dict_param(
                step_dict_by_key,
                "step_dict_by_key",
                key_type=str,
                value_type=(
                    ExecutionStep,
                    UnresolvedMappedExecutionStep,
                    UnresolvedCollectExecutionStep,
                ),
            ),
            executor_name=check.opt_str_param(executor_name, "executor_name"),
            repository_load_data=check.opt_inst_param(
                repository_load_data, "repository_load_data", RepositoryLoadData
            ),
        )

    @property
    def steps(self) -> Sequence[IExecutionStep]:
        return list(self.step_dict.values())
