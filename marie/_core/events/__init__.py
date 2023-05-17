"""Structured representations of system events."""
import logging

import os
from enum import Enum
from typing import (
    Sequence,
    Union,
    Optional,
    NamedTuple,
    Mapping,
    cast,
    AbstractSet,
    Any,
)

from marie import check
from marie._annotations import public
from marie._core.definitions.dependency import NodeHandle
from marie._core.definitions import (
    AssetKey,
    ExpectationResult,
    AssetMaterialization,
    AssetObservation,
    HookDefinition,
)
from marie._core.definitions.events import (
    AssetLineageInfo,
    ObjectStoreOperationType,
    ObjectStoreOperation,
)
from marie._core.definitions.metadata import (
    MetadataValue,
    normalize_metadata,
    RawMetadataValue,
)
from marie._core.errors import HookExecutionError
from marie._core.execution.context.system import (
    IPlanContext,
    IStepContext,
    StepExecutionContext,
)

from marie._core.execution.plan.handle import StepHandle, ResolvedFromDynamicStepHandle
from marie._core.execution.plan.inputs import StepInputData
from marie._core.execution.plan.objects import (
    StepSuccessData,
    StepFailureData,
    StepRetryData,
)
from marie._core.execution.plan.outputs import StepOutputData
from marie._core.execution.plan.plan import ExecutionPlan
from marie._core.execution.plan.step import StepKind
from marie._core.log_manager import DagsterLogManager
from marie._core.storage.marie_run import DagsterRunStatus
from marie.utils.error import (
    SerializableErrorInfo,
    serializable_error_info_from_exc_info,
)
from marie.utils.timing import format_duration


class MarieEventType(str, Enum):
    """The types of events that may be yielded by op and job execution."""

    STEP_OUTPUT = "STEP_OUTPUT"
    STEP_INPUT = "STEP_INPUT"
    STEP_FAILURE = "STEP_FAILURE"
    STEP_START = "STEP_START"
    STEP_SUCCESS = "STEP_SUCCESS"
    STEP_SKIPPED = "STEP_SKIPPED"

    # The process carrying out step execution is starting/started. Shown as a
    # marker start/end
    STEP_WORKER_STARTING = "STEP_WORKER_STARTING"
    STEP_WORKER_STARTED = "STEP_WORKER_STARTED"

    # Resource initialization for execution has started/succeede/failed. Shown
    # as a marker start/end
    RESOURCE_INIT_STARTED = "RESOURCE_INIT_STARTED"
    RESOURCE_INIT_SUCCESS = "RESOURCE_INIT_SUCCESS"
    RESOURCE_INIT_FAILURE = "RESOURCE_INIT_FAILURE"

    STEP_UP_FOR_RETRY = "STEP_UP_FOR_RETRY"  # "failed" but want to retry
    STEP_RESTARTED = "STEP_RESTARTED"

    ASSET_MATERIALIZATION = "ASSET_MATERIALIZATION"
    ASSET_MATERIALIZATION_PLANNED = "ASSET_MATERIALIZATION_PLANNED"
    ASSET_OBSERVATION = "ASSET_OBSERVATION"
    STEP_EXPECTATION_RESULT = "STEP_EXPECTATION_RESULT"

    # We want to display RUN_* events in dagit and in our LogManager output, but in order to
    # support backcompat for our storage layer, we need to keep the persisted value to be strings
    # of the form "PIPELINE_*".  We may have user code that pass in the DagsterEventType
    # enum values into storage APIs (like get_event_records, which takes in an EventRecordsFilter).
    RUN_ENQUEUED = "PIPELINE_ENQUEUED"
    RUN_DEQUEUED = "PIPELINE_DEQUEUED"
    RUN_STARTING = (
        "PIPELINE_STARTING"  # Launch is happening, execution hasn't started yet
    )
    RUN_START = "PIPELINE_START"  # Execution has started
    RUN_SUCCESS = "PIPELINE_SUCCESS"
    RUN_FAILURE = "PIPELINE_FAILURE"
    RUN_CANCELING = "PIPELINE_CANCELING"
    RUN_CANCELED = "PIPELINE_CANCELED"

    # Keep these legacy enum values around, to keep back-compatability for user code that might be
    # using these constants to filter event records
    PIPELINE_ENQUEUED = RUN_ENQUEUED
    PIPELINE_DEQUEUED = RUN_DEQUEUED
    PIPELINE_STARTING = RUN_STARTING
    PIPELINE_START = RUN_START
    PIPELINE_SUCCESS = RUN_SUCCESS
    PIPELINE_FAILURE = RUN_FAILURE
    PIPELINE_CANCELING = RUN_CANCELING
    PIPELINE_CANCELED = RUN_CANCELED

    OBJECT_STORE_OPERATION = "OBJECT_STORE_OPERATION"
    ASSET_STORE_OPERATION = "ASSET_STORE_OPERATION"
    LOADED_INPUT = "LOADED_INPUT"
    HANDLED_OUTPUT = "HANDLED_OUTPUT"

    ENGINE_EVENT = "ENGINE_EVENT"

    HOOK_COMPLETED = "HOOK_COMPLETED"
    HOOK_ERRORED = "HOOK_ERRORED"
    HOOK_SKIPPED = "HOOK_SKIPPED"

    ALERT_START = "ALERT_START"
    ALERT_SUCCESS = "ALERT_SUCCESS"
    ALERT_FAILURE = "ALERT_FAILURE"

    LOGS_CAPTURED = "LOGS_CAPTURED"


STEP_EVENTS = {
    MarieEventType.STEP_INPUT,
    MarieEventType.STEP_START,
    MarieEventType.STEP_OUTPUT,
    MarieEventType.STEP_FAILURE,
    MarieEventType.STEP_SUCCESS,
    MarieEventType.STEP_SKIPPED,
    MarieEventType.ASSET_MATERIALIZATION,
    MarieEventType.ASSET_OBSERVATION,
    MarieEventType.STEP_EXPECTATION_RESULT,
    MarieEventType.OBJECT_STORE_OPERATION,
    MarieEventType.HANDLED_OUTPUT,
    MarieEventType.LOADED_INPUT,
    MarieEventType.STEP_RESTARTED,
    MarieEventType.STEP_UP_FOR_RETRY,
}

FAILURE_EVENTS = {
    MarieEventType.RUN_FAILURE,
    MarieEventType.STEP_FAILURE,
    MarieEventType.RUN_CANCELED,
}

PIPELINE_EVENTS = {
    MarieEventType.RUN_ENQUEUED,
    MarieEventType.RUN_DEQUEUED,
    MarieEventType.RUN_STARTING,
    MarieEventType.RUN_START,
    MarieEventType.RUN_SUCCESS,
    MarieEventType.RUN_FAILURE,
    MarieEventType.RUN_CANCELING,
    MarieEventType.RUN_CANCELED,
}

HOOK_EVENTS = {
    MarieEventType.HOOK_COMPLETED,
    MarieEventType.HOOK_ERRORED,
    MarieEventType.HOOK_SKIPPED,
}

ALERT_EVENTS = {
    MarieEventType.ALERT_START,
    MarieEventType.ALERT_SUCCESS,
    MarieEventType.ALERT_FAILURE,
}

MARKER_EVENTS = {
    MarieEventType.ENGINE_EVENT,
    MarieEventType.STEP_WORKER_STARTING,
    MarieEventType.STEP_WORKER_STARTED,
    MarieEventType.RESOURCE_INIT_STARTED,
    MarieEventType.RESOURCE_INIT_SUCCESS,
    MarieEventType.RESOURCE_INIT_FAILURE,
}

EVENT_TYPE_TO_PIPELINE_RUN_STATUS = {
    MarieEventType.RUN_START: DagsterRunStatus.STARTED,
    MarieEventType.RUN_SUCCESS: DagsterRunStatus.SUCCESS,
    MarieEventType.RUN_FAILURE: DagsterRunStatus.FAILURE,
    MarieEventType.RUN_ENQUEUED: DagsterRunStatus.QUEUED,
    MarieEventType.RUN_STARTING: DagsterRunStatus.STARTING,
    MarieEventType.RUN_CANCELING: DagsterRunStatus.CANCELING,
    MarieEventType.RUN_CANCELED: DagsterRunStatus.CANCELED,
}

PIPELINE_RUN_STATUS_TO_EVENT_TYPE = {
    v: k for k, v in EVENT_TYPE_TO_PIPELINE_RUN_STATUS.items()
}

ASSET_EVENTS = {
    MarieEventType.ASSET_MATERIALIZATION,
    MarieEventType.ASSET_OBSERVATION,
    MarieEventType.ASSET_MATERIALIZATION_PLANNED,
}

DagsterEventType = MarieEventType


def _assert_type(
    method: str,
    expected_type: Union[MarieEventType, Sequence[MarieEventType]],
    actual_type: MarieEventType,
) -> None:
    _expected_type = (
        [expected_type] if isinstance(expected_type, MarieEventType) else expected_type
    )
    check.invariant(
        actual_type in _expected_type,
        (
            f"{method} only callable when event_type is"
            f" {','.join([t.value for t in _expected_type])}, called on {actual_type}"
        ),
    )


def _validate_event_specific_data(
    event_type: MarieEventType, event_specific_data: Optional["EventSpecificData"]
) -> Optional["EventSpecificData"]:
    return event_specific_data


def log_step_event(step_context: IStepContext, event: "DagsterEvent") -> None:
    event_type = DagsterEventType(event.event_type_value)
    log_level = logging.ERROR if event_type in FAILURE_EVENTS else logging.DEBUG

    step_context.log.log_dagster_event(
        level=log_level,
        msg=event.message or f"{event_type} for step {step_context.step.key}",
        dagster_event=event,
    )


def log_job_event(job_context: IPlanContext, event: "DagsterEvent") -> None:
    event_type = DagsterEventType(event.event_type_value)
    log_level = logging.ERROR if event_type in FAILURE_EVENTS else logging.DEBUG

    job_context.log.log_dagster_event(
        level=log_level,
        msg=event.message or f"{event_type} for pipeline {job_context.job_name}",
        dagster_event=event,
    )


def log_resource_event(log_manager: DagsterLogManager, event: "DagsterEvent") -> None:
    event_specific_data = cast(EngineEventData, event.event_specific_data)

    log_level = logging.ERROR if event_specific_data.error else logging.DEBUG
    log_manager.log_dagster_event(
        level=log_level, msg=event.message or "", dagster_event=event
    )


class EventSpecificData:
    pass


class CapturedLogContext:
    pass


class DagsterEvent(
    NamedTuple(
        "_DagsterEvent",
        [
            ("event_type_value", str),
            ("job_name", str),
            ("step_handle", Optional[Union[StepHandle, ResolvedFromDynamicStepHandle]]),
            ("node_handle", Optional[NodeHandle]),
            ("step_kind_value", Optional[str]),
            ("logging_tags", Optional[Mapping[str, str]]),
            ("event_specific_data", Optional["EventSpecificData"]),
            ("message", Optional[str]),
            ("pid", Optional[int]),
            ("step_key", Optional[str]),
        ],
    )
):
    """Events yielded by op and job execution.

    Users should not instantiate this class.

    Attributes:
        event_type_value (str): Value for a DagsterEventType.
        job_name (str)
        node_handle (NodeHandle)
        step_kind_value (str): Value for a StepKind.
        logging_tags (Dict[str, str])
        event_specific_data (Any): Type must correspond to event_type_value.
        message (str)
        pid (int)
        step_key (Optional[str]): DEPRECATED
    """

    @staticmethod
    def from_step(
        event_type: "DagsterEventType",
        step_context: IStepContext,
        event_specific_data: Optional["EventSpecificData"] = None,
        message: Optional[str] = None,
    ) -> "DagsterEvent":
        event = DagsterEvent(
            event_type_value=check.inst_param(
                event_type, "event_type", DagsterEventType
            ).value,
            job_name=step_context.job_name,
            step_handle=step_context.step.handle,
            node_handle=step_context.step.node_handle,
            step_kind_value=step_context.step.kind.value,
            logging_tags=step_context.event_tags,
            event_specific_data=_validate_event_specific_data(
                event_type, event_specific_data
            ),
            message=check.opt_str_param(message, "message"),
            pid=os.getpid(),
        )

        log_step_event(step_context, event)

        return event

    @staticmethod
    def from_job(
        event_type: DagsterEventType,
        job_context: IPlanContext,
        message: Optional[str] = None,
        event_specific_data: Optional["EventSpecificData"] = None,
        step_handle: Optional[Union[StepHandle, ResolvedFromDynamicStepHandle]] = None,
    ) -> "DagsterEvent":
        check.opt_inst_param(
            step_handle, "step_handle", (StepHandle, ResolvedFromDynamicStepHandle)
        )

        event = DagsterEvent(
            event_type_value=check.inst_param(
                event_type, "event_type", DagsterEventType
            ).value,
            job_name=job_context.job_name,
            message=check.opt_str_param(message, "message"),
            event_specific_data=_validate_event_specific_data(
                event_type, event_specific_data
            ),
            step_handle=step_handle,
            pid=os.getpid(),
        )

        log_job_event(job_context, event)

        return event

    @staticmethod
    def from_resource(
        event_type: DagsterEventType,
        job_name: str,
        execution_plan: "ExecutionPlan",
        log_manager: DagsterLogManager,
        message: Optional[str] = None,
        event_specific_data: Optional["EngineEventData"] = None,
    ) -> "DagsterEvent":
        event = DagsterEvent(
            event_type_value=check.inst_param(
                event_type, "event_type", DagsterEventType
            ).value,
            job_name=job_name,
            message=check.opt_str_param(message, "message"),
            event_specific_data=_validate_event_specific_data(
                DagsterEventType.ENGINE_EVENT, event_specific_data
            ),
            step_handle=execution_plan.step_handle_for_single_step_plans(),
            pid=os.getpid(),
        )
        log_resource_event(log_manager, event)
        return event

    def __new__(
        cls,
        event_type_value: str,
        job_name: str,
        step_handle: Optional[Union[StepHandle, ResolvedFromDynamicStepHandle]] = None,
        node_handle: Optional[NodeHandle] = None,
        step_kind_value: Optional[str] = None,
        logging_tags: Optional[Mapping[str, str]] = None,
        event_specific_data: Optional["EventSpecificData"] = None,
        message: Optional[str] = None,
        pid: Optional[int] = None,
        # legacy
        step_key: Optional[str] = None,
    ):
        # old events may contain node_handle but not step_handle
        if node_handle is not None and step_handle is None:
            step_handle = StepHandle(node_handle)

        # Legacy events may have step_key set directly, preserve those to stay in sync
        # with legacy execution plan snapshots.
        if step_handle is not None and step_key is None:
            step_key = step_handle.to_key()

        return super(DagsterEvent, cls).__new__(
            cls,
            check.str_param(event_type_value, "event_type_value"),
            check.str_param(job_name, "job_name"),
            check.opt_inst_param(
                step_handle, "step_handle", (StepHandle, ResolvedFromDynamicStepHandle)
            ),
            check.opt_inst_param(node_handle, "node_handle", NodeHandle),
            check.opt_str_param(step_kind_value, "step_kind_value"),
            check.opt_mapping_param(logging_tags, "logging_tags"),
            _validate_event_specific_data(
                DagsterEventType(event_type_value), event_specific_data
            ),
            check.opt_str_param(message, "message"),
            check.opt_int_param(pid, "pid"),
            check.opt_str_param(step_key, "step_key"),
        )

    @property
    def node_name(self) -> str:
        check.invariant(self.node_handle is not None)
        node_handle = cast(NodeHandle, self.node_handle)
        return node_handle.name

    @public
    @property
    def event_type(self) -> DagsterEventType:
        """DagsterEventType: The type of this event."""
        return DagsterEventType(self.event_type_value)

    @public
    @property
    def is_step_event(self) -> bool:
        return self.event_type in STEP_EVENTS

    @public
    @property
    def is_hook_event(self) -> bool:
        return self.event_type in HOOK_EVENTS

    @public
    @property
    def is_alert_event(self) -> bool:
        return self.event_type in ALERT_EVENTS

    @property
    def step_kind(self) -> "StepKind":
        from marie._core.execution.plan.step import StepKind

        return StepKind(self.step_kind_value)

    @public
    @property
    def is_step_success(self) -> bool:
        return self.event_type == DagsterEventType.STEP_SUCCESS

    @public
    @property
    def is_successful_output(self) -> bool:
        return self.event_type == DagsterEventType.STEP_OUTPUT

    @public
    @property
    def is_step_start(self) -> bool:
        return self.event_type == DagsterEventType.STEP_START

    @public
    @property
    def is_step_failure(self) -> bool:
        return self.event_type == DagsterEventType.STEP_FAILURE

    @public
    @property
    def is_resource_init_failure(self) -> bool:
        return self.event_type == DagsterEventType.RESOURCE_INIT_FAILURE

    @public
    @property
    def is_step_skipped(self) -> bool:
        return self.event_type == DagsterEventType.STEP_SKIPPED

    @public
    @property
    def is_step_up_for_retry(self) -> bool:
        return self.event_type == DagsterEventType.STEP_UP_FOR_RETRY

    @public
    @property
    def is_step_restarted(self) -> bool:
        return self.event_type == DagsterEventType.STEP_RESTARTED

    @property
    def is_job_success(self) -> bool:
        return self.event_type == DagsterEventType.RUN_SUCCESS

    @property
    def is_job_failure(self) -> bool:
        return self.event_type == DagsterEventType.RUN_FAILURE

    @property
    def is_run_failure(self) -> bool:
        return self.event_type == DagsterEventType.RUN_FAILURE

    @public
    @property
    def is_failure(self) -> bool:
        return self.event_type in FAILURE_EVENTS

    @property
    def is_job_event(self) -> bool:
        return self.event_type in PIPELINE_EVENTS

    @public
    @property
    def is_engine_event(self) -> bool:
        return self.event_type == DagsterEventType.ENGINE_EVENT

    @public
    @property
    def is_handled_output(self) -> bool:
        return self.event_type == DagsterEventType.HANDLED_OUTPUT

    @public
    @property
    def is_loaded_input(self) -> bool:
        return self.event_type == DagsterEventType.LOADED_INPUT

    @public
    @property
    def is_step_materialization(self) -> bool:
        return self.event_type == DagsterEventType.ASSET_MATERIALIZATION

    @public
    @property
    def is_expectation_result(self) -> bool:
        return self.event_type == DagsterEventType.STEP_EXPECTATION_RESULT

    @public
    @property
    def is_asset_observation(self) -> bool:
        return self.event_type == DagsterEventType.ASSET_OBSERVATION

    @public
    @property
    def is_asset_materialization_planned(self) -> bool:
        return self.event_type == DagsterEventType.ASSET_MATERIALIZATION_PLANNED

    @public
    @property
    def asset_key(self) -> Optional[AssetKey]:
        if self.event_type == DagsterEventType.ASSET_MATERIALIZATION:
            return self.step_materialization_data.materialization.asset_key
        elif self.event_type == DagsterEventType.ASSET_OBSERVATION:
            return self.asset_observation_data.asset_observation.asset_key
        elif self.event_type == DagsterEventType.ASSET_MATERIALIZATION_PLANNED:
            return self.asset_materialization_planned_data.asset_key
        else:
            return None

    @public
    @property
    def partition(self) -> Optional[str]:
        if self.event_type == DagsterEventType.ASSET_MATERIALIZATION:
            return self.step_materialization_data.materialization.partition
        elif self.event_type == DagsterEventType.ASSET_OBSERVATION:
            return self.asset_observation_data.asset_observation.partition
        elif self.event_type == DagsterEventType.ASSET_MATERIALIZATION_PLANNED:
            return self.asset_materialization_planned_data.partition
        else:
            return None

    @property
    def step_input_data(self) -> "StepInputData":
        _assert_type("step_input_data", DagsterEventType.STEP_INPUT, self.event_type)
        return cast(StepInputData, self.event_specific_data)

    @property
    def step_output_data(self) -> StepOutputData:
        _assert_type("step_output_data", DagsterEventType.STEP_OUTPUT, self.event_type)
        return cast(StepOutputData, self.event_specific_data)

    @property
    def step_success_data(self) -> "StepSuccessData":
        _assert_type(
            "step_success_data", DagsterEventType.STEP_SUCCESS, self.event_type
        )
        return cast(StepSuccessData, self.event_specific_data)

    @property
    def step_failure_data(self) -> "StepFailureData":
        _assert_type(
            "step_failure_data", DagsterEventType.STEP_FAILURE, self.event_type
        )
        return cast(StepFailureData, self.event_specific_data)

    @property
    def step_retry_data(self) -> "StepRetryData":
        _assert_type(
            "step_retry_data", DagsterEventType.STEP_UP_FOR_RETRY, self.event_type
        )
        return cast(StepRetryData, self.event_specific_data)

    @property
    def step_materialization_data(self) -> "StepMaterializationData":
        _assert_type(
            "step_materialization_data",
            DagsterEventType.ASSET_MATERIALIZATION,
            self.event_type,
        )
        return cast(StepMaterializationData, self.event_specific_data)

    @property
    def asset_observation_data(self) -> "AssetObservationData":
        _assert_type(
            "asset_observation_data",
            DagsterEventType.ASSET_OBSERVATION,
            self.event_type,
        )
        return cast(AssetObservationData, self.event_specific_data)

    @property
    def asset_materialization_planned_data(self) -> "AssetMaterializationPlannedData":
        _assert_type(
            "asset_materialization_planned",
            DagsterEventType.ASSET_MATERIALIZATION_PLANNED,
            self.event_type,
        )
        return cast(AssetMaterializationPlannedData, self.event_specific_data)

    @property
    def step_expectation_result_data(self) -> "StepExpectationResultData":
        _assert_type(
            "step_expectation_result_data",
            DagsterEventType.STEP_EXPECTATION_RESULT,
            self.event_type,
        )
        return cast(StepExpectationResultData, self.event_specific_data)

    @property
    def materialization(self) -> AssetMaterialization:
        _assert_type(
            "step_materialization_data",
            DagsterEventType.ASSET_MATERIALIZATION,
            self.event_type,
        )
        return cast(StepMaterializationData, self.event_specific_data).materialization

    @property
    def job_failure_data(self) -> "JobFailureData":
        _assert_type("job_failure_data", DagsterEventType.RUN_FAILURE, self.event_type)
        return cast(JobFailureData, self.event_specific_data)

    @property
    def engine_event_data(self) -> "EngineEventData":
        _assert_type(
            "engine_event_data",
            [
                DagsterEventType.ENGINE_EVENT,
                DagsterEventType.RESOURCE_INIT_STARTED,
                DagsterEventType.RESOURCE_INIT_SUCCESS,
                DagsterEventType.RESOURCE_INIT_FAILURE,
                DagsterEventType.STEP_WORKER_STARTED,
                DagsterEventType.STEP_WORKER_STARTING,
            ],
            self.event_type,
        )
        return cast(EngineEventData, self.event_specific_data)

    @property
    def hook_completed_data(self) -> Optional["EventSpecificData"]:
        _assert_type(
            "hook_completed_data", DagsterEventType.HOOK_COMPLETED, self.event_type
        )
        return self.event_specific_data

    @property
    def hook_errored_data(self) -> "HookErroredData":
        _assert_type(
            "hook_errored_data", DagsterEventType.HOOK_ERRORED, self.event_type
        )
        return cast(HookErroredData, self.event_specific_data)

    @property
    def hook_skipped_data(self) -> Optional["EventSpecificData"]:
        _assert_type(
            "hook_skipped_data", DagsterEventType.HOOK_SKIPPED, self.event_type
        )
        return self.event_specific_data

    @property
    def logs_captured_data(self) -> "ComputeLogsCaptureData":
        _assert_type(
            "logs_captured_data", DagsterEventType.LOGS_CAPTURED, self.event_type
        )
        return cast(ComputeLogsCaptureData, self.event_specific_data)

    @staticmethod
    def step_output_event(
        step_context: StepExecutionContext, step_output_data: StepOutputData
    ) -> "DagsterEvent":
        output_def = step_context.op.output_def_named(
            step_output_data.step_output_handle.output_name
        )

        return DagsterEvent.from_step(
            event_type=DagsterEventType.STEP_OUTPUT,
            step_context=step_context,
            event_specific_data=step_output_data,
            message=(
                'Yielded output "{output_name}"{mapping_clause} of type'
                ' "{output_type}".{type_check_clause}'.format(
                    output_name=step_output_data.step_output_handle.output_name,
                    output_type=output_def.dagster_type.display_name,
                    type_check_clause=(
                        " Warning! Type check failed."
                        if not step_output_data.type_check_data.success
                        else " (Type check passed)."
                    )
                    if step_output_data.type_check_data
                    else " (No type check).",
                    mapping_clause=(
                        f' mapping key "{step_output_data.step_output_handle.mapping_key}"'
                    )
                    if step_output_data.step_output_handle.mapping_key
                    else "",
                )
            ),
        )

    @staticmethod
    def step_failure_event(
        step_context: IStepContext,
        step_failure_data: "StepFailureData",
        message=None,
    ) -> "DagsterEvent":
        return DagsterEvent.from_step(
            event_type=DagsterEventType.STEP_FAILURE,
            step_context=step_context,
            event_specific_data=step_failure_data,
            message=(message or f'Execution of step "{step_context.step.key}" failed.'),
        )

    @staticmethod
    def step_retry_event(
        step_context: IStepContext, step_retry_data: "StepRetryData"
    ) -> "DagsterEvent":
        return DagsterEvent.from_step(
            event_type=DagsterEventType.STEP_UP_FOR_RETRY,
            step_context=step_context,
            event_specific_data=step_retry_data,
            message=(
                'Execution of step "{step_key}" failed and has requested a retry{wait_str}.'.format(
                    step_key=step_context.step.key,
                    wait_str=f" in {step_retry_data.seconds_to_wait} seconds"
                    if step_retry_data.seconds_to_wait
                    else "",
                )
            ),
        )

    @staticmethod
    def step_input_event(
        step_context: StepExecutionContext, step_input_data: "StepInputData"
    ) -> "DagsterEvent":
        input_def = step_context.op_def.input_def_named(step_input_data.input_name)

        return DagsterEvent.from_step(
            event_type=DagsterEventType.STEP_INPUT,
            step_context=step_context,
            event_specific_data=step_input_data,
            message='Got input "{input_name}" of type "{input_type}".{type_check_clause}'.format(
                input_name=step_input_data.input_name,
                input_type=input_def.dagster_type.display_name,
                type_check_clause=(
                    " Warning! Type check failed."
                    if not step_input_data.type_check_data.success
                    else " (Type check passed)."
                )
                if step_input_data.type_check_data
                else " (No type check).",
            ),
        )

    @staticmethod
    def step_start_event(step_context: IStepContext) -> "DagsterEvent":
        return DagsterEvent.from_step(
            event_type=DagsterEventType.STEP_START,
            step_context=step_context,
            message='Started execution of step "{step_key}".'.format(
                step_key=step_context.step.key
            ),
        )

    @staticmethod
    def step_restarted_event(
        step_context: IStepContext, previous_attempts: int
    ) -> "DagsterEvent":
        return DagsterEvent.from_step(
            event_type=DagsterEventType.STEP_RESTARTED,
            step_context=step_context,
            message='Started re-execution (attempt # {n}) of step "{step_key}".'.format(
                step_key=step_context.step.key, n=previous_attempts + 1
            ),
        )

    @staticmethod
    def step_success_event(
        step_context: IStepContext, success: "StepSuccessData"
    ) -> "DagsterEvent":
        return DagsterEvent.from_step(
            event_type=DagsterEventType.STEP_SUCCESS,
            step_context=step_context,
            event_specific_data=success,
            message='Finished execution of step "{step_key}" in {duration}.'.format(
                step_key=step_context.step.key,
                duration=format_duration(success.duration_ms),
            ),
        )

    @staticmethod
    def step_skipped_event(step_context: IStepContext) -> "DagsterEvent":
        return DagsterEvent.from_step(
            event_type=DagsterEventType.STEP_SKIPPED,
            step_context=step_context,
            message='Skipped execution of step "{step_key}".'.format(
                step_key=step_context.step.key
            ),
        )

    @staticmethod
    def asset_materialization(
        step_context: IStepContext,
        materialization: AssetMaterialization,
    ) -> "DagsterEvent":
        return DagsterEvent.from_step(
            event_type=DagsterEventType.ASSET_MATERIALIZATION,
            step_context=step_context,
            event_specific_data=StepMaterializationData(materialization),
            message=materialization.description
            if materialization.description
            else "Materialized value{label_clause}.".format(
                label_clause=f" {materialization.label}"
                if materialization.label
                else ""
            ),
        )

    @staticmethod
    def asset_observation(
        step_context: IStepContext, observation: AssetObservation
    ) -> "DagsterEvent":
        return DagsterEvent.from_step(
            event_type=DagsterEventType.ASSET_OBSERVATION,
            step_context=step_context,
            event_specific_data=AssetObservationData(observation),
        )

    @staticmethod
    def step_expectation_result(
        step_context: IStepContext, expectation_result: ExpectationResult
    ) -> "DagsterEvent":
        def _msg():
            if expectation_result.description:
                return expectation_result.description

            return "Expectation{label_clause} {result_verb}".format(
                label_clause=" " + expectation_result.label
                if expectation_result.label
                else "",
                result_verb="passed" if expectation_result.success else "failed",
            )

        return DagsterEvent.from_step(
            event_type=DagsterEventType.STEP_EXPECTATION_RESULT,
            step_context=step_context,
            event_specific_data=StepExpectationResultData(expectation_result),
            message=_msg(),
        )

    @staticmethod
    def job_start(job_context: IPlanContext) -> "DagsterEvent":
        return DagsterEvent.from_job(
            DagsterEventType.RUN_START,
            job_context,
            message=f'Started execution of run for "{job_context.job_name}".',
        )

    @staticmethod
    def job_success(job_context: IPlanContext) -> "DagsterEvent":
        return DagsterEvent.from_job(
            DagsterEventType.RUN_SUCCESS,
            job_context,
            message=f'Finished execution of run for "{job_context.job_name}".',
        )

    @staticmethod
    def job_failure(
        job_context_or_name: Union[IPlanContext, str],
        context_msg: str,
        error_info: Optional[SerializableErrorInfo] = None,
    ) -> "DagsterEvent":
        check.str_param(context_msg, "context_msg")
        if isinstance(job_context_or_name, IPlanContext):
            return DagsterEvent.from_job(
                DagsterEventType.RUN_FAILURE,
                job_context_or_name,
                message=(
                    f'Execution of run for "{job_context_or_name.job_name}" failed. {context_msg}'
                ),
                event_specific_data=JobFailureData(error_info),
            )
        else:
            # when the failure happens trying to bring up context, the job_context hasn't been
            # built and so can't use from_pipeline
            check.str_param(job_context_or_name, "pipeline_name")
            event = DagsterEvent(
                event_type_value=DagsterEventType.RUN_FAILURE.value,
                job_name=job_context_or_name,
                event_specific_data=JobFailureData(error_info),
                message=f'Execution of run for "{job_context_or_name}" failed. {context_msg}',
                pid=os.getpid(),
            )
            return event

    @staticmethod
    def job_canceled(
        job_context: IPlanContext, error_info: Optional[SerializableErrorInfo] = None
    ) -> "DagsterEvent":
        return DagsterEvent.from_job(
            DagsterEventType.RUN_CANCELED,
            job_context,
            message=f'Execution of run for "{job_context.job_name}" canceled.',
            event_specific_data=JobCanceledData(
                check.opt_inst_param(error_info, "error_info", SerializableErrorInfo)
            ),
        )

    @staticmethod
    def step_worker_starting(
        step_context: IStepContext,
        message: str,
        metadata: Mapping[str, MetadataValue],
    ) -> "DagsterEvent":
        return DagsterEvent.from_step(
            DagsterEventType.STEP_WORKER_STARTING,
            step_context,
            message=message,
            event_specific_data=EngineEventData(
                metadata=metadata, marker_start="step_process_start"
            ),
        )

    @staticmethod
    def step_worker_started(
        log_manager: DagsterLogManager,
        job_name: str,
        message: str,
        metadata: Mapping[str, MetadataValue],
        step_key: Optional[str],
    ) -> "DagsterEvent":
        event = DagsterEvent(
            DagsterEventType.STEP_WORKER_STARTED.value,
            job_name=job_name,
            message=message,
            event_specific_data=EngineEventData(
                metadata=metadata, marker_end="step_process_start"
            ),
            pid=os.getpid(),
            step_key=step_key,
        )
        log_manager.log_dagster_event(
            level=logging.DEBUG,
            msg=message,
            dagster_event=event,
        )
        return event

    @staticmethod
    def resource_init_start(
        job_name: str,
        execution_plan: "ExecutionPlan",
        log_manager: DagsterLogManager,
        resource_keys: AbstractSet[str],
    ) -> "DagsterEvent":
        return DagsterEvent.from_resource(
            DagsterEventType.RESOURCE_INIT_STARTED,
            job_name=job_name,
            execution_plan=execution_plan,
            log_manager=log_manager,
            message="Starting initialization of resources [{}].".format(
                ", ".join(sorted(resource_keys))
            ),
            event_specific_data=EngineEventData(metadata={}, marker_start="resources"),
        )

    @staticmethod
    def resource_init_success(
        job_name: str,
        execution_plan: "ExecutionPlan",
        log_manager: DagsterLogManager,
        resource_instances: Mapping[str, Any],
        resource_init_times: Mapping[str, str],
    ) -> "DagsterEvent":
        metadata = {}
        for key in resource_instances.keys():
            metadata[key] = MetadataValue.python_artifact(
                resource_instances[key].__class__
            )
            metadata[f"{key}:init_time"] = resource_init_times[key]

        return DagsterEvent.from_resource(
            DagsterEventType.RESOURCE_INIT_SUCCESS,
            job_name=job_name,
            execution_plan=execution_plan,
            log_manager=log_manager,
            message="Finished initialization of resources [{}].".format(
                ", ".join(sorted(resource_init_times.keys()))
            ),
            event_specific_data=EngineEventData(
                metadata=metadata,
                marker_end="resources",
            ),
        )

    @staticmethod
    def resource_init_failure(
        job_name: str,
        execution_plan: "ExecutionPlan",
        log_manager: DagsterLogManager,
        resource_keys: AbstractSet[str],
        error: SerializableErrorInfo,
    ) -> "DagsterEvent":
        return DagsterEvent.from_resource(
            DagsterEventType.RESOURCE_INIT_FAILURE,
            job_name=job_name,
            execution_plan=execution_plan,
            log_manager=log_manager,
            message="Initialization of resources [{}] failed.".format(
                ", ".join(resource_keys)
            ),
            event_specific_data=EngineEventData(
                metadata={},
                marker_end="resources",
                error=error,
            ),
        )

    @staticmethod
    def resource_teardown_failure(
        job_name: str,
        execution_plan: "ExecutionPlan",
        log_manager: DagsterLogManager,
        resource_keys: AbstractSet[str],
        error: SerializableErrorInfo,
    ) -> "DagsterEvent":
        return DagsterEvent.from_resource(
            DagsterEventType.ENGINE_EVENT,
            job_name=job_name,
            execution_plan=execution_plan,
            log_manager=log_manager,
            message="Teardown of resources [{}] failed.".format(
                ", ".join(resource_keys)
            ),
            event_specific_data=EngineEventData(
                metadata={},
                marker_start=None,
                marker_end=None,
                error=error,
            ),
        )

    @staticmethod
    def engine_event(
        plan_context: IPlanContext,
        message: str,
        event_specific_data: Optional["EngineEventData"] = None,
    ) -> "DagsterEvent":
        if isinstance(plan_context, IStepContext):
            return DagsterEvent.from_step(
                DagsterEventType.ENGINE_EVENT,
                step_context=plan_context,
                event_specific_data=event_specific_data,
                message=message,
            )
        else:
            return DagsterEvent.from_job(
                DagsterEventType.ENGINE_EVENT,
                plan_context,
                message,
                event_specific_data=event_specific_data,
            )

    @staticmethod
    def object_store_operation(
        step_context: IStepContext,
        object_store_operation_result: "ObjectStoreOperation",
    ) -> "DagsterEvent":
        object_store_name = (
            "{object_store_name} ".format(
                object_store_name=object_store_operation_result.object_store_name
            )
            if object_store_operation_result.object_store_name
            else ""
        )

        serialization_strategy_modifier = (
            " using {serialization_strategy_name}".format(
                serialization_strategy_name=object_store_operation_result.serialization_strategy_name
            )
            if object_store_operation_result.serialization_strategy_name
            else ""
        )

        value_name = object_store_operation_result.value_name

        if (
            ObjectStoreOperationType(object_store_operation_result.op)
            == ObjectStoreOperationType.SET_OBJECT
        ):
            message = (
                "Stored intermediate object for output {value_name} in "
                "{object_store_name}object store{serialization_strategy_modifier}."
            ).format(
                value_name=value_name,
                object_store_name=object_store_name,
                serialization_strategy_modifier=serialization_strategy_modifier,
            )
        elif (
            ObjectStoreOperationType(object_store_operation_result.op)
            == ObjectStoreOperationType.GET_OBJECT
        ):
            message = (
                "Retrieved intermediate object for input {value_name} in "
                "{object_store_name}object store{serialization_strategy_modifier}."
            ).format(
                value_name=value_name,
                object_store_name=object_store_name,
                serialization_strategy_modifier=serialization_strategy_modifier,
            )
        elif (
            ObjectStoreOperationType(object_store_operation_result.op)
            == ObjectStoreOperationType.CP_OBJECT
        ):
            message = (
                "Copied intermediate object for input {value_name} from {key} to {dest_key}"
            ).format(
                value_name=value_name,
                key=object_store_operation_result.key,
                dest_key=object_store_operation_result.dest_key,
            )
        else:
            message = ""

        return DagsterEvent.from_step(
            DagsterEventType.OBJECT_STORE_OPERATION,
            step_context,
            event_specific_data=ObjectStoreOperationResultData(
                op=object_store_operation_result.op,
                value_name=value_name,
                address=object_store_operation_result.key,
                metadata={"key": MetadataValue.path(object_store_operation_result.key)},
                version=object_store_operation_result.version,
                mapping_key=object_store_operation_result.mapping_key,
            ),
            message=message,
        )

    @staticmethod
    def handled_output(
        step_context: IStepContext,
        output_name: str,
        manager_key: str,
        message_override: Optional[str] = None,
        metadata: Optional[Mapping[str, MetadataValue]] = None,
    ) -> "DagsterEvent":
        message = f'Handled output "{output_name}" using IO manager "{manager_key}"'
        return DagsterEvent.from_step(
            event_type=DagsterEventType.HANDLED_OUTPUT,
            step_context=step_context,
            event_specific_data=HandledOutputData(
                output_name=output_name,
                manager_key=manager_key,
                metadata=metadata if metadata else {},
            ),
            message=message_override or message,
        )

    @staticmethod
    def loaded_input(
        step_context: IStepContext,
        input_name: str,
        manager_key: str,
        upstream_output_name: Optional[str] = None,
        upstream_step_key: Optional[str] = None,
        message_override: Optional[str] = None,
        metadata: Optional[Mapping[str, MetadataValue]] = None,
    ) -> "DagsterEvent":
        message = f'Loaded input "{input_name}" using input manager "{manager_key}"'
        if upstream_output_name:
            message += (
                f', from output "{upstream_output_name}" of step "{upstream_step_key}"'
            )

        return DagsterEvent.from_step(
            event_type=DagsterEventType.LOADED_INPUT,
            step_context=step_context,
            event_specific_data=LoadedInputData(
                input_name=input_name,
                manager_key=manager_key,
                upstream_output_name=upstream_output_name,
                upstream_step_key=upstream_step_key,
                metadata=metadata if metadata else {},
            ),
            message=message_override or message,
        )

    @staticmethod
    def hook_completed(
        step_context: StepExecutionContext, hook_def: HookDefinition
    ) -> "DagsterEvent":
        event_type = DagsterEventType.HOOK_COMPLETED

        event = DagsterEvent(
            event_type_value=event_type.value,
            job_name=step_context.job_name,
            step_handle=step_context.step.handle,
            node_handle=step_context.step.node_handle,
            step_kind_value=step_context.step.kind.value,
            logging_tags=step_context.event_tags,
            message=(
                f'Finished the execution of hook "{hook_def.name}" triggered for'
                f' "{step_context.op.name}".'
            ),
        )

        step_context.log.log_dagster_event(
            level=logging.DEBUG, msg=event.message or "", dagster_event=event
        )

        return event

    @staticmethod
    def hook_errored(
        step_context: StepExecutionContext, error: HookExecutionError
    ) -> "DagsterEvent":
        event_type = DagsterEventType.HOOK_ERRORED

        event = DagsterEvent(
            event_type_value=event_type.value,
            job_name=step_context.job_name,
            step_handle=step_context.step.handle,
            node_handle=step_context.step.node_handle,
            step_kind_value=step_context.step.kind.value,
            logging_tags=step_context.event_tags,
            event_specific_data=_validate_event_specific_data(
                event_type,
                HookErroredData(
                    error=serializable_error_info_from_exc_info(error.original_exc_info)
                ),
            ),
        )

        step_context.log.log_dagster_event(
            level=logging.ERROR, msg=str(error), dagster_event=event
        )

        return event

    @staticmethod
    def hook_skipped(
        step_context: StepExecutionContext, hook_def: HookDefinition
    ) -> "DagsterEvent":
        event_type = DagsterEventType.HOOK_SKIPPED

        event = DagsterEvent(
            event_type_value=event_type.value,
            job_name=step_context.job_name,
            step_handle=step_context.step.handle,
            node_handle=step_context.step.node_handle,
            step_kind_value=step_context.step.kind.value,
            logging_tags=step_context.event_tags,
            message=(
                'Skipped the execution of hook "{hook_name}". It did not meet its triggering '
                'condition during the execution of "{solid_name}".'
            ).format(hook_name=hook_def.name, solid_name=step_context.op.name),
        )

        step_context.log.log_dagster_event(
            level=logging.DEBUG, msg=event.message or "", dagster_event=event
        )

        return event

    @staticmethod
    def legacy_compute_log_step_event(step_context: StepExecutionContext):
        step_key = step_context.step.key
        return DagsterEvent.from_step(
            DagsterEventType.LOGS_CAPTURED,
            step_context,
            message=f"Started capturing logs for step: {step_key}.",
            event_specific_data=ComputeLogsCaptureData(
                step_keys=[step_key],
                file_key=step_key,
            ),
        )

    @staticmethod
    def capture_logs(
        job_context: IPlanContext,
        step_keys: Sequence[str],
        log_key: Sequence[str],
        log_context: CapturedLogContext,
    ):
        file_key = log_key[-1]
        return DagsterEvent.from_job(
            DagsterEventType.LOGS_CAPTURED,
            job_context,
            message=f"Started capturing logs in process (pid: {os.getpid()}).",
            event_specific_data=ComputeLogsCaptureData(
                step_keys=step_keys,
                file_key=file_key,
                external_stdout_url=log_context.external_stdout_url,
                external_stderr_url=log_context.external_stderr_url,
                external_url=log_context.external_url,
            ),
        )


def get_step_output_event(
    events: Sequence[DagsterEvent], step_key: str, output_name: Optional[str] = "result"
) -> Optional["DagsterEvent"]:
    check.sequence_param(events, "events", of_type=DagsterEvent)
    check.str_param(step_key, "step_key")
    check.str_param(output_name, "output_name")
    for event in events:
        if (
            event.event_type == DagsterEventType.STEP_OUTPUT
            and event.step_key == step_key
            and event.step_output_data.output_name == output_name
        ):
            return event
    return None


class AssetObservationData(
    NamedTuple("_AssetObservation", [("asset_observation", AssetObservation)])
):
    def __new__(cls, asset_observation: AssetObservation):
        return super(AssetObservationData, cls).__new__(
            cls,
            asset_observation=check.inst_param(
                asset_observation, "asset_observation", AssetObservation
            ),
        )


class StepMaterializationData(
    NamedTuple(
        "_StepMaterializationData",
        [
            ("materialization", AssetMaterialization),
            ("asset_lineage", Sequence[AssetLineageInfo]),
        ],
    )
):
    def __new__(
        cls,
        materialization: AssetMaterialization,
        asset_lineage: Optional[Sequence[AssetLineageInfo]] = None,
    ):
        return super(StepMaterializationData, cls).__new__(
            cls,
            materialization=check.inst_param(
                materialization, "materialization", AssetMaterialization
            ),
            asset_lineage=check.opt_sequence_param(
                asset_lineage, "asset_lineage", of_type=AssetLineageInfo
            ),
        )


class AssetMaterializationPlannedData(
    NamedTuple(
        "_AssetMaterializationPlannedData",
        [("asset_key", AssetKey), ("partition", Optional[str])],
    )
):
    def __new__(cls, asset_key: AssetKey, partition: Optional[str] = None):
        return super(AssetMaterializationPlannedData, cls).__new__(
            cls,
            asset_key=check.inst_param(asset_key, "asset_key", AssetKey),
            partition=check.opt_str_param(partition, "partition"),
        )


class StepExpectationResultData(
    NamedTuple(
        "_StepExpectationResultData",
        [
            ("expectation_result", ExpectationResult),
        ],
    )
):
    def __new__(cls, expectation_result: ExpectationResult):
        return super(StepExpectationResultData, cls).__new__(
            cls,
            expectation_result=check.inst_param(
                expectation_result, "expectation_result", ExpectationResult
            ),
        )


class ObjectStoreOperationResultData(
    NamedTuple(
        "_ObjectStoreOperationResultData",
        [
            ("op", ObjectStoreOperationType),
            ("value_name", Optional[str]),
            ("metadata", Mapping[str, MetadataValue]),
            ("address", Optional[str]),
            ("version", Optional[str]),
            ("mapping_key", Optional[str]),
        ],
    )
):
    def __new__(
        cls,
        op: ObjectStoreOperationType,
        value_name: Optional[str] = None,
        metadata: Optional[Mapping[str, MetadataValue]] = None,
        address: Optional[str] = None,
        version: Optional[str] = None,
        mapping_key: Optional[str] = None,
    ):
        return super(ObjectStoreOperationResultData, cls).__new__(
            cls,
            op=cast(ObjectStoreOperationType, check.str_param(op, "op")),
            value_name=check.opt_str_param(value_name, "value_name"),
            metadata=normalize_metadata(
                check.opt_mapping_param(metadata, "metadata", key_type=str)
            ),
            address=check.opt_str_param(address, "address"),
            version=check.opt_str_param(version, "version"),
            mapping_key=check.opt_str_param(mapping_key, "mapping_key"),
        )


class EngineEventData(
    NamedTuple(
        "_EngineEventData",
        [
            ("metadata", Mapping[str, MetadataValue]),
            ("error", Optional[SerializableErrorInfo]),
            ("marker_start", Optional[str]),
            ("marker_end", Optional[str]),
        ],
    )
):
    # serdes log
    # * added optional error
    # * added marker_start / marker_end
    #
    def __new__(
        cls,
        metadata: Optional[Mapping[str, RawMetadataValue]] = None,
        error: Optional[SerializableErrorInfo] = None,
        marker_start: Optional[str] = None,
        marker_end: Optional[str] = None,
    ):
        return super(EngineEventData, cls).__new__(
            cls,
            metadata=normalize_metadata(
                check.opt_mapping_param(metadata, "metadata", key_type=str)
            ),
            error=check.opt_inst_param(error, "error", SerializableErrorInfo),
            marker_start=check.opt_str_param(marker_start, "marker_start"),
            marker_end=check.opt_str_param(marker_end, "marker_end"),
        )

    @staticmethod
    def in_process(
        pid: int, step_keys_to_execute: Optional[Sequence[str]] = None
    ) -> "EngineEventData":
        return EngineEventData(
            metadata={
                "pid": MetadataValue.text(str(pid)),
                **(
                    {"step_keys": MetadataValue.text(str(step_keys_to_execute))}
                    if step_keys_to_execute
                    else {}
                ),
            }
        )

    @staticmethod
    def multiprocess(
        pid: int, step_keys_to_execute: Optional[Sequence[str]] = None
    ) -> "EngineEventData":
        return EngineEventData(
            metadata={
                "pid": MetadataValue.text(str(pid)),
                **(
                    {"step_keys": MetadataValue.text(str(step_keys_to_execute))}
                    if step_keys_to_execute
                    else {}
                ),
            }
        )

    @staticmethod
    def interrupted(steps_interrupted: Sequence[str]) -> "EngineEventData":
        return EngineEventData(
            metadata={"steps_interrupted": MetadataValue.text(str(steps_interrupted))}
        )

    @staticmethod
    def engine_error(error: SerializableErrorInfo) -> "EngineEventData":
        return EngineEventData(metadata={}, error=error)


class JobFailureData(
    NamedTuple(
        "_JobFailureData",
        [
            ("error", Optional[SerializableErrorInfo]),
        ],
    )
):
    def __new__(cls, error: Optional[SerializableErrorInfo]):
        return super(JobFailureData, cls).__new__(
            cls, error=check.opt_inst_param(error, "error", SerializableErrorInfo)
        )


class JobCanceledData(
    NamedTuple(
        "_JobCanceledData",
        [
            ("error", Optional[SerializableErrorInfo]),
        ],
    )
):
    def __new__(cls, error: Optional[SerializableErrorInfo]):
        return super(JobCanceledData, cls).__new__(
            cls, error=check.opt_inst_param(error, "error", SerializableErrorInfo)
        )


class HookErroredData(
    NamedTuple(
        "_HookErroredData",
        [
            ("error", SerializableErrorInfo),
        ],
    )
):
    def __new__(cls, error: SerializableErrorInfo):
        return super(HookErroredData, cls).__new__(
            cls, error=check.inst_param(error, "error", SerializableErrorInfo)
        )


class HandledOutputData(
    NamedTuple(
        "_HandledOutputData",
        [
            ("output_name", str),
            ("manager_key", str),
            ("metadata", Mapping[str, MetadataValue]),
        ],
    )
):
    def __new__(
        cls,
        output_name: str,
        manager_key: str,
        metadata: Optional[Mapping[str, MetadataValue]] = None,
    ):
        return super(HandledOutputData, cls).__new__(
            cls,
            output_name=check.str_param(output_name, "output_name"),
            manager_key=check.str_param(manager_key, "manager_key"),
            metadata=normalize_metadata(
                check.opt_mapping_param(metadata, "metadata", key_type=str)
            ),
        )


class LoadedInputData(
    NamedTuple(
        "_LoadedInputData",
        [
            ("input_name", str),
            ("manager_key", str),
            ("upstream_output_name", Optional[str]),
            ("upstream_step_key", Optional[str]),
            ("metadata", Mapping[str, MetadataValue]),
        ],
    )
):
    def __new__(
        cls,
        input_name: str,
        manager_key: str,
        upstream_output_name: Optional[str] = None,
        upstream_step_key: Optional[str] = None,
        metadata: Optional[Mapping[str, MetadataValue]] = None,
    ):
        return super(LoadedInputData, cls).__new__(
            cls,
            input_name=check.str_param(input_name, "input_name"),
            manager_key=check.str_param(manager_key, "manager_key"),
            upstream_output_name=check.opt_str_param(
                upstream_output_name, "upstream_output_name"
            ),
            upstream_step_key=check.opt_str_param(
                upstream_step_key, "upstream_step_key"
            ),
            metadata=normalize_metadata(
                check.opt_mapping_param(metadata, "metadata", key_type=str)
            ),
        )


class ComputeLogsCaptureData(
    NamedTuple(
        "_ComputeLogsCaptureData",
        [
            ("file_key", str),  # renamed log_key => file_key to avoid confusion
            ("step_keys", Sequence[str]),
            ("external_url", Optional[str]),
            ("external_stdout_url", Optional[str]),
            ("external_stderr_url", Optional[str]),
        ],
    )
):
    def __new__(
        cls,
        file_key: str,
        step_keys: Sequence[str],
        external_url: Optional[str] = None,
        external_stdout_url: Optional[str] = None,
        external_stderr_url: Optional[str] = None,
    ):
        return super(ComputeLogsCaptureData, cls).__new__(
            cls,
            file_key=check.str_param(file_key, "file_key"),
            step_keys=check.opt_list_param(step_keys, "step_keys", of_type=str),
            external_url=check.opt_str_param(external_url, "external_url"),
            external_stdout_url=check.opt_str_param(
                external_stdout_url, "external_stdout_url"
            ),
            external_stderr_url=check.opt_str_param(
                external_stderr_url, "external_stderr_url"
            ),
        )
