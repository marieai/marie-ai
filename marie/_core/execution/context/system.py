from abc import abstractmethod, ABC
from typing import (
    Mapping,
    Optional,
    Any,
    Dict,
    NamedTuple,
    AbstractSet,
    cast,
    Set,
    List,
)

from botocore.retries.standard import RetryPolicy

from marie import check
from marie._annotations import public
from marie._core.definitions import NodeHandle
from marie._core.definitions.data_version import DataVersion
from marie._core.definitions.events import AssetKey
from marie._core.definitions.job_base import IJob
from marie._core.definitions.job_definition import JobDefinition
from marie._core.definitions.scoped_resources_builder import (
    ScopedResourcesBuilder,
    Resources,
)
from marie._core.errors import DagsterInvariantViolationError
from marie._core.event_api import EventLogRecord
from marie._core.execution.context.hook import HookContext
from marie._core.execution.context.output import OutputContext
from marie._core.execution.plan.handle import ResolvedFromDynamicStepHandle, StepHandle
from marie._core.execution.plan.outputs import StepOutputHandle
from marie._core.execution.plan.plan import ExecutionPlan
from marie._core.execution.plan.state import KnownExecutionState
from marie._core.execution.plan.step import ExecutionStep
from marie._core.execution.retries import RetryMode
from marie._core.instance import DagsterInstance
from marie._core.log_manager import DagsterLogManager
from marie._core.storage.marie_run import DagsterRun
from marie._core.system_config.objects import ResolvedRunConfig


class IOManager:
    pass


class OpDefinition:
    pass


class OpNode:
    pass


class PartitionsDefinition:
    pass


class DagsterType:
    pass


class InputContext:
    pass


class Executor:
    pass


class IPlanContext(ABC):
    """Context interface to represent run information that does not require access to user code.

    The information available via this interface is accessible to the system throughout a run.
    """

    @property
    @abstractmethod
    def plan_data(self) -> "PlanData":
        raise NotImplementedError()

    @property
    def job(self) -> IJob:
        return self.plan_data.job

    @property
    def dagster_run(self) -> DagsterRun:
        return self.plan_data.dagster_run

    @property
    def run_id(self) -> str:
        return self.dagster_run.run_id

    @property
    def run_config(self) -> Mapping[str, object]:
        return self.dagster_run.run_config

    @property
    def job_name(self) -> str:
        return self.dagster_run.job_name

    @property
    def instance(self) -> "DagsterInstance":
        return self.plan_data.instance

    @property
    def raise_on_error(self) -> bool:
        return self.plan_data.raise_on_error

    @property
    def retry_mode(self) -> RetryMode:
        return self.plan_data.retry_mode

    @property
    def execution_plan(self) -> "ExecutionPlan":
        return self.plan_data.execution_plan

    @property
    @abstractmethod
    def output_capture(self) -> Optional[Mapping[StepOutputHandle, Any]]:
        raise NotImplementedError()

    @property
    def log(self) -> DagsterLogManager:
        raise NotImplementedError()

    @property
    def logging_tags(self) -> Mapping[str, str]:
        return self.log.logging_metadata.all_tags()

    @property
    def event_tags(self) -> Mapping[str, str]:
        return self.log.logging_metadata.event_tags()

    def has_tag(self, key: str) -> bool:
        check.str_param(key, "key")
        return key in self.dagster_run.tags

    def get_tag(self, key: str) -> Optional[str]:
        check.str_param(key, "key")
        return self.dagster_run.tags.get(key)


class PlanData(NamedTuple):
    """The data about a run that is available during both orchestration and execution.

    This object does not contain any information that requires access to user code, such as the
    pipeline definition and resources.
    """

    job: IJob
    dagster_run: DagsterRun
    instance: "DagsterInstance"
    execution_plan: "ExecutionPlan"
    raise_on_error: bool = False
    retry_mode: RetryMode = RetryMode.DISABLED


class ExecutionData(NamedTuple):
    """The data that is available to the system during execution.

    This object contains information that requires access to user code, such as the pipeline
    definition and resources.
    """

    # scoped_resources_builder: ScopedResourcesBuilder
    # resolved_run_config: ResolvedRunConfig
    job_def: JobDefinition


class IStepContext(IPlanContext):
    """Interface to represent data to be available during either step orchestration or execution."""

    @property
    @abstractmethod
    def step(self) -> ExecutionStep:
        raise NotImplementedError()

    @property
    @abstractmethod
    def node_handle(self) -> "NodeHandle":
        raise NotImplementedError()


class PlanOrchestrationContext(IPlanContext):
    """Context for the orchestration of a run.

    This context assumes inability to run user code directly.
    """

    def __init__(
        self,
        plan_data: PlanData,
        log_manager: DagsterLogManager,
        executor: Executor,
        output_capture: Optional[Dict[StepOutputHandle, Any]],
        resume_from_failure: bool = False,
    ):
        self._plan_data = plan_data
        self._log_manager = log_manager
        self._executor = executor
        self._output_capture = output_capture
        self._resume_from_failure = resume_from_failure

    @property
    def plan_data(self) -> PlanData:
        return self._plan_data

    @property
    def log(self) -> DagsterLogManager:
        return self._log_manager

    @property
    def executor(self) -> Executor:
        return self._executor

    @property
    def output_capture(self) -> Optional[Dict[StepOutputHandle, Any]]:
        return self._output_capture

    def for_step(self, step: ExecutionStep) -> "IStepContext":
        return StepOrchestrationContext(
            plan_data=self.plan_data,
            log_manager=self._log_manager.with_tags(**step.logging_tags),
            executor=self.executor,
            step=step,
            output_capture=self.output_capture,
        )

    @property
    def resume_from_failure(self) -> bool:
        return self._resume_from_failure


class StepOrchestrationContext(PlanOrchestrationContext, IStepContext):
    """Context for the orchestration of a step.

    This context assumes inability to run user code directly. Thus, it does not include any resource
    information.
    """

    def __init__(
        self,
        plan_data: PlanData,
        log_manager: DagsterLogManager,
        executor: Executor,
        step: ExecutionStep,
        output_capture: Optional[Dict[StepOutputHandle, Any]],
    ):
        super(StepOrchestrationContext, self).__init__(
            plan_data, log_manager, executor, output_capture
        )
        self._step = step

    @property
    def step(self) -> ExecutionStep:
        return self._step

    @property
    def node_handle(self) -> "NodeHandle":
        return self.step.node_handle


class PlanExecutionContext(IPlanContext):
    """Context for the execution of a plan.

    This context assumes that user code can be run directly, and thus includes resource and
    information.
    """

    def __init__(
        self,
        plan_data: PlanData,
        execution_data: ExecutionData,
        log_manager: DagsterLogManager,
        output_capture: Optional[Dict[StepOutputHandle, Any]] = None,
    ):
        self._plan_data = plan_data
        self._execution_data = execution_data
        self._log_manager = log_manager
        self._output_capture = output_capture

    @property
    def plan_data(self) -> PlanData:
        return self._plan_data

    @property
    def output_capture(self) -> Optional[Dict[StepOutputHandle, Any]]:
        return self._output_capture

    def for_step(
        self,
        step: ExecutionStep,
        known_state: Optional["KnownExecutionState"] = None,
    ) -> IStepContext:
        return StepExecutionContext(
            plan_data=self.plan_data,
            execution_data=self._execution_data,
            log_manager=self._log_manager.with_tags(**step.logging_tags),
            step=step,
            output_capture=self.output_capture,
            known_state=known_state,
        )

    @property
    def job_def(self) -> JobDefinition:
        return self._execution_data.job_def

    @property
    def resolved_run_config(self) -> ResolvedRunConfig:
        return self._execution_data.resolved_run_config

    @property
    def scoped_resources_builder(self) -> ScopedResourcesBuilder:
        return self._execution_data.scoped_resources_builder

    @property
    def log(self) -> DagsterLogManager:
        return self._log_manager

    @property
    def partitions_def(self) -> Optional[PartitionsDefinition]:
        from marie._core.definitions.job_definition import JobDefinition

        job_def = self._execution_data.job_def
        if not isinstance(job_def, JobDefinition):
            check.failed(
                "Can only call 'partitions_def', when using jobs, not legacy pipelines",
            )
        partitions_def = job_def.partitions_def
        return partitions_def

    @property
    def has_partitions(self) -> bool:
        raise NotImplementedError()

    @property
    def partition_key(self) -> str:
        raise NotImplementedError()

    @property
    def asset_partition_key_range(self) -> Any:
        raise NotImplementedError()

    @property
    def partition_time_window(self) -> Any:
        raise NotImplementedError()

    @property
    def has_partition_key(self) -> bool:
        raise NotImplementedError()

    @property
    def has_partition_key_range(self) -> bool:
        raise NotImplementedError()

    def for_type(self, dagster_type: DagsterType) -> "TypeCheckContext":
        return TypeCheckContext(
            self.run_id,
            self.log,
            self._execution_data.scoped_resources_builder,
            dagster_type,
        )


class PartitionKeyRange:
    pass


class PartitionsSubset:
    pass


class HookDefinition:
    pass


class StepExecutionContext(PlanExecutionContext, IStepContext):
    """Context for the execution of a step. Users should not instantiate this class directly.

    This context assumes that user code can be run directly, and thus includes resource and information.
    """

    def __init__(
        self,
        plan_data: PlanData,
        execution_data: ExecutionData,
        log_manager: DagsterLogManager,
        step: ExecutionStep,
        output_capture: Optional[Dict[StepOutputHandle, Any]],
        known_state: Optional["KnownExecutionState"],
    ):
        pass

    @property
    def step(self) -> ExecutionStep:
        return self._step

    @property
    def node_handle(self) -> "NodeHandle":
        return self.step.node_handle

    @property
    def required_resource_keys(self) -> AbstractSet[str]:
        return self._required_resource_keys

    @property
    def resources(self) -> "Resources":
        return self._resources

    @property
    def step_launcher(self) -> Optional[Any]:  # Optional[StepLauncher]
        return self._step_launcher

    @property
    def op_def(self) -> OpDefinition:
        return self.op.definition

    @property
    def job_def(self) -> "JobDefinition":
        return self._execution_data.job_def

    @property
    def op(self) -> OpNode:
        return self.job_def.get_op(self._step.node_handle)

    @property
    def op_retry_policy(self) -> Optional[RetryPolicy]:
        return self.job_def.get_retry_policy_for_handle(self.node_handle)

    def describe_op(self) -> str:
        return f'op "{str(self.node_handle)}"'

    def get_io_manager(self, step_output_handle: StepOutputHandle) -> IOManager:
        step_output = self.execution_plan.get_step_output(step_output_handle)
        io_manager_key = (
            self.job_def.get_node(step_output.node_handle)
            .output_def_named(step_output.name)
            .io_manager_key
        )

        output_manager = getattr(self.resources, io_manager_key)
        return check.inst(output_manager, IOManager)

    def get_output_context(self, step_output_handle: StepOutputHandle) -> OutputContext:
        raise NotImplementedError()

    def for_input_manager(
        self,
        name: str,
        config: Any,
        metadata: Any,
        dagster_type: DagsterType,
        source_handle: Optional[StepOutputHandle] = None,
        resource_config: Any = None,
        resources: Optional["Resources"] = None,
        artificial_output_context: Optional["OutputContext"] = None,
    ) -> InputContext:
        raise NotImplementedError()

    def for_hook(self, hook_def: HookDefinition) -> "HookContext":
        from .hook import HookContext

        return HookContext(self, hook_def)

    def get_known_state(self) -> "KnownExecutionState":
        if not self._known_state:
            check.failed(
                "Attempted to access KnownExecutionState but it was not provided at context"
                " creation"
            )
        return self._known_state

    def can_load(
        self,
        step_output_handle: StepOutputHandle,
    ) -> bool:
        # can load from upstream in the same run
        if step_output_handle in self.get_known_state().ready_outputs:
            return True

        if (
            self._should_load_from_previous_runs(step_output_handle)
            # should and can load from a previous run
            and self._get_source_run_id_from_logs(step_output_handle)
        ):
            return True

        return False

    def observe_output(
        self, output_name: str, mapping_key: Optional[str] = None
    ) -> None:
        if mapping_key:
            if output_name not in self._seen_outputs:
                self._seen_outputs[output_name] = set()
            cast(Set[str], self._seen_outputs[output_name]).add(mapping_key)
        else:
            self._seen_outputs[output_name] = "seen"

    def has_seen_output(
        self, output_name: str, mapping_key: Optional[str] = None
    ) -> bool:
        if mapping_key:
            return (
                output_name in self._seen_outputs
                and mapping_key in self._seen_outputs[output_name]
            )
        return output_name in self._seen_outputs

    def add_output_metadata(
        self,
        metadata: Mapping[str, Any],
        output_name: Optional[str] = None,
        mapping_key: Optional[str] = None,
    ) -> None:
        if output_name is None and len(self.op_def.output_defs) == 1:
            output_def = self.op_def.output_defs[0]
            output_name = output_def.name
        elif output_name is None:
            raise DagsterInvariantViolationError(
                "Attempted to log metadata without providing output_name, but multiple outputs"
                " exist. Please provide an output_name to the invocation of"
                " `context.add_output_metadata`."
            )
        else:
            output_def = self.op_def.output_def_named(output_name)

        if self.has_seen_output(output_name, mapping_key):
            output_desc = (
                f"output '{output_def.name}'"
                if not mapping_key
                else f"output '{output_def.name}' with mapping_key '{mapping_key}'"
            )
            raise DagsterInvariantViolationError(
                f"In {self.op_def.node_type_str} '{self.op.name}', attempted to log output"
                f" metadata for {output_desc} which has already been yielded. Metadata must be"
                " logged before the output is yielded."
            )
        if output_def.is_dynamic and not mapping_key:
            raise DagsterInvariantViolationError(
                f"In {self.op_def.node_type_str} '{self.op.name}', attempted to log metadata"
                f" for dynamic output '{output_def.name}' without providing a mapping key. When"
                " logging metadata for a dynamic output, it is necessary to provide a mapping key."
            )

        if output_name in self._output_metadata:
            if not mapping_key or mapping_key in self._output_metadata[output_name]:
                raise DagsterInvariantViolationError(
                    f"In {self.op_def.node_type_str} '{self.op.name}', attempted to log"
                    f" metadata for output '{output_name}' more than once."
                )
        if mapping_key:
            if output_name not in self._output_metadata:
                self._output_metadata[output_name] = {}
            self._output_metadata[output_name][mapping_key] = metadata

        else:
            self._output_metadata[output_name] = metadata

    def get_output_metadata(
        self, output_name: str, mapping_key: Optional[str] = None
    ) -> Optional[Mapping[str, Any]]:
        metadata = self._output_metadata.get(output_name)
        if mapping_key and metadata:
            return metadata.get(mapping_key)
        return metadata

    def _get_source_run_id_from_logs(
        self, step_output_handle: StepOutputHandle
    ) -> Optional[str]:
        # walk through event logs to find the right run_id based on the run lineage

        parent_state = self.get_known_state().parent_state
        while parent_state:
            # if the parent run has yielded an StepOutput event for the given step output,
            # we find the source run id
            if step_output_handle in parent_state.produced_outputs:
                return parent_state.run_id

            # else, keep looking backwards
            parent_state = parent_state.get_parent_state()

        # When a fixed path is provided via io manager, it's able to run step subset using an execution
        # plan when the ascendant outputs were not previously created by dagster-controlled
        # computations. for example, in backfills, with fixed path io manager, we allow users to
        # "re-execute" runs with steps where the outputs weren't previously stored by dagster.

        # Warn about this special case because it will also reach here when all previous runs have
        # skipped yielding this output. From the logs, we have no easy way to differentiate the fixed
        # path case and the skipping case, until we record the skipping info in KnownExecutionState,
        # i.e. resolve https://github.com/dagster-io/dagster/issues/3511
        self.log.warning(
            f"No previously stored outputs found for source {step_output_handle}. "
            "This is either because you are using an IO Manager that does not depend on run ID, "
            "or because all the previous runs have skipped the output in conditional execution."
        )
        return None

    def _should_load_from_previous_runs(
        self, step_output_handle: StepOutputHandle
    ) -> bool:
        # should not load if not a re-execution
        if self.dagster_run.parent_run_id is None:
            return False
        # should not load if re-executing the entire pipeline
        if self.dagster_run.step_keys_to_execute is None:
            return False

        # should not load if the entire dynamic step is being executed in the current run
        handle = StepHandle.parse_from_key(step_output_handle.step_key)
        if (
            isinstance(handle, ResolvedFromDynamicStepHandle)
            and handle.unresolved_form.to_key() in self.dagster_run.step_keys_to_execute
        ):
            return False

        # should not load if this step is being executed in the current run
        return step_output_handle.step_key not in self.dagster_run.step_keys_to_execute

    def _get_source_run_id(self, step_output_handle: StepOutputHandle) -> Optional[str]:
        if self._should_load_from_previous_runs(step_output_handle):
            return self._get_source_run_id_from_logs(step_output_handle)
        else:
            return self.dagster_run.run_id

    def capture_step_exception(self, exception: BaseException):
        self._step_exception = check.inst_param(exception, "exception", BaseException)

    @property
    def step_exception(self) -> Optional[BaseException]:
        return self._step_exception

    @property
    def step_output_capture(self) -> Optional[Dict[StepOutputHandle, Any]]:
        return self._step_output_capture

    @property
    def previous_attempt_count(self) -> int:
        return (
            self.get_known_state().get_retry_state().get_attempt_count(self._step.key)
        )

    @property
    def op_config(self) -> Any:
        op_config = self.resolved_run_config.ops.get(str(self.node_handle))
        return op_config.config if op_config else None

    @property
    def step_materializes_assets(self) -> bool:
        step_outputs = self.step.step_outputs
        if len(step_outputs) == 0:
            return False
        else:
            asset_info = self.job_def.asset_layer.asset_info_for_output(
                self.node_handle, step_outputs[0].name
            )
            return asset_info is not None

    def set_data_version(
        self, asset_key: AssetKey, data_version: "DataVersion"
    ) -> None:
        self._data_version_cache[asset_key] = data_version

    def has_data_version(self, asset_key: AssetKey) -> bool:
        return asset_key in self._data_version_cache

    def get_data_version(self, asset_key: AssetKey) -> "DataVersion":
        return self._data_version_cache[asset_key]

    @property
    def input_asset_records(
        self,
    ) -> Optional[Mapping[AssetKey, Optional["EventLogRecord"]]]:
        return self._input_asset_records

    @property
    def is_external_input_asset_records_loaded(self) -> bool:
        return self._is_external_input_asset_records_loaded

    def get_input_asset_record(self, key: AssetKey) -> Optional["EventLogRecord"]:
        if key not in self._input_asset_records:
            self._fetch_input_asset_record(key)
        return self._input_asset_records[key]

    # "external" refers to records for inputs generated outside of this step
    def fetch_external_input_asset_records(self) -> None:
        output_keys: List[AssetKey] = []
        for step_output in self.step.step_outputs:
            asset_info = self.job_def.asset_layer.asset_info_for_output(
                self.node_handle, step_output.name
            )
            if asset_info is None or not asset_info.is_required:
                continue
            output_keys.append(asset_info.key)

        all_dep_keys: List[AssetKey] = []
        for output_key in output_keys:
            if output_key not in self.job_def.asset_layer._asset_deps:  # noqa: SLF001
                continue
            dep_keys = self.job_def.asset_layer.upstream_assets_for_asset(output_key)
            for key in dep_keys:
                if key not in all_dep_keys and key not in output_keys:
                    all_dep_keys.append(key)

        self._input_asset_records = {}
        for key in all_dep_keys:
            self._fetch_input_asset_record(key)
        self._is_external_input_asset_records_loaded = True

    def _fetch_input_asset_record(self, key: AssetKey, retries: int = 0) -> None:
        event = self.instance.get_latest_data_version_record(key)
        if key in self._data_version_cache and retries <= 5:
            event_data_version = event.data_version if event else None

            if event_data_version == self._data_version_cache[key]:
                self._input_asset_records[key] = event
            else:
                self._fetch_input_asset_record(key, retries + 1)
        else:
            self._input_asset_records[key] = event

    # Call this to clear the cache for an input asset record. This is necessary when an old
    # materialization for an asset was loaded during `fetch_external_input_asset_records` because an
    # intrastep asset is not required, but then that asset is materialized during the step. If we
    # don't clear the cache for this asset, then we won't use the most up-to-date asset record.
    def wipe_input_asset_record(self, key: AssetKey) -> None:
        if key in self._input_asset_records:
            del self._input_asset_records[key]

    def has_asset_partitions_for_input(self, input_name: str) -> bool:
        asset_layer = self.job_def.asset_layer
        upstream_asset_key = asset_layer.asset_key_for_input(
            self.node_handle, input_name
        )

        return (
            upstream_asset_key is not None
            and asset_layer.partitions_def_for_asset(upstream_asset_key) is not None
        )

    def asset_partition_key_range_for_input(self, input_name: str) -> PartitionKeyRange:
        subset = self.asset_partitions_subset_for_input(input_name)
        partition_key_ranges = subset.get_partition_key_ranges(
            dynamic_partitions_store=self.instance
        )

        if len(partition_key_ranges) != 1:
            check.failed(
                (
                    "Tried to access asset partition key range, but there are "
                    f"({len(partition_key_ranges)}) key ranges associated with this input."
                ),
            )

        return partition_key_ranges[0]

    def asset_partitions_subset_for_input(self, input_name: str) -> PartitionsSubset:
        raise NotImplementedError()

    def asset_partition_key_for_input(self, input_name: str) -> str:
        start, end = self.asset_partition_key_range_for_input(input_name)
        if start == end:
            return start
        else:
            check.failed(
                f"Tried to access partition key for input '{input_name}' of step '{self.step.key}',"
                f" but the step input has a partition range: '{start}' to '{end}'."
            )

    def _partitions_def_for_output(
        self, output_name: str
    ) -> Optional[PartitionsDefinition]:
        asset_info = self.job_def.asset_layer.asset_info_for_output(
            node_handle=self.node_handle, output_name=output_name
        )
        if asset_info:
            return asset_info.partitions_def
        else:
            return None

    def has_asset_partitions_for_output(self, output_name: str) -> bool:
        return self._partitions_def_for_output(output_name) is not None

    def asset_partition_key_range_for_output(
        self, output_name: str
    ) -> PartitionKeyRange:
        if self._partitions_def_for_output(output_name) is not None:
            return self.asset_partition_key_range

        check.failed("The output has no asset partitions")

    def asset_partition_key_for_output(self, output_name: str) -> str:
        start, end = self.asset_partition_key_range_for_output(output_name)
        if start == end:
            return start
        else:
            check.failed(
                f"Tried to access partition key for output '{output_name}' of step"
                f" '{self.step.key}', but the step output has a partition range: '{start}' to"
                f" '{end}'."
            )

    def get_type_loader_context(self) -> "DagsterTypeLoaderContext":
        return DagsterTypeLoaderContext(
            plan_data=self.plan_data,
            execution_data=self._execution_data,
            log_manager=self._log_manager,
            step=self.step,
            output_capture=self._output_capture,
            known_state=self._known_state,
        )


class TypeCheckContext:
    """The ``context`` object available to a type check function on a DagsterType."""

    def __init__(
        self,
        run_id: str,
        log_manager: DagsterLogManager,
        scoped_resources_builder: ScopedResourcesBuilder,
        dagster_type: DagsterType,
    ):
        self._run_id = run_id
        self._log = log_manager
        self._resources = scoped_resources_builder.build(
            dagster_type.required_resource_keys
        )

    @public
    @property
    def resources(self) -> "Resources":
        """An object whose attributes contain the resources available to this op."""
        return self._resources

    @public
    @property
    def run_id(self) -> str:
        """The id of this job run."""
        return self._run_id

    @public
    @property
    def log(self) -> DagsterLogManager:
        """Centralized log dispatch from user code."""
        return self._log


class DagsterTypeLoaderContext(StepExecutionContext):
    """The context object provided to a :py:class:`@dagster_type_loader <dagster_type_loader>`-decorated function during execution.

    Users should not construct this object directly.
    """

    @public
    @property
    def resources(self) -> "Resources":
        """The resources available to the type loader, specified by the `required_resource_keys` argument of the decorator."""
        return super(DagsterTypeLoaderContext, self).resources

    @public
    @property
    def job_def(self) -> "JobDefinition":
        """The underlying job definition being executed."""
        return super(DagsterTypeLoaderContext, self).job_def

    @public
    @property
    def op_def(self) -> "OpDefinition":
        """The op for which type loading is occurring."""
        return super(DagsterTypeLoaderContext, self).op_def
