from enum import Enum

from marie import check
from marie._annotations import PublicAttr, public
from datetime import datetime
from enum import Enum
from typing import (
    TYPE_CHECKING,
    AbstractSet,
    Any,
    Dict,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Union,
    Self,
)

from marie.core.definitions.events import AssetKey
from marie.core.origin import JobPythonOrigin
from marie.core.storage.tags import ROOT_RUN_ID_TAG, PARENT_RUN_ID_TAG, REPOSITORY_LABEL_TAG, SCHEDULE_NAME_TAG, \
    SENSOR_NAME_TAG, BACKFILL_ID_TAG, RESUME_RETRY_TAG
from marie.core.utils import make_new_run_id


class DagsterRunStatus(Enum):
    """The status of run execution."""

    # Runs waiting to be launched by the Marie Daemon.
    QUEUED = "QUEUED"

    # Runs that have been launched, but execution has not yet started."""
    NOT_STARTED = "NOT_STARTED"

    # Runs that are managed outside of the Marie control plane.
    MANAGED = "MANAGED"

    # Runs that have been launched, but execution has not yet started.
    STARTING = "STARTING"

    # Runs that have been launched and execution has started.
    STARTED = "STARTED"

    # Runs that have successfully completed.
    SUCCESS = "SUCCESS"

    # Runs that have failed to complete.
    FAILURE = "FAILURE"

    # Runs that are in-progress and pending to be canceled.
    CANCELING = "CANCELING"

    # Runs that have been canceled before completion.
    CANCELED = "CANCELED"

    # Runs that have been terminated before completion.
    TERMINATED = "TERMINATED"


# These statuses that indicate a run may be using compute resources
IN_PROGRESS_RUN_STATUSES = [
    DagsterRunStatus.STARTING,
    DagsterRunStatus.STARTED,
    DagsterRunStatus.CANCELING,
]

# This serves as an explicit list of run statuses that indicate that the run is not using compute
# resources. This and the enum above should cover all run statuses.
NON_IN_PROGRESS_RUN_STATUSES = [
    DagsterRunStatus.QUEUED,
    DagsterRunStatus.NOT_STARTED,
    DagsterRunStatus.SUCCESS,
    DagsterRunStatus.FAILURE,
    DagsterRunStatus.MANAGED,
    DagsterRunStatus.CANCELED,
]

FINISHED_STATUSES = [
    DagsterRunStatus.SUCCESS,
    DagsterRunStatus.FAILURE,
    DagsterRunStatus.CANCELED,
]


class DagsterRunStatsSnapshot(
    NamedTuple(
        "_DagsterRunStatsSnapshot",
        [
            ("run_id", str),
            ("steps_succeeded", int),
            ("steps_failed", int),
            ("materializations", int),
            ("expectations", int),
            ("enqueued_time", Optional[float]),
            ("launch_time", Optional[float]),
            ("start_time", Optional[float]),
            ("end_time", Optional[float]),
        ],
    )
):
    def __new__(
            cls,
            run_id: str,
            steps_succeeded: int,
            steps_failed: int,
            materializations: int,
            expectations: int,
            enqueued_time: Optional[float],
            launch_time: Optional[float],
            start_time: Optional[float],
            end_time: Optional[float],
    ):
        return super(DagsterRunStatsSnapshot, cls).__new__(
            cls,
            run_id=check.str_param(run_id, "run_id"),
            steps_succeeded=check.int_param(steps_succeeded, "steps_succeeded"),
            steps_failed=check.int_param(steps_failed, "steps_failed"),
            materializations=check.int_param(materializations, "materializations"),
            expectations=check.int_param(expectations, "expectations"),
            enqueued_time=check.opt_float_param(enqueued_time, "enqueued_time"),
            launch_time=check.opt_float_param(launch_time, "launch_time"),
            start_time=check.opt_float_param(start_time, "start_time"),
            end_time=check.opt_float_param(end_time, "end_time"),
        )


class DagsterRun(
    NamedTuple(
        "_DagsterRun",
        [
            ("job_name", PublicAttr[str]),
            ("run_id", str),
            ("run_config", Mapping[str, object]),
            ("asset_selection", Optional[AbstractSet[AssetKey]]),
            ("op_selection", Optional[Sequence[str]]),
            ("resolved_op_selection", Optional[AbstractSet[str]]),
            ("step_keys_to_execute", Optional[Sequence[str]]),
            ("status", DagsterRunStatus),
            ("tags", Mapping[str, str]),
            ("root_run_id", Optional[str]),
            ("parent_run_id", Optional[str]),
            ("job_snapshot_id", Optional[str]),
            ("execution_plan_snapshot_id", Optional[str]),
            ("external_job_origin", Optional["ExternalJobOrigin"]),
            ("job_code_origin", Optional[JobPythonOrigin]),
            ("has_repository_load_data", bool),
        ],
    )
):
    """Serializable internal representation of a dagster run, as stored in a
    :py:class:`~dagster._core.storage.runs.RunStorage`.
    """

    def __new__(
            cls,
            job_name: str,
            run_id: Optional[str] = None,
            run_config: Optional[Mapping[str, object]] = None,
            asset_selection: Optional[AbstractSet[AssetKey]] = None,
            op_selection: Optional[Sequence[str]] = None,
            resolved_op_selection: Optional[AbstractSet[str]] = None,
            step_keys_to_execute: Optional[Sequence[str]] = None,
            status: Optional[DagsterRunStatus] = None,
            tags: Optional[Mapping[str, str]] = None,
            root_run_id: Optional[str] = None,
            parent_run_id: Optional[str] = None,
            job_snapshot_id: Optional[str] = None,
            execution_plan_snapshot_id: Optional[str] = None,
            external_job_origin: Optional["ExternalJobOrigin"] = None,
            job_code_origin: Optional[JobPythonOrigin] = None,
            has_repository_load_data: Optional[bool] = None,
    ):
        check.invariant(
            (root_run_id is not None and parent_run_id is not None)
            or (root_run_id is None and parent_run_id is None),
            (
                "Must set both root_run_id and parent_run_id when creating a PipelineRun that "
                "belongs to a run group"
            ),
        )
        # a set which contains the names of the ops to execute
        resolved_op_selection = check.opt_nullable_set_param(
            resolved_op_selection, "resolved_op_selection", of_type=str
        )
        # a list of op queries provided by the user
        # possible to be None when resolved_op_selection is set by the user directly
        op_selection = check.opt_nullable_sequence_param(op_selection, "op_selection", of_type=str)
        check.opt_nullable_sequence_param(step_keys_to_execute, "step_keys_to_execute", of_type=str)

        asset_selection = check.opt_nullable_set_param(
            asset_selection, "asset_selection", of_type=AssetKey
        )

        # Placing this with the other imports causes a cyclic import
        # https://github.com/dagster-io/dagster/issues/3181
        from marie.core.host_representation.origin import ExternalJobOrigin

        if status == DagsterRunStatus.QUEUED:
            check.inst_param(
                external_job_origin,
                "external_job_origin",
                ExternalJobOrigin,
                "external_job_origin is required for queued runs",
            )

        if run_id is None:
            run_id = make_new_run_id()

        return super(DagsterRun, cls).__new__(
            cls,
            job_name=check.str_param(job_name, "job_name"),
            run_id=check.str_param(run_id, "run_id"),
            run_config=check.opt_mapping_param(run_config, "run_config", key_type=str),
            op_selection=op_selection,
            asset_selection=asset_selection,
            resolved_op_selection=resolved_op_selection,
            step_keys_to_execute=step_keys_to_execute,
            status=check.opt_inst_param(
                status, "status", DagsterRunStatus, DagsterRunStatus.NOT_STARTED
            ),
            tags=check.opt_mapping_param(tags, "tags", key_type=str, value_type=str),
            root_run_id=check.opt_str_param(root_run_id, "root_run_id"),
            parent_run_id=check.opt_str_param(parent_run_id, "parent_run_id"),
            job_snapshot_id=check.opt_str_param(job_snapshot_id, "job_snapshot_id"),
            execution_plan_snapshot_id=check.opt_str_param(
                execution_plan_snapshot_id, "execution_plan_snapshot_id"
            ),
            external_job_origin=check.opt_inst_param(
                external_job_origin, "external_job_origin", ExternalJobOrigin
            ),
            job_code_origin=check.opt_inst_param(
                job_code_origin, "job_code_origin", JobPythonOrigin
            ),
            has_repository_load_data=check.opt_bool_param(
                has_repository_load_data, "has_repository_load_data", default=False
            ),
        )

    def with_status(self, status: DagsterRunStatus) -> Self:
        if status == DagsterRunStatus.QUEUED:
            # Placing this with the other imports causes a cyclic import
            # https://github.com/dagster-io/dagster/issues/3181
            from marie.core.host_representation.origin import ExternalJobOrigin

            check.inst(
                self.external_job_origin,
                ExternalJobOrigin,
                "external_pipeline_origin is required for queued runs",
            )

        return self._replace(status=status)

    def with_job_origin(self, origin: "ExternalJobOrigin") -> Self:
        from marie.core.host_representation.origin import ExternalJobOrigin

        check.inst_param(origin, "origin", ExternalJobOrigin)
        return self._replace(external_job_origin=origin)

    def with_tags(self, tags: Mapping[str, str]) -> Self:
        return self._replace(tags=tags)

    def get_root_run_id(self) -> Optional[str]:
        return self.tags.get(ROOT_RUN_ID_TAG)

    def get_parent_run_id(self) -> Optional[str]:
        return self.tags.get(PARENT_RUN_ID_TAG)

    def tags_for_storage(self) -> Mapping[str, str]:
        repository_tags = {}
        if self.external_job_origin:
            # tag the run with a label containing the repository name / location name, to allow for
            # per-repository filtering of runs from dagit.
            repository_tags[
                REPOSITORY_LABEL_TAG
            ] = self.external_job_origin.external_repository_origin.get_label()

        if not self.tags:
            return repository_tags

        return {**repository_tags, **self.tags}

    @public
    @property
    def is_finished(self) -> bool:
        return self.status in FINISHED_STATUSES

    @public
    @property
    def is_success(self) -> bool:
        return self.status == DagsterRunStatus.SUCCESS

    @public
    @property
    def is_failure(self) -> bool:
        return self.status == DagsterRunStatus.FAILURE

    @public
    @property
    def is_failure_or_canceled(self):
        return self.status == DagsterRunStatus.FAILURE or self.status == DagsterRunStatus.CANCELED

    @public
    @property
    def is_resume_retry(self) -> bool:
        return self.tags.get(RESUME_RETRY_TAG) == "true"

    @property
    def previous_run_id(self) -> Optional[str]:
        # Compat
        return self.parent_run_id

    @staticmethod
    def tags_for_schedule(schedule) -> Mapping[str, str]:
        return {SCHEDULE_NAME_TAG: schedule.name}

    @staticmethod
    def tags_for_sensor(sensor) -> Mapping[str, str]:
        return {SENSOR_NAME_TAG: sensor.name}

    @staticmethod
    def tags_for_backfill_id(backfill_id: str) -> Mapping[str, str]:
        return {BACKFILL_ID_TAG: backfill_id}


class RunsFilter(
    NamedTuple(
        "_RunsFilter",
        [
            ("run_ids", Sequence[str]),
            ("job_name", Optional[str]),
            ("statuses", Sequence[DagsterRunStatus]),
            ("tags", Mapping[str, Union[str, Sequence[str]]]),
            ("snapshot_id", Optional[str]),
            ("updated_after", Optional[datetime]),
            ("updated_before", Optional[datetime]),
            ("created_after", Optional[datetime]),
            ("created_before", Optional[datetime]),
        ],
    )
):
    """Defines a filter across job runs, for use when querying storage directly.

    Each field of the RunsFilter represents a logical AND with each other. For
    example, if you specify job_name and tags, then you will receive only runs
    with the specified job_name AND the specified tags. If left blank, then
    all values will be permitted for that field.

    Args:
        run_ids (Optional[List[str]]): A list of job run_id values.
        job_name (Optional[str]):
            Name of the job to query for. If blank, all job_names will be accepted.
        statuses (Optional[List[DagsterRunStatus]]):
            A list of run statuses to filter by. If blank, all run statuses will be allowed.
        tags (Optional[Dict[str, Union[str, List[str]]]]):
            A dictionary of run tags to query by. All tags specified here must be present for a given run to pass the filter.
        snapshot_id (Optional[str]): The ID of the job snapshot to query for. Intended for internal use.
        updated_after (Optional[DateTime]): Filter by runs that were last updated before this datetime.
        created_before (Optional[DateTime]): Filter by runs that were created before this datetime.

    """

    def __new__(
            cls,
            run_ids: Optional[Sequence[str]] = None,
            job_name: Optional[str] = None,
            statuses: Optional[Sequence[DagsterRunStatus]] = None,
            tags: Optional[Mapping[str, Union[str, Sequence[str]]]] = None,
            snapshot_id: Optional[str] = None,
            updated_after: Optional[datetime] = None,
            updated_before: Optional[datetime] = None,
            created_after: Optional[datetime] = None,
            created_before: Optional[datetime] = None,
    ):
        check.invariant(run_ids != [], "When filtering on run ids, a non-empty list must be used.")

        return super(RunsFilter, cls).__new__(
            cls,
            run_ids=check.opt_sequence_param(run_ids, "run_ids", of_type=str),
            job_name=check.opt_str_param(job_name, "job_name"),
            statuses=check.opt_sequence_param(statuses, "statuses", of_type=DagsterRunStatus),
            tags=check.opt_mapping_param(tags, "tags", key_type=str),
            snapshot_id=check.opt_str_param(snapshot_id, "snapshot_id"),
            updated_after=check.opt_inst_param(updated_after, "updated_after", datetime),
            updated_before=check.opt_inst_param(updated_before, "updated_before", datetime),
            created_after=check.opt_inst_param(created_after, "created_after", datetime),
            created_before=check.opt_inst_param(created_before, "created_before", datetime),
        )

    @staticmethod
    def for_schedule(schedule: "ExternalSchedule") -> "RunsFilter":
        return RunsFilter(tags=DagsterRun.tags_for_schedule(schedule))

    @staticmethod
    def for_sensor(sensor: "ExternalSensor") -> "RunsFilter":
        return RunsFilter(tags=DagsterRun.tags_for_sensor(sensor))

    @staticmethod
    def for_backfill(backfill_id: str) -> "RunsFilter":
        return RunsFilter(tags=DagsterRun.tags_for_backfill_id(backfill_id))


class JobBucket(NamedTuple):
    job_names: List[str]
    bucket_limit: Optional[int]


class TagBucket(NamedTuple):
    tag_key: str
    tag_values: List[str]
    bucket_limit: Optional[int]


class RunRecord(
    NamedTuple(
        "_RunRecord",
        [
            ("storage_id", int),
            ("dagster_run", DagsterRun),
            ("create_timestamp", datetime),
            ("update_timestamp", datetime),
            ("start_time", Optional[float]),
            ("end_time", Optional[float]),
        ],
    )
):
    """Internal representation of a run record, as stored in a
    :py:class:`~dagster._core.storage.runs.RunStorage`.

    Users should not invoke this class directly.
    """

    def __new__(
            cls,
            storage_id: int,
            dagster_run: DagsterRun,
            create_timestamp: datetime,
            update_timestamp: datetime,
            start_time: Optional[float] = None,
            end_time: Optional[float] = None,
    ):
        return super(RunRecord, cls).__new__(
            cls,
            storage_id=check.int_param(storage_id, "storage_id"),
            dagster_run=check.inst_param(dagster_run, "dagster_run", DagsterRun),
            create_timestamp=check.inst_param(create_timestamp, "create_timestamp", datetime),
            update_timestamp=check.inst_param(update_timestamp, "update_timestamp", datetime),
            # start_time and end_time fields will be populated once the run has started and ended, respectively, but will be None beforehand.
            start_time=check.opt_float_param(start_time, "start_time"),
            end_time=check.opt_float_param(end_time, "end_time"),
        )


class RunPartitionData(
    NamedTuple(
        "_RunPartitionData",
        [
            ("run_id", str),
            ("partition", str),
            ("status", DagsterRunStatus),
            ("start_time", Optional[float]),
            ("end_time", Optional[float]),
        ],
    )
):
    def __new__(
            cls,
            run_id: str,
            partition: str,
            status: DagsterRunStatus,
            start_time: Optional[float],
            end_time: Optional[float],
    ):
        return super(RunPartitionData, cls).__new__(
            cls,
            run_id=check.str_param(run_id, "run_id"),
            partition=check.str_param(partition, "partition"),
            status=check.inst_param(status, "status", DagsterRunStatus),
            start_time=check.opt_inst(start_time, float),
            end_time=check.opt_inst(end_time, float),
        )
