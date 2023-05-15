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
from marie.core.utils import make_new_run_id


class MarieRunStatus(Enum):
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
    MarieRunStatus.STARTING,
    MarieRunStatus.STARTED,
    MarieRunStatus.CANCELING,
]

# This serves as an explicit list of run statuses that indicate that the run is not using compute
# resources. This and the enum above should cover all run statuses.
NON_IN_PROGRESS_RUN_STATUSES = [
    MarieRunStatus.QUEUED,
    MarieRunStatus.NOT_STARTED,
    MarieRunStatus.SUCCESS,
    MarieRunStatus.FAILURE,
    MarieRunStatus.MANAGED,
    MarieRunStatus.CANCELED,
]

FINISHED_STATUSES = [
    MarieRunStatus.SUCCESS,
    MarieRunStatus.FAILURE,
    MarieRunStatus.CANCELED,
]


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
            ("status", MarieRunStatus),
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
        status: Optional[MarieRunStatus] = None,
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
        op_selection = check.opt_nullable_sequence_param(
            op_selection, "op_selection", of_type=str
        )
        check.opt_nullable_sequence_param(
            step_keys_to_execute, "step_keys_to_execute", of_type=str
        )

        asset_selection = check.opt_nullable_set_param(
            asset_selection, "asset_selection", of_type=AssetKey
        )

        # Placing this with the other imports causes a cyclic import
        # https://github.com/dagster-io/dagster/issues/3181
        from marie.core.host_representation.origin import ExternalJobOrigin

        if status == MarieRunStatus.QUEUED:
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
                status, "status", MarieRunStatus, MarieRunStatus.NOT_STARTED
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

    def with_status(self, status: MarieRunStatus) -> Self:
        if status == MarieRunStatus.QUEUED:
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
        return self.status == MarieRunStatus.SUCCESS

    @public
    @property
    def is_failure(self) -> bool:
        return self.status == MarieRunStatus.FAILURE

    @public
    @property
    def is_failure_or_canceled(self):
        return (
            self.status == MarieRunStatus.FAILURE
            or self.status == MarieRunStatus.CANCELED
        )

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
