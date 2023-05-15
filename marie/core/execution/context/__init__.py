from abc import abstractmethod, ABC
from typing import Mapping, Optional, Any

from marie import check
from marie.core.execution.retries import RetryMode
from marie.logging.logger import MarieLogger
from marie.storage.marie_run import DagsterRun


class IJob:
    pass


class StepOutputHandle:
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
    def log(self) -> MarieLogger:
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
