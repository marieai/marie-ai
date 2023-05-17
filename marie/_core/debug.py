from typing import NamedTuple, Sequence

import marie.check as check
from marie._core.events.log import EventLogEntry
from marie._core.instance import DagsterInstance
from marie._core.snap import ExecutionPlanSnapshot, JobSnapshot
from marie.storage.dagster_run import DagsterRun
from marie._core import serialize_value, whitelist_for_serdes


class DebugRunPayload(
    NamedTuple(
        "_DebugRunPayload",
        [
            ("version", str),
            ("dagster_run", DagsterRun),
            ("event_list", Sequence[EventLogEntry]),
            ("job_snapshot", JobSnapshot),
            ("execution_plan_snapshot", ExecutionPlanSnapshot),
        ],
    )
):
    def __new__(
        cls,
        version: str,
        dagster_run: DagsterRun,
        event_list: Sequence[EventLogEntry],
        job_snapshot: JobSnapshot,
        execution_plan_snapshot: ExecutionPlanSnapshot,
    ):
        return super(DebugRunPayload, cls).__new__(
            cls,
            version=check.str_param(version, "version"),
            dagster_run=check.inst_param(dagster_run, "dagster_run", DagsterRun),
            event_list=check.sequence_param(event_list, "event_list", EventLogEntry),
            job_snapshot=check.inst_param(job_snapshot, "job_snapshot", JobSnapshot),
            execution_plan_snapshot=check.inst_param(
                execution_plan_snapshot,
                "execution_plan_snapshot",
                ExecutionPlanSnapshot,
            ),
        )

    @classmethod
    def build(cls, instance: DagsterInstance, run: DagsterRun) -> "DebugRunPayload":
        from marie import __version__ as marie_version

        return cls(
            version=marie_version,
            dagster_run=run,
            event_list=instance.all_logs(run.run_id),
            job_snapshot=instance.get_job_snapshot(run.job_snapshot_id),  # type: ignore  # (possible none)
            execution_plan_snapshot=instance.get_execution_plan_snapshot(
                run.execution_plan_snapshot_id  # type: ignore  # (possible none)
            ),
        )

    def write(self, output_file):
        return output_file.write(serialize_value(self).encode("utf-8"))
