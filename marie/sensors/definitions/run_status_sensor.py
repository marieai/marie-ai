"""
Run status sensor implementation.

Run status sensors monitor job completion and trigger follow-up jobs
based on the completion status of monitored jobs.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from marie.sensors.context import SensorEvaluationContext
from marie.sensors.definitions.base import BaseSensor
from marie.sensors.exceptions import SensorConfigError
from marie.sensors.registry import register_sensor
from marie.sensors.types import RunRequest, SensorResult, SensorType


@register_sensor(SensorType.RUN_STATUS)
class RunStatusSensor(BaseSensor):
    """
    Job completion monitoring sensor.

    This sensor watches for job completions and triggers follow-up jobs
    when monitored jobs reach specific terminal states.

    Configuration:
        monitored_job_names: list[str] - Job names to watch (optional)
        monitored_dag_ids: list[str] - DAG IDs to watch (optional)
        status_types: list[str] - Status types to trigger on:
            - "SUCCEEDED" / "completed"
            - "FAILED" / "failed"
            - "STOPPED" / "cancelled"
        require_all: bool - If True, wait for all monitored jobs
        pipeline_mode: bool - Use output of completed job as input

    Cursor:
        Timestamp of the last checked job completion

    Run Key:
        run_status:{sensor_id}:{completed_job_id}
    """

    sensor_type = SensorType.RUN_STATUS

    # Map from various status names to canonical form
    STATUS_MAPPING = {
        "SUCCEEDED": "completed",
        "succeeded": "completed",
        "completed": "completed",
        "COMPLETED": "completed",
        "FAILED": "failed",
        "failed": "failed",
        "STOPPED": "cancelled",
        "stopped": "cancelled",
        "cancelled": "cancelled",
        "CANCELLED": "cancelled",
        "expired": "expired",
        "EXPIRED": "expired",
        "skipped": "skipped",
        "SKIPPED": "skipped",
    }

    def __init__(self, sensor_data: Dict[str, Any]):
        super().__init__(sensor_data)
        self.monitored_job_names: List[str] = self.get_config_value(
            "monitored_job_names", []
        )
        self.monitored_dag_ids: List[str] = self.get_config_value(
            "monitored_dag_ids", []
        )
        self.status_types: List[str] = self.get_config_value(
            "status_types", ["completed"]
        )
        self.require_all: bool = self.get_config_value("require_all", False)
        self.pipeline_mode: bool = self.get_config_value("pipeline_mode", False)
        self.include_output: bool = self.get_config_value("include_output", True)

        # Normalize status types
        self.normalized_statuses: Set[str] = {
            self.STATUS_MAPPING.get(s, s) for s in self.status_types
        }

    async def evaluate(self, context: SensorEvaluationContext) -> SensorResult:
        """
        Evaluate the run status sensor.

        Checks for completed jobs that match the monitoring criteria.
        """
        # Get completed jobs from context
        completed_jobs = context.completed_jobs
        if not completed_jobs:
            return SensorResult.skip(
                "No job completions to process",
                cursor=context.cursor,
            )

        # Filter to matching jobs
        matching_jobs = self._filter_matching_jobs(completed_jobs)

        if not matching_jobs:
            # Update cursor even if no matches
            cursor = self._get_latest_cursor(completed_jobs, context.cursor)
            return SensorResult.skip(
                "No matching job completions",
                cursor=cursor,
            )

        # Handle require_all mode
        if self.require_all:
            return self._evaluate_require_all(matching_jobs, context)

        # Standard mode: trigger for each matching job
        return self._evaluate_individual(matching_jobs, context)

    def _filter_matching_jobs(self, jobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter jobs to those matching our criteria."""
        matching = []

        for job in jobs:
            # Check status
            job_status = job.get("state") or job.get("status")
            normalized_status = self.STATUS_MAPPING.get(job_status, job_status)

            if normalized_status not in self.normalized_statuses:
                continue

            # Check job name filter
            if self.monitored_job_names:
                job_name = job.get("name", "")
                if not any(
                    self._matches_pattern(job_name, pattern)
                    for pattern in self.monitored_job_names
                ):
                    continue

            # Check DAG ID filter
            if self.monitored_dag_ids:
                dag_id = job.get("dag_id", "")
                if dag_id not in self.monitored_dag_ids:
                    continue

            matching.append(job)

        return matching

    def _matches_pattern(self, name: str, pattern: str) -> bool:
        """
        Check if a job name matches a pattern.

        Supports simple wildcards:
        - * matches any characters
        """
        if "*" not in pattern:
            return name == pattern

        # Simple wildcard matching
        import fnmatch

        return fnmatch.fnmatch(name, pattern)

    def _evaluate_individual(
        self, jobs: List[Dict[str, Any]], context: SensorEvaluationContext
    ) -> SensorResult:
        """Create a RunRequest for each matching job."""
        run_requests = []
        latest_cursor = context.cursor

        for job in jobs:
            job_id = job.get("id")
            job_name = job.get("name", "unknown")
            job_status = job.get("state") or job.get("status")

            run_key = self.build_run_key("run_status", self.sensor_id, job_id)

            run_config = {
                "trigger_job_id": job_id,
                "trigger_job_name": job_name,
                "trigger_job_status": job_status,
                "trigger_dag_id": job.get("dag_id"),
            }

            # Include job output if configured
            if self.include_output:
                run_config["trigger_job_output"] = job.get("output", {})
                run_config["trigger_job_data"] = job.get("data", {})

            run_requests.append(
                RunRequest(
                    run_key=run_key,
                    job_name=self.target_job_name,
                    dag_id=self.target_dag_id,
                    run_config=run_config,
                    tags={
                        "trigger": "run_status",
                        "sensor_id": self.sensor_id,
                        "trigger_job_id": str(job_id),
                        "trigger_status": str(job_status),
                    },
                )
            )

            # Update cursor
            completed_at = job.get("completed_at") or job.get("updated_at")
            if completed_at:
                if isinstance(completed_at, datetime):
                    completed_at = completed_at.isoformat()
                latest_cursor = completed_at

        context.log_info(f"Triggering {len(run_requests)} jobs from run completions")

        return SensorResult.fire_multiple(
            run_requests=run_requests,
            cursor=latest_cursor,
        )

    def _evaluate_require_all(
        self, jobs: List[Dict[str, Any]], context: SensorEvaluationContext
    ) -> SensorResult:
        """
        Require all monitored jobs to be complete before triggering.

        This is useful for fan-in patterns where you want to wait for
        multiple jobs to finish before starting a downstream job.
        """
        # Track which patterns are satisfied
        satisfied_patterns: Set[str] = set()

        for job in jobs:
            job_name = job.get("name", "")

            for pattern in self.monitored_job_names:
                if self._matches_pattern(job_name, pattern):
                    satisfied_patterns.add(pattern)

            dag_id = job.get("dag_id", "")
            if dag_id in self.monitored_dag_ids:
                satisfied_patterns.add(f"dag:{dag_id}")

        # Check if all required patterns are satisfied
        required_patterns = set(self.monitored_job_names) | {
            f"dag:{d}" for d in self.monitored_dag_ids
        }

        if not required_patterns.issubset(satisfied_patterns):
            missing = required_patterns - satisfied_patterns
            return SensorResult.skip(
                f"Waiting for all monitored jobs. Missing: {missing}",
                cursor=context.cursor,
            )

        # All conditions met - create single run request
        context.log_info(f"All {len(required_patterns)} monitored jobs completed")

        # Create composite run key from all job IDs
        job_ids = sorted(j.get("id") for j in jobs)
        composite_key = "-".join(str(jid)[:8] for jid in job_ids[:5])
        run_key = self.build_run_key("run_status_all", self.sensor_id, composite_key)

        run_config = {
            "trigger_jobs": [
                {
                    "id": j.get("id"),
                    "name": j.get("name"),
                    "status": j.get("state") or j.get("status"),
                    "output": j.get("output", {}) if self.include_output else None,
                }
                for j in jobs
            ],
            "job_count": len(jobs),
        }

        cursor = self._get_latest_cursor(jobs, context.cursor)

        return SensorResult.fire(
            run_key=run_key,
            job_name=self.target_job_name,
            dag_id=self.target_dag_id,
            run_config=run_config,
            tags={
                "trigger": "run_status_all",
                "sensor_id": self.sensor_id,
                "job_count": str(len(jobs)),
            },
            cursor=cursor,
        )

    def _get_latest_cursor(
        self, jobs: List[Dict[str, Any]], current_cursor: Optional[str]
    ) -> str:
        """Get the latest completion timestamp as cursor."""
        latest = current_cursor

        for job in jobs:
            completed_at = job.get("completed_at") or job.get("updated_at")
            if completed_at:
                if isinstance(completed_at, datetime):
                    completed_at = completed_at.isoformat()
                if latest is None or completed_at > latest:
                    latest = completed_at

        return latest or datetime.now(timezone.utc).isoformat()

    def validate_config(self) -> None:
        """Validate run status sensor configuration."""
        if not self.monitored_job_names and not self.monitored_dag_ids:
            raise SensorConfigError(
                "Run status sensor requires either 'monitored_job_names' "
                "or 'monitored_dag_ids' configuration"
            )

        valid_statuses = set(self.STATUS_MAPPING.keys())
        for status in self.status_types:
            if status not in valid_statuses:
                raise SensorConfigError(
                    f"Invalid status_type '{status}'. "
                    f"Must be one of: {sorted(valid_statuses)}",
                    field="status_types",
                )
