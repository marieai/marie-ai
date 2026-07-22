"""KB document sensor: watches the KB S3 prefix and triggers kb_indexing runs.

Key pattern: tenants/{tenant_id}/kb-indexes/{index_id}/sources/{source_id}/{filename}
Run params come from the scheduler-owned resource workflow binding.
"""

import os
import re
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from uuid import UUID

from marie.sensors.context import SensorEvaluationContext
from marie.sensors.definitions.data_sink.base import FileObject
from marie.sensors.definitions.data_sink.s3_sensor import S3DataSinkSensor
from marie.sensors.registry import register_sensor
from marie.sensors.types import RunRequest, SensorResult, SensorType

if TYPE_CHECKING:
    from marie.storage.database.postgres_pool import AsyncPostgresConnectionPool

KB_KEY_RE = re.compile(
    r"^tenants/(?P<tenant_id>[0-9a-fA-F-]{36})"
    r"/kb-indexes/(?P<index_id>[0-9a-fA-F-]{36})"
    r"/sources/(?P<source_id>[0-9a-fA-F-]{36})"
    r"/(?P<filename>.+)$"
)


@register_sensor(SensorType.DATA_SINK, subtype="kb_document")
class KbDocumentSensor(S3DataSinkSensor):
    """Sensor for monitoring KB document uploads under `tenants/.../kb-indexes/...`.

    Unlike the generic S3 data sink, run params are not part of the sensor
    config: they are loaded from the scheduler's per-resource workflow
    bindings. Batch mode is not supported; each file always fires its own
    RunRequest.
    """

    def __init__(self, sensor_data: Dict[str, Any]):
        config = dict(sensor_data.get("config", {}) or {})
        if "bucket" not in config:
            # bucket is environment-specific (spec A2): fall back to env,
            # defaulting to "" so construction never fails on a missing
            # bucket - callers that actually list objects need it set.
            config["bucket"] = os.getenv("KB_SENSOR_BUCKET") or os.getenv(
                "S3_BUCKET", ""
            )
        sensor_data = {**sensor_data, "config": config}
        super().__init__(sensor_data)
        self.batch_mode = False

    async def load_binding(
        self, context: SensorEvaluationContext, index_id: str
    ) -> Optional[Dict[str, Any]]:
        """Read the scheduler-owned workflow binding for a KB index."""
        pool: Optional["AsyncPostgresConnectionPool"] = context.resources.get(
            "postgres_pool"
        )
        if pool is None:
            raise RuntimeError("Sensor PostgreSQL pool is not available")
        row = await pool.fetchrow(
            """
            SELECT workflow_name, run_params
            FROM marie_scheduler.resource_workflow_binding
            WHERE resource_type = 'kb_index' AND resource_id = $1
            """,
            UUID(index_id),
        )
        if row is None:
            return None
        return {
            "workflow_name": row["workflow_name"],
            "run_params": row["run_params"] or {},
        }

    async def build_run_request(
        self, context: SensorEvaluationContext, obj: FileObject, bucket: str
    ) -> Optional[RunRequest]:
        """Build the kb_indexing RunRequest for a KB file, or None to skip it."""
        m = KB_KEY_RE.match(obj.key)
        if not m:
            return None
        binding = await self.load_binding(context, m.group("index_id"))
        if binding is None:
            return None
        return self._build_run_request(obj, bucket, m, binding)

    def _build_run_request(
        self,
        obj: FileObject,
        bucket: str,
        m: "re.Match[str]",
        binding: Dict[str, Any],
    ) -> RunRequest:
        index_id = m.group("index_id")
        tenant_id = m.group("tenant_id")
        return RunRequest(
            run_key=f"kb_indexing:{index_id}:{obj.key}",
            job_name=binding["workflow_name"],
            run_config={
                "uri": f"s3://{bucket}/{obj.key}",
                "ref_id": obj.key,
                "ref_type": "kb_document",
                "project_id": tenant_id,
                "tenant_id": tenant_id,
                "index_id": index_id,
                "index_name": index_id,
                "source_id": m.group("source_id"),
                "run_params": binding["run_params"],
            },
            tags={
                "trigger": "kb_document",
                "sensor_id": str(self.sensor_id),
                "tenant_id": tenant_id,
            },
        )

    async def evaluate(self, context: SensorEvaluationContext) -> SensorResult:
        """Evaluate by listing new KB files and firing one RunRequest per file.

        Overrides the base data-sink evaluate() because run requests here are
        built from scheduler-owned workflow bindings, not the generic
        provider/bucket run_config shape, and objects outside the KB key
        pattern (or with no binding) are skipped rather than emitted.
        """
        cursor_timestamp: Optional[datetime] = None
        if context.cursor:
            try:
                cursor_timestamp = datetime.fromisoformat(context.cursor)
            except ValueError:
                context.log_warning(f"Invalid cursor format: {context.cursor}")

        try:
            objects = await self.list_objects(after_timestamp=cursor_timestamp)
        except Exception as e:
            context.log_error(f"Failed to list objects: {e}")
            from marie.sensors.exceptions import SensorEvaluationError

            raise SensorEvaluationError(
                f"Failed to list objects in {self.bucket}: {e}",
                sensor_id=self.sensor_id,
                cause=e,
            )

        objects = [obj for obj in objects if self.matches_patterns(obj.key)]
        if len(objects) > self.max_files_per_tick:
            objects = objects[: self.max_files_per_tick]
            context.log_info(
                f"Limited to {self.max_files_per_tick} files (more files available)"
            )

        if not objects:
            return SensorResult.skip("No new files detected", cursor=context.cursor)

        # Cursor semantics: the prefix watched here ('tenants/') also carries
        # submission uploads and other non-KB traffic, so a non-KB key (regex
        # no-match) is permanently unindexable by this sensor and the cursor
        # may advance past it - otherwise unrelated traffic would stall the
        # sensor forever. A KB key with no binding yet is recoverable (the
        # binding may be created moments later): the cursor must not advance
        # past it, or it would fall below the S3 listing filter (strict '<'
        # in _list_objects_sync) on the next tick and be lost forever. So if
        # any recoverable-skip objects exist, the new cursor stops at the
        # earliest of them; otherwise it advances to the latest listed object.
        run_requests: List[RunRequest] = []
        recoverable_skips: List[FileObject] = []
        bindings: Dict[str, Optional[Dict[str, Any]]] = {}
        for obj in objects:
            m = KB_KEY_RE.match(obj.key)
            if not m:
                continue
            index_id = m.group("index_id")
            if index_id not in bindings:
                bindings[index_id] = await self.load_binding(context, index_id)
            binding = bindings[index_id]
            if binding is None:
                context.log_warning(
                    f"No index binding for index_id={index_id} (key={obj.key}); "
                    "will retry until a binding is created"
                )
                recoverable_skips.append(obj)
                continue
            run_requests.append(self._build_run_request(obj, self.bucket, m, binding))

        if recoverable_skips:
            new_cursor = min(obj.last_modified for obj in recoverable_skips).isoformat()
        else:
            new_cursor = max(obj.last_modified for obj in objects).isoformat()

        if not run_requests:
            return SensorResult.skip(
                "No KB documents required indexing", cursor=new_cursor
            )

        context.log_info(
            f"Detected {len(run_requests)} KB document(s) for indexing in "
            f"{self.bucket}/{self.prefix}"
        )
        return SensorResult.fire_multiple(run_requests, cursor=new_cursor)
