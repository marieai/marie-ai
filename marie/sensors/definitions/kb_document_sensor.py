"""Admit pending, authorized KB items through the existing sensor worker."""

import re
from typing import Any

from marie.kb.item_state import KbItemStore, item_run_parameters
from marie.sensors.context import SensorEvaluationContext
from marie.sensors.definitions.base import BaseSensor
from marie.sensors.definitions.data_sink.base import FileObject
from marie.sensors.registry import register_sensor
from marie.sensors.types import RunRequest, SensorResult, SensorType

KB_KEY_RE = re.compile(
    r"^tenants/(?P<tenant_id>[0-9a-fA-F-]{36})"
    r"/kb-indexes/(?P<index_id>[0-9a-fA-F-]{36})"
    r"/sources/(?P<source_id>[0-9a-fA-F-]{36})"
    r"/(?P<filename>.+)$"
)


@register_sensor(SensorType.DATA_SINK, subtype="kb_document")
class KbDocumentSensor(BaseSensor):
    owner_type = 'kb_document'

    def _store(self, context: SensorEvaluationContext) -> KbItemStore:
        pool = context.resources.get('postgres_pool')
        if pool is None:
            raise RuntimeError('Sensor PostgreSQL pool is not available')
        return KbItemStore(pool)

    def _request(self, item: dict[str, Any]) -> RunRequest:
        parameters = item_run_parameters(item)
        return RunRequest(
            run_key=f"kb_indexing:{parameters['ref_id']}",
            job_name=self.target_job_name or 'kb_indexing',
            run_config=parameters,
            tags={'trigger': self.owner_type, 'sensor_id': str(self.sensor_id)},
        )

    async def build_run_request(
        self, context: SensorEvaluationContext, obj: FileObject, bucket: str
    ) -> RunRequest | None:
        match = KB_KEY_RE.match(obj.key)
        if match is None:
            return None
        item = await self._store(context).pool.fetchrow(
            """SELECT * FROM marie_scheduler.kb_item_state
               WHERE tenant_id=$1::uuid AND index_id=$2::uuid AND source_id=$3
                 AND uri=$4 AND owner_type=$5 AND state='pending' AND admitted_at IS NULL
                 AND (input_ref->>'kind' IS DISTINCT FROM 'upstream' OR input_ref ? 'artifact')""",
            match.group('tenant_id'),
            match.group('index_id'),
            match.group('source_id'),
            f's3://{bucket}/{obj.key}',
            self.owner_type,
        )
        return self._request(item) if item else None

    async def evaluate(self, context: SensorEvaluationContext) -> SensorResult:
        limit = int(self.config.get('max_files_per_tick', 100))
        if not 1 <= limit <= 1000:
            raise ValueError('max_files_per_tick must be between 1 and 1000')
        items = await self._store(context).pending(self.owner_type, limit)
        if not items:
            return SensorResult.skip(
                'No registered KB items await admission', cursor=context.cursor
            )
        return SensorResult.fire_multiple(
            [self._request(item) for item in items], cursor=context.cursor
        )
