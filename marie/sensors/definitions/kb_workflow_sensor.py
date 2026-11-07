"""Submit registered KB destinations after their upstream workflow completes."""

from marie.kb.item_state import KbItemStore, item_run_parameters
from marie.sensors.context import SensorEvaluationContext
from marie.sensors.definitions.run_status_sensor import RunStatusSensor
from marie.sensors.registry import register_sensor
from marie.sensors.types import RunRequest, SensorResult, SensorType


@register_sensor(SensorType.RUN_STATUS, subtype='kb_workflow')
class KbWorkflowSensor(RunStatusSensor):
    async def evaluate(self, context: SensorEvaluationContext) -> SensorResult:
        store = KbItemStore(context.resources['postgres_pool'].pool)
        requests = []
        limit = int(self.config.get('max_destinations_per_tick', 100))
        if not 1 <= limit <= 1000:
            raise ValueError('max_destinations_per_tick must be between 1 and 1000')
        for completion in self._filter_matching_jobs(context.completed_jobs):
            for item in await store.complete_upstream(
                completion, limit=limit - len(requests)
            ):
                parameters = item_run_parameters(item)
                source = (completion.get('data') or {}).get('metadata') or {}
                if source.get('fabric_group_id'):
                    parameters['fabric_group_id'] = source['fabric_group_id']
                requests.append(
                    RunRequest(
                        run_key=f"kb_indexing:{parameters['ref_id']}",
                        job_name=self.target_job_name or 'kb_indexing',
                        run_config=parameters,
                        tags={'trigger': 'run_status', 'sensor_id': self.sensor_id},
                    )
                )
            if len(requests) >= limit:
                break
        return (
            SensorResult.fire_multiple(requests, cursor=context.cursor)
            if requests
            else SensorResult.skip(
                'No registered KB destinations await this workflow output',
                cursor=context.cursor,
            )
        )
