"""
Event sensor implementation.

Event sensors read events from the event_log that were ingested
from message queues (RabbitMQ, Kafka) or internal event sources.
"""

from typing import Any, Dict, List

from marie.sensors.context import SensorEvaluationContext
from marie.sensors.definitions.base import BaseSensor
from marie.sensors.exceptions import SensorConfigError
from marie.sensors.registry import register_sensor
from marie.sensors.types import RunRequest, SensorResult, SensorType


@register_sensor(SensorType.EVENT)
class EventSensor(BaseSensor):
    """
    Message queue event sensor.

    This sensor processes events that were ingested from message queues
    (RabbitMQ, Kafka) or internal event publishers. Events are first
    written to the durable event_log, then processed by this sensor.

    Configuration:
        provider: str - Message provider ("rabbitmq", "kafka", "internal")
        queue: str - Queue name (required for RabbitMQ)
        topic: str - Topic name (required for Kafka)
        exchange: str - Exchange name (RabbitMQ)
        routing_key: str - Routing key filter (RabbitMQ)
        events: list[str] - Event types to process (optional filter)
        group_by: str - Field to group events by for batching

    Cursor:
        event_log_id of the last processed event

    Run Key:
        event:{sensor_id}:{event_key}
    """

    sensor_type = SensorType.EVENT

    def __init__(self, sensor_data: Dict[str, Any]):
        super().__init__(sensor_data)
        self.provider: str = self.get_config_value("provider", "internal")
        self.queue: str = self.get_config_value("queue", "")
        self.topic: str = self.get_config_value("topic", "")
        self.exchange: str = self.get_config_value("exchange", "")
        self.routing_key: str = self.get_config_value("routing_key", "")
        self.event_types: List[str] = self.get_config_value("events", [])
        self.group_by: str = self.get_config_value("group_by", "")
        self.max_events_per_tick: int = self.get_config_value(
            "max_events_per_tick", 100
        )
        self.batch_mode: bool = self.get_config_value("batch_mode", False)

    async def evaluate(self, context: SensorEvaluationContext) -> SensorResult:
        """
        Evaluate the event sensor by checking for pending events.

        Reads events from event_log that match this sensor's configuration.
        """
        # Check if we have pending events from the context
        if not context.pending_events:
            return SensorResult.skip(
                "No pending events",
                cursor=context.cursor,
            )

        # Filter events by configured criteria
        matching_events = self._filter_events(context.pending_events)

        if not matching_events:
            # Even if filtered out, update cursor to latest event_log_id
            last_id = max(e.get("event_log_id", 0) for e in context.pending_events)
            return SensorResult.skip(
                "No events matched filter criteria",
                cursor=str(last_id) if last_id else context.cursor,
            )

        # Process events based on mode
        if self.batch_mode:
            return self._process_batch(matching_events, context)
        else:
            return self._process_individual(matching_events, context)

    def _filter_events(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter events based on sensor configuration."""
        filtered = []

        for event in events:
            # Check routing_key if configured
            if self.routing_key:
                event_routing_key = event.get("routing_key", "")
                if not self._matches_routing_key(event_routing_key):
                    continue

            # Check event_types if configured
            if self.event_types:
                payload = event.get("payload", {})
                event_type = payload.get("type") or payload.get("event_type")
                if event_type not in self.event_types:
                    continue

            filtered.append(event)

        return filtered[: self.max_events_per_tick]

    def _matches_routing_key(self, event_key: str) -> bool:
        """
        Check if event routing key matches configured pattern.

        Supports simple wildcard matching:
        - * matches exactly one word
        - # matches zero or more words
        """
        if not self.routing_key:
            return True

        # Simple equality check (no wildcards)
        if "*" not in self.routing_key and "#" not in self.routing_key:
            return event_key == self.routing_key

        # Wildcard matching (AMQP-style)
        pattern_parts = self.routing_key.split(".")
        event_parts = event_key.split(".")

        return self._match_routing_parts(pattern_parts, event_parts)

    def _match_routing_parts(
        self, pattern: List[str], event: List[str], pi: int = 0, ei: int = 0
    ) -> bool:
        """Recursive routing key pattern matching."""
        if pi == len(pattern) and ei == len(event):
            return True

        if pi >= len(pattern):
            return False

        if pattern[pi] == "#":
            # # matches zero or more words
            # Try matching zero words, then one, etc.
            for skip in range(len(event) - ei + 1):
                if self._match_routing_parts(pattern, event, pi + 1, ei + skip):
                    return True
            return False

        if ei >= len(event):
            return False

        if pattern[pi] == "*" or pattern[pi] == event[ei]:
            return self._match_routing_parts(pattern, event, pi + 1, ei + 1)

        return False

    def _process_individual(
        self, events: List[Dict[str, Any]], context: SensorEvaluationContext
    ) -> SensorResult:
        """Process events individually, one RunRequest per event."""
        run_requests = []
        last_event_log_id = context.cursor

        for event in events:
            event_key = event.get("event_key") or event.get("event_id")
            run_key = self.build_run_key("event", self.sensor_id, event_key)

            run_config = {
                "event_id": event.get("event_id"),
                "event_log_id": event.get("event_log_id"),
                "payload": event.get("payload", {}),
                "routing_key": event.get("routing_key"),
                "received_at": event.get("received_at"),
                "source": event.get("source", self.provider),
            }

            run_requests.append(
                RunRequest(
                    run_key=run_key,
                    job_name=self.target_job_name,
                    dag_id=self.target_dag_id,
                    run_config=run_config,
                    tags={
                        "trigger": "event",
                        "sensor_id": self.sensor_id,
                        "provider": self.provider,
                        "event_id": str(event.get("event_id")),
                    },
                )
            )

            event_log_id = event.get("event_log_id")
            if event_log_id:
                last_event_log_id = str(event_log_id)

        context.log_info(f"Processing {len(run_requests)} events")

        return SensorResult.fire_multiple(
            run_requests=run_requests,
            cursor=last_event_log_id,
        )

    def _process_batch(
        self, events: List[Dict[str, Any]], context: SensorEvaluationContext
    ) -> SensorResult:
        """
        Process events in batch mode, one RunRequest for all events.

        If group_by is configured, creates one RunRequest per group.
        """
        if self.group_by:
            return self._process_grouped(events, context)

        # Single batch for all events
        first_event = events[0]
        last_event = events[-1]

        # Create batch run key from first and last event
        batch_key = (
            f"{first_event.get('event_log_id')}-{last_event.get('event_log_id')}"
        )
        run_key = self.build_run_key("event-batch", self.sensor_id, batch_key)

        run_config = {
            "events": [
                {
                    "event_id": e.get("event_id"),
                    "event_log_id": e.get("event_log_id"),
                    "payload": e.get("payload", {}),
                    "routing_key": e.get("routing_key"),
                    "received_at": e.get("received_at"),
                }
                for e in events
            ],
            "event_count": len(events),
            "source": self.provider,
        }

        context.log_info(f"Processing batch of {len(events)} events")

        last_event_log_id = str(last_event.get("event_log_id"))

        return SensorResult.fire(
            run_key=run_key,
            job_name=self.target_job_name,
            dag_id=self.target_dag_id,
            run_config=run_config,
            tags={
                "trigger": "event-batch",
                "sensor_id": self.sensor_id,
                "provider": self.provider,
                "event_count": str(len(events)),
            },
            cursor=last_event_log_id,
        )

    def _process_grouped(
        self, events: List[Dict[str, Any]], context: SensorEvaluationContext
    ) -> SensorResult:
        """Process events grouped by the group_by field."""
        groups: Dict[str, List[Dict[str, Any]]] = {}

        for event in events:
            payload = event.get("payload", {})
            group_value = str(
                self._get_nested_value(payload, self.group_by) or "default"
            )

            if group_value not in groups:
                groups[group_value] = []
            groups[group_value].append(event)

        run_requests = []
        last_event_log_id = context.cursor

        for group_value, group_events in groups.items():
            first_event = group_events[0]
            last_event = group_events[-1]

            batch_key = f"{group_value}-{first_event.get('event_log_id')}"
            run_key = self.build_run_key("event-group", self.sensor_id, batch_key)

            run_config = {
                "group_key": self.group_by,
                "group_value": group_value,
                "events": [
                    {
                        "event_id": e.get("event_id"),
                        "event_log_id": e.get("event_log_id"),
                        "payload": e.get("payload", {}),
                    }
                    for e in group_events
                ],
                "event_count": len(group_events),
            }

            run_requests.append(
                RunRequest(
                    run_key=run_key,
                    job_name=self.target_job_name,
                    dag_id=self.target_dag_id,
                    run_config=run_config,
                    tags={
                        "trigger": "event-group",
                        "sensor_id": self.sensor_id,
                        "group_value": group_value,
                    },
                )
            )

            event_log_id = last_event.get("event_log_id")
            if event_log_id:
                last_event_log_id = str(event_log_id)

        context.log_info(f"Processing {len(events)} events in {len(groups)} groups")

        return SensorResult.fire_multiple(
            run_requests=run_requests,
            cursor=last_event_log_id,
        )

    def _get_nested_value(self, data: Dict[str, Any], key: str) -> Any:
        """Get a nested value using dot notation."""
        parts = key.split(".")
        value = data
        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            else:
                return None
        return value

    def validate_config(self) -> None:
        """Validate event sensor configuration."""
        valid_providers = ["rabbitmq", "kafka", "internal"]
        if self.provider not in valid_providers:
            raise SensorConfigError(
                f"Invalid provider '{self.provider}'. "
                f"Must be one of: {valid_providers}",
                field="provider",
            )

        if self.provider == "rabbitmq" and not self.queue:
            raise SensorConfigError(
                "RabbitMQ event sensor requires 'queue' configuration",
                field="queue",
            )

        if self.provider == "kafka" and not self.topic:
            raise SensorConfigError(
                "Kafka event sensor requires 'topic' configuration",
                field="topic",
            )
