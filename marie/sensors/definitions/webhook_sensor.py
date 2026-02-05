"""
Webhook sensor implementation.

Webhook sensors read events from the event_log that were ingested
via the webhook receiver endpoint. This follows the Dagster pattern
of durable event logging with pull-based processing.
"""

import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, List

from marie.sensors.context import SensorEvaluationContext
from marie.sensors.definitions.base import BaseSensor
from marie.sensors.registry import register_sensor
from marie.sensors.types import RunRequest, SensorResult, SensorType


@register_sensor(SensorType.WEBHOOK)
class WebhookSensor(BaseSensor):
    """
    Webhook-based sensor that reads from event_log.

    This sensor processes events that were ingested via HTTP webhooks.
    Events are first written to the durable event_log, then processed
    by this sensor during daemon evaluation.

    Configuration:
        path: str - Webhook path (e.g., "/webhooks/github-push")
        methods: list[str] - Allowed HTTP methods (default: ["POST"])
        auth_type: str - Authentication type ("none", "api_key", "hmac", "basic")
        filter: dict - Optional event filtering rules

    Cursor:
        event_log_id of the last processed event

    Run Key:
        webhook:{sensor_id}:{event_id} or webhook:{sensor_id}:{event_key}
    """

    sensor_type = SensorType.WEBHOOK

    def __init__(self, sensor_data: Dict[str, Any]):
        super().__init__(sensor_data)
        self.path: str = self.get_config_value("path", "")
        self.methods: List[str] = self.get_config_value("methods", ["POST"])
        self.auth_type: str = self.get_config_value("auth_type", "none")
        self.event_filter: Dict[str, Any] = self.get_config_value("filter", {})
        self.max_events_per_tick: int = self.get_config_value(
            "max_events_per_tick", 100
        )

    async def evaluate(self, context: SensorEvaluationContext) -> SensorResult:
        """
        Evaluate the webhook sensor by checking for pending events.

        Reads events from event_log where:
        - event_log_id > cursor
        - sensor_external_id matches this sensor
        """
        # Check if we have pending events from the context
        if not context.pending_events:
            return SensorResult.skip(
                "No pending webhook events",
                cursor=context.cursor,
            )

        run_requests = []
        last_event_log_id = context.cursor

        for event in context.pending_events[: self.max_events_per_tick]:
            # Apply any configured filters
            if not self._matches_filter(event):
                context.log_debug(f"Event {event.get('event_id')} filtered out")
                continue

            # Generate run key from event_key or event_id
            event_key = event.get("event_key") or event.get("event_id")
            run_key = self.build_run_key("webhook", self.sensor_id, event_key)

            # Build run config from event payload
            run_config = {
                "event_id": event.get("event_id"),
                "event_log_id": event.get("event_log_id"),
                "payload": event.get("payload", {}),
                "headers": event.get("headers", {}),
                "received_at": event.get("received_at"),
                "source": event.get("source", "webhook"),
            }

            run_requests.append(
                RunRequest(
                    run_key=run_key,
                    job_name=self.target_job_name,
                    dag_id=self.target_dag_id,
                    run_config=run_config,
                    tags={
                        "trigger": "webhook",
                        "sensor_id": self.sensor_id,
                        "event_id": str(event.get("event_id")),
                    },
                )
            )

            # Track the highest event_log_id we've seen
            event_log_id = event.get("event_log_id")
            if event_log_id:
                last_event_log_id = str(event_log_id)

        if not run_requests:
            return SensorResult.skip(
                "All events filtered out",
                cursor=last_event_log_id,
            )

        context.log_info(f"Processing {len(run_requests)} webhook events")

        return SensorResult.fire_multiple(
            run_requests=run_requests,
            cursor=last_event_log_id,
        )

    def _matches_filter(self, event: Dict[str, Any]) -> bool:
        """
        Check if an event matches the configured filter.

        Filter rules:
        - event_types: list of allowed event types in payload
        - source_ips: list of allowed source IPs (from headers)
        - payload_match: dict of key-value pairs that must match in payload
        """
        if not self.event_filter:
            return True

        payload = event.get("payload", {})
        headers = event.get("headers", {})

        # Check event_types filter
        event_types = self.event_filter.get("event_types")
        if event_types:
            event_type = payload.get("type") or payload.get("event_type")
            if event_type not in event_types:
                return False

        # Check source_ips filter
        source_ips = self.event_filter.get("source_ips")
        if source_ips:
            source_ip = headers.get("x-forwarded-for", "").split(",")[0].strip()
            if source_ip not in source_ips:
                return False

        # Check payload_match filter
        payload_match = self.event_filter.get("payload_match", {})
        for key, expected in payload_match.items():
            actual = self._get_nested_value(payload, key)
            if actual != expected:
                return False

        return True

    def _get_nested_value(self, data: Dict[str, Any], key: str) -> Any:
        """Get a nested value using dot notation (e.g., 'repository.name')."""
        parts = key.split(".")
        value = data
        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            else:
                return None
        return value

    def validate_config(self) -> None:
        """Validate webhook sensor configuration."""
        from marie.sensors.exceptions import SensorConfigError

        if not self.path:
            raise SensorConfigError(
                "Webhook sensor requires 'path' configuration", field="path"
            )

        # Validate auth_type
        valid_auth_types = ["none", "api_key", "hmac", "basic"]
        if self.auth_type not in valid_auth_types:
            raise SensorConfigError(
                f"Invalid auth_type '{self.auth_type}'. "
                f"Must be one of: {valid_auth_types}",
                field="auth_type",
            )

    @staticmethod
    def generate_event_key(
        payload: Dict[str, Any], routing_key: str, bucket_seconds: int = 60
    ) -> str:
        """
        Generate a stable event_key when the source doesn't provide one.

        Creates a hash from payload + routing_key + time bucket.
        """
        import json

        now = datetime.now(timezone.utc)
        bucket = int(now.timestamp() / bucket_seconds)

        key_data = json.dumps(payload, sort_keys=True) + routing_key + str(bucket)
        return hashlib.sha256(key_data.encode()).hexdigest()[:32]
