import pytest

from marie.sensors.context import SensorEvaluationContext
from marie.sensors.definitions.event_sensor import EventSensor
from marie.sensors.types import SensorType


@pytest.mark.asyncio
async def test_extension_trigger_event_row_fires_event_sensor() -> None:
    sensor = EventSensor(
        {
            "id": "sensor-1",
            "external_id": "44444444-4444-4444-4444-444444444444",
            "name": "Gmail message sensor",
            "sensor_type": "event",
            "config": {
                "provider": "gmail",
                "events": ["gmail_message_added"],
                "routing_key": "gmail.gmail_message_added",
            },
            "target_job_name": "process_gmail_message",
            "target_dag_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        }
    )
    context = SensorEvaluationContext(
        sensor_id="sensor-1",
        sensor_name="Gmail message sensor",
        sensor_type=SensorType.EVENT,
        pending_events=[
            {
                "event_log_id": 42,
                "event_id": "event-1",
                "event_key": "delivery-1:gmail_message_added",
                "source": "extension_trigger",
                "routing_key": "gmail.gmail_message_added",
                "received_at": "2026-06-03T15:00:00Z",
                "payload": {
                    "type": "gmail_message_added",
                    "event_type": "gmail_message_added",
                    "provider": "gmail",
                    "provider_ref": "provider/gmail",
                    "delivery_id": "delivery-1",
                    "data": {
                        "messages": [{"id": "message-1", "subject": "Invoice received"}],
                    },
                },
            }
        ],
    )

    result = await sensor.evaluate(context)

    assert result.cursor == "42"
    assert len(result.run_requests) == 1
    run_request = result.run_requests[0]
    assert run_request.run_key == "event:sensor-1:delivery-1:gmail_message_added"
    assert run_request.job_name == "process_gmail_message"
    assert run_request.tags["trigger"] == "event"
    assert run_request.tags["provider"] == "gmail"
    assert run_request.run_config["source"] == "extension_trigger"
    assert run_request.run_config["payload"]["type"] == "gmail_message_added"
