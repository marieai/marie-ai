import json
from dataclasses import asdict

import pytest
from grpc_health.v1.health_pb2 import HealthCheckResponse

from marie.state.state_store import DesiredDoc, StatusDoc


@pytest.mark.parametrize("encoded", [False, True])
@pytest.mark.parametrize(
    "document",
    [
        DesiredDoc("SCHEDULED", 7, {"job": "pending"}, "updated"),
        StatusDoc(1, "SERVING", "worker", 7, "updated", "heartbeat", {"busy": True}),
    ],
)
def test_unknown_document_fields_preserve_known_values(
    document: DesiredDoc | StatusDoc, encoded: bool
) -> None:
    payload = json.dumps({**asdict(document), "future_field": {"version": 2}})
    if encoded:
        payload = payload.encode()

    assert type(document).from_json(payload) == document


def test_desired_document_defaults_optional_metadata() -> None:
    first = DesiredDoc.from_json('{"phase": "SCHEDULED", "epoch": 7}')
    second = DesiredDoc.from_json('{"phase": "SCHEDULED", "epoch": 7}')

    assert first.phase == "SCHEDULED"
    assert first.epoch == 7
    assert first.params == {}
    assert first.updated_at == ""
    first.params["job"] = "new"
    assert second.params == {}


def test_status_document_defaults_optional_metadata() -> None:
    document = StatusDoc.from_json('{"status_code": 1, "owner": "worker", "epoch": 7}')

    assert document.status_code == HealthCheckResponse.SERVING
    assert document.status_name == "SERVING"
    assert document.owner == "worker"
    assert document.epoch == 7
    assert document.updated_at == ""
    assert document.heartbeat_at == ""
    assert document.details is None


@pytest.mark.parametrize(
    ("document_type", "payload", "required"),
    [
        (DesiredDoc, {"phase": "SCHEDULED", "epoch": 7}, "phase"),
        (DesiredDoc, {"phase": "SCHEDULED", "epoch": 7}, "epoch"),
        (StatusDoc, {"status_code": 1, "owner": "worker", "epoch": 7}, "owner"),
        (StatusDoc, {"status_code": 1, "owner": "worker", "epoch": 7}, "epoch"),
        (StatusDoc, {"status_code": 1, "owner": "worker", "epoch": 7}, "status_code"),
    ],
)
def test_document_hydration_does_not_invent_required_state(
    document_type: type[DesiredDoc] | type[StatusDoc], payload: dict, required: str
) -> None:
    payload.pop(required)

    with pytest.raises(TypeError):
        document_type.from_json(json.dumps(payload))
