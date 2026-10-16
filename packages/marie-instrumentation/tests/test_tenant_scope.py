from __future__ import annotations

import asyncio

import pytest
from marie.instrumentation.tenant_scope import (
    TelemetryScope,
    current_telemetry_scope,
    scoped_attributes,
    using_telemetry_scope,
)

ORG = "11111111-1111-4111-8111-111111111111"
WS1 = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaa1"
WS2 = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaa2"


def test_untrusted_attributes_cannot_claim_verified_ownership() -> None:
    attrs = scoped_attributes(
        {
            "marie.organization_id": ORG,
            "marie.workspace_id": WS1,
            "marie.project_id": WS2,
            "marie.identity_state": "verified",
            "marie.identity_version": 1,
            "safe": "value",
        }
    )
    assert attrs == {
        "safe": "value",
        "marie.organization_id": "",
        "marie.workspace_id": "",
        "marie.project_id": "",
        "marie.identity_state": "unattributed",
        "marie.identity_version": 0,
    }


def test_canonical_scope_overwrites_conflicting_legacy_alias() -> None:
    with using_telemetry_scope(TelemetryScope(ORG, WS1.upper())):
        attrs = scoped_attributes({"marie.workspace_id": WS2, "marie.project_id": WS2})
        assert attrs["marie.workspace_id"] == WS1
        assert attrs["marie.project_id"] == WS1
        assert attrs["marie.identity_state"] == "verified"
        with using_telemetry_scope(TelemetryScope(ORG, WS2)):
            assert current_telemetry_scope().workspace_id == WS2
        assert current_telemetry_scope().workspace_id == WS1
    assert current_telemetry_scope() is None


@pytest.mark.parametrize(
    "value", ["", "unverified", "00000000-0000-0000-0000-000000000000"]
)
def test_scope_rejects_invalid_identifiers(value: str) -> None:
    with pytest.raises(ValueError):
        TelemetryScope(value, WS1)


def test_concurrent_items_and_captured_queue_snapshot() -> None:
    async def item(workspace: str) -> dict:
        with using_telemetry_scope(TelemetryScope(ORG, workspace)):
            await asyncio.sleep(0)
            return scoped_attributes({"job.id": "duplicate"})

    async def run() -> list[dict]:
        return await asyncio.gather(item(WS1), item(WS2))

    snapshots = asyncio.run(run())
    assert [row["marie.workspace_id"] for row in snapshots] == [WS1, WS2]
    assert current_telemetry_scope() is None
    assert snapshots[0]["marie.workspace_id"] == WS1
