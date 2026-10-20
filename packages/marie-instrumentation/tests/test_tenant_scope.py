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
ORG2 = "22222222-2222-4222-8222-222222222222"
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


@pytest.mark.parametrize("workspace", [None, ""])
def test_organization_only_scope_overwrites_all_reserved_claims(
    workspace: str | None,
) -> None:
    scope = TelemetryScope(ORG) if workspace is None else TelemetryScope(ORG, workspace)
    with using_telemetry_scope(scope):
        snapshot = scoped_attributes(
            {
                "marie.organization_id": ORG2,
                "marie.workspace_id": WS1,
                "marie.project_id": WS2,
                "marie_organization_id": ORG2,
                "marie_workspace_id": WS1,
                "marie_project_id": WS2,
                "marie.identity_state": "platform",
                "marie.identity_version": 0,
                "safe": "value",
            }
        )
        with using_telemetry_scope(TelemetryScope(ORG, WS1)):
            assert current_telemetry_scope().workspace_id == WS1
        assert current_telemetry_scope() == scope
    assert current_telemetry_scope() is None
    assert snapshot == {
        "safe": "value",
        "marie.organization_id": ORG,
        "marie.workspace_id": "",
        "marie.project_id": "",
        "marie.identity_state": "verified",
        "marie.identity_version": 1,
    }


@pytest.mark.parametrize(
    "value", ["", "unverified", "00000000-0000-0000-0000-000000000000"]
)
def test_organization_only_scope_requires_valid_organization(value: str) -> None:
    with pytest.raises(ValueError):
        TelemetryScope(value)


@pytest.mark.parametrize(
    "workspace", ["unverified", "00000000-0000-0000-0000-000000000000"]
)
def test_organization_scope_rejects_invalid_workspace_association(
    workspace: str,
) -> None:
    with pytest.raises(ValueError):
        TelemetryScope(ORG, workspace)


def test_concurrent_organization_only_and_workspace_snapshots_remain_captured() -> None:
    async def item(
        organization: str, workspace: str = ""
    ) -> tuple[TelemetryScope, dict]:
        with using_telemetry_scope(TelemetryScope(organization, workspace)):
            scope = current_telemetry_scope()
            snapshot = scoped_attributes({"job.id": "duplicate"})
            await asyncio.sleep(0)
            assert current_telemetry_scope() == scope
            return scope, snapshot

    async def run() -> list[tuple[TelemetryScope, dict]]:
        return await asyncio.gather(item(ORG), item(ORG, WS1), item(ORG2))

    captured = asyncio.run(run())
    assert current_telemetry_scope() is None
    with using_telemetry_scope(TelemetryScope(ORG2, WS2)):
        assert [
            (snapshot["marie.organization_id"], snapshot["marie.workspace_id"])
            for _, snapshot in captured
        ] == [(ORG, ""), (ORG, WS1), (ORG2, "")]
        for scope, snapshot in captured:
            with using_telemetry_scope(scope):
                assert scoped_attributes({"job.id": "duplicate"}) == snapshot
        assert current_telemetry_scope().workspace_id == WS2
    assert current_telemetry_scope() is None


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
