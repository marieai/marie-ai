from __future__ import annotations

import json
from pathlib import Path

import pytest

from marie.agent.tools.plugin_tool import PluginToolSpec
from marie.plugin_daemon import build_invocation_envelope, sign_envelope

# The marie daemon ships a signed fixture; signing it in Python must reproduce the
# exact value the daemon verifies (proves our canonicalization matches Go + Studio).
_DAEMON_FIXTURE = (
    Path(__file__).resolve().parents[4]
    / "packages"
    / "marie-plugin-daemon"
    / "internal"
    / "marie"
    / "auth"
    / "testdata"
    / "studio-signed-envelope.json"
)


@pytest.mark.skipif(not _DAEMON_FIXTURE.exists(), reason="daemon fixture not present")
def test_signing_matches_daemon_fixture_byte_for_byte():
    fixture = json.loads(_DAEMON_FIXTURE.read_text())
    expected = fixture["signature"]["value"]
    unsigned = {k: v for k, v in fixture.items() if k != "signature"}

    signed = sign_envelope(
        unsigned, key_id="marie-api-test", secret="test-runtime-secret"
    )
    assert signed["signature"]["value"] == expected
    assert signed["signature"]["algorithm"] == "hmac-sha256"

# Allowed action types (mirrors policy.allowedActionType in the marie daemon).
_ALLOWED_ACTION_TYPES = {
    "tool",
    "model",
    "datasource",
    "trigger",
    "endpoint",
    "mcp",
    "webapp",
    "stub",
}


def _spec(**overrides):
    base = {
        "source": "extension_tool",
        "slug": "jina_reader",
        "toolRef": "jina_reader",
        "installId": "inst-1",
        "providerId": "prov-1",
        "packageId": "pkg-1",
        "packageRef": "ext.m3forge.reader",
        "packageDigest": "sha256:abc",
    }
    base.update(overrides)
    return PluginToolSpec.model_validate(base)


def test_envelope_is_policy_compliant():
    env = build_invocation_envelope(
        _spec(),
        payload={"url": "https://example.com"},
        organization_id="org-1",
        workspace_id="ws-1",
        user_id="user-1",
    )

    # Identity / tenant claims required by authorizeEnvelope + policy.
    assert env["organizationId"] == "org-1"
    assert env["workspaceId"] == "ws-1"

    # Package trust-policy claims (policy.VerifyRuntimeEnvelope).
    assert env["packageId"] and env["packageRef"] and env["packageDigest"]
    assert env["packageTrustLevel"] != "blocked"

    # Action claims.
    assert env["actionType"] in _ALLOWED_ACTION_TYPES
    assert env["actionId"] == "tools/jina_reader"

    # Mode + credentials + network policy gates.
    assert env["mode"] == "stub"
    assert env["credentialBindingIds"] == []
    assert env["runtimePolicy"]["networkPolicy"] == "none"

    # Payload is forwarded opaquely to the plugin.
    assert env["payload"] == {"url": "https://example.com"}

    # No signature in Slice 3a (daemon runs with the dev-insecure bypass).
    assert "signature" not in env


def test_envelope_requires_package_ref_and_digest():
    # package_ref/digest are not in the dropped tool by default; must be hydrated.
    spec = _spec(packageRef=None, packageDigest=None)
    with pytest.raises(ValueError, match="package_ref and package_digest"):
        build_invocation_envelope(
            spec,
            payload={},
            organization_id="org-1",
            workspace_id="ws-1",
        )


def test_envelope_requires_org_and_workspace():
    with pytest.raises(ValueError, match="organization_id and workspace_id"):
        build_invocation_envelope(
            _spec(),
            payload={},
            organization_id="",
            workspace_id="ws-1",
        )
