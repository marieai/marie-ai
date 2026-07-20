"""Runtime invocation envelopes for the daemon's ``POST /v1/dispatch/invoke``.

Builds policy-compliant envelopes and signs them with HMAC-SHA256, matching the
daemon's Go verifier and the Studio signer byte-for-byte.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    from marie.plugins.agent_tool import PluginToolSpec

DEFAULT_RUNTIME_POLICY: dict[str, Any] = {
    "maxConcurrent": 1,
    "maxMemoryBytes": 536870912,
    "timeoutMs": 30000,
    "networkPolicy": "none",
}


def _normalize_for_signature(value: Any, omit_signature: bool = False) -> Any:
    """Mirror of the Studio signer's normalizeForSignature.

    Recursively sorts object keys, drops `None` (JS `undefined`) values, and omits
    the top-level `signature` key. Must match
    `extension-runtime-envelope.service.ts` byte-for-byte so the daemon verifies.
    """
    if isinstance(value, list):
        return [_normalize_for_signature(item) for item in value]
    if not isinstance(value, dict):
        return value
    out: dict[str, Any] = {}
    for key in sorted(value.keys()):
        if omit_signature and key == "signature":
            continue
        item = value[key]
        if item is not None:
            out[key] = _normalize_for_signature(item)
    return out


def canonical_runtime_envelope(envelope: dict[str, Any]) -> str:
    """JSON canonical form used as the HMAC input (matches the daemon + Studio)."""
    return json.dumps(
        _normalize_for_signature(envelope, omit_signature=True),
        separators=(",", ":"),
        ensure_ascii=False,
    )


def sign_envelope(
    envelope: dict[str, Any], *, key_id: str, secret: str
) -> dict[str, Any]:
    """Return a copy of `envelope` with an HMAC-SHA256 `signature` attached.

    Replicates `createHmac('sha256', secret).update(canonical).digest('base64url')`
    (base64url, no padding).
    """
    digest = hmac.new(
        secret.encode("utf-8"),
        canonical_runtime_envelope(envelope).encode("utf-8"),
        hashlib.sha256,
    ).digest()
    value = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    return {
        **envelope,
        "signature": {"keyId": key_id, "algorithm": "hmac-sha256", "value": value},
    }


def build_invocation_envelope(
    spec: "PluginToolSpec",
    *,
    payload: dict[str, Any],
    organization_id: str,
    workspace_id: str,
    user_id: str | None = None,
    action_type: str = "tool",
    action_id: str | None = None,
    credential_binding_ids: list[str] | None = None,
    request_id: str | None = None,
    trace_id: str | None = None,
    timeout_ms: int = 30000,
    ttl_seconds: int = 300,
    runtime_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a policy-compliant runtime envelope for ``/v1/dispatch/invoke``.

    Satisfies ``policy.VerifyRuntimeEnvelope``: package identity claims
    (``packageId``/``packageRef``/``packageDigest``), a valid ``actionType`` +
    ``actionId``, ``mode == "stub"``, a ``credentialBindingIds`` array, and a
    ``runtimePolicy`` with an allowed ``networkPolicy``. No signature is attached —
    pass the result through ``sign_envelope`` (or run the daemon with
    ``MARIE_PLUGIN_DAEMON_DEV_INSECURE``).
    """
    if not spec.package_ref or not spec.package_digest:
        raise ValueError(
            "plugin invocation requires package_ref and package_digest "
            "(hydrate them from the extension catalog)"
        )
    if not organization_id or not workspace_id:
        raise ValueError("plugin invocation requires organization_id and workspace_id")

    rp = dict(DEFAULT_RUNTIME_POLICY)
    rp.update(runtime_policy or {})
    rp.setdefault("networkPolicy", "none")
    rp["timeoutMs"] = timeout_ms

    resolved_action_id = (action_id or f"tools/{spec.tool_name}").strip()
    if not resolved_action_id:
        raise ValueError("plugin invocation requires an action_id")

    expires_at = (datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    return {
        "requestId": request_id or str(uuid4()),
        "traceId": trace_id or str(uuid4()),
        "organizationId": organization_id,
        "workspaceId": workspace_id,
        "userId": user_id or "",
        "installId": spec.install_id or "",
        "packageId": spec.package_id or "",
        "packageRef": spec.package_ref,
        "packageDigest": spec.package_digest,
        "packageTrustLevel": spec.package_trust_level or "community",
        "providerId": spec.provider_id or "",
        "actionId": resolved_action_id,
        "actionType": action_type,
        "credentialBindingIds": list(credential_binding_ids or []),
        "input": payload,
        # Opaque to the daemon; forwarded to the plugin subprocess as the request.
        "payload": payload,
        "runtimePolicy": rp,
        "expiresAt": expires_at,
        "nonce": str(uuid4()),
        "mode": "stub",
    }
