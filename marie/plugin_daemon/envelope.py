"""Runtime invocation envelopes for the daemon's ``POST /v1/dispatch/invoke``.

Builds policy-compliant envelopes and signs them with HMAC-SHA256, matching the
daemon's Go verifier and the Studio signer byte-for-byte.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import math
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    from marie.plugin_daemon.invocation import PluginInvocationSpec

DEFAULT_RUNTIME_POLICY: dict[str, Any] = {
    "maxConcurrent": 1,
    "maxMemoryBytes": 536870912,
    "timeoutMs": 30000,
    "networkPolicy": "none",
}


def _normalize_for_signature(value: Any, omit_signature: bool = False) -> Any:
    """Mirror of the Studio signer's normalizeForSignature.

    Recursively sorts object keys, preserves JSON null, and omits
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
        out[key] = _normalize_for_signature(value[key])
    return out


def canonical_runtime_envelope(envelope: dict[str, Any]) -> str:
    """JSON canonical form used as the HMAC input (matches the daemon + Studio)."""
    return _canonical_json(_normalize_for_signature(envelope, omit_signature=True))


def _canonical_json(value: Any) -> str:
    if isinstance(value, dict):
        return (
            '{'
            + ','.join(
                _canonical_json(key) + ':' + _canonical_json(item)
                for key, item in value.items()
            )
            + '}'
        )
    if isinstance(value, list):
        return '[' + ','.join(_canonical_json(item) for item in value) + ']'
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError('Runtime envelopes require finite JSON numbers')
        if value == 0:
            return '-0' if math.copysign(1, value) < 0 else '0'
        text = repr(value)
        if 1e-6 <= abs(value) < 1e21:
            text = format(Decimal(text), 'f')
            return text.rstrip('0').rstrip('.') if '.' in text else text
        coefficient, exponent = text.split('e')
        return (
            coefficient.removesuffix('.0')
            + 'e'
            + ('+' if int(exponent) >= 0 else '-')
            + str(abs(int(exponent)))
        )
    return (
        json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
        .replace('\u2028', '\\u2028')
        .replace('\u2029', '\\u2029')
    )


def verify_envelope(envelope: dict[str, Any], *, key_id: str, secret: str) -> None:
    """Verify signed control-plane admission before resolving any credentials."""
    signature = envelope.get('signature') or {}
    if (
        not isinstance(signature, dict)
        or not key_id
        or not secret
        or signature.get('keyId') != key_id
        or signature.get('algorithm') != 'hmac-sha256'
    ):
        raise ValueError('Invalid plugin admission signature')
    expected = sign_envelope(envelope, key_id=key_id, secret=secret)['signature'][
        'value'
    ]
    actual = signature.get('value')
    if not isinstance(actual, str) or not hmac.compare_digest(expected, actual):
        raise ValueError('Invalid plugin admission signature')
    try:
        expires_at = datetime.fromisoformat(
            envelope['expiresAt'].replace('Z', '+00:00')
        )
        valid = expires_at > datetime.now(timezone.utc)
    except (KeyError, TypeError, ValueError, AttributeError):
        valid = False
    if not valid or not envelope.get('nonce'):
        raise ValueError('Plugin admission is expired or incomplete')
    policy = envelope.get('runtimePolicy')
    if (
        not isinstance(policy, dict)
        or not all(
            type(policy.get(key)) is int and policy[key] > 0
            for key in ('timeoutMs', 'maxConcurrent', 'maxMemoryBytes')
        )
        or policy.get('networkPolicy')
        not in {'none', 'manifest_declared', 'internal_only'}
    ):
        raise ValueError('Plugin admission requires a bounded runtime policy')


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
    spec: PluginInvocationSpec,
    *,
    payload: dict[str, Any],
    organization_id: str,
    workspace_id: str,
    user_id: str | None = None,
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
        "actionId": spec.action_id,
        "actionType": spec.action_type,
        "credentialBindingIds": list(spec.credential_binding_ids),
        "input": payload,
        # Opaque to the daemon; forwarded to the plugin subprocess as the request.
        "payload": payload,
        "runtimePolicy": rp,
        "expiresAt": expires_at,
        "nonce": str(uuid4()),
        "mode": "stub",
    }
