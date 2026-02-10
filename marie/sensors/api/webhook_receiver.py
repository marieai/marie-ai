"""
Webhook receiver endpoint.

This endpoint receives HTTP webhooks and ingests them into the event_log.
The sensor daemon then processes these events during its evaluation loop.

Pattern: Write-before-ACK (event is durably stored before returning 202)
"""

import hashlib
import hmac
import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response
from pydantic import BaseModel

from marie.logging_core.logger import MarieLogger
from marie.sensors.exceptions import WebhookAuthError
from marie.sensors.state.psql_storage import PostgreSQLSensorStorage
from marie.sensors.types import SensorType

logger = MarieLogger("WebhookReceiver")

router = APIRouter(prefix="/webhooks", tags=["webhooks"])


# =============================================================================
# MODELS
# =============================================================================


class WebhookResponse(BaseModel):
    """Response after webhook ingestion."""

    event_id: str
    status: str = "accepted"
    message: str = "Event queued for processing"


# =============================================================================
# DEPENDENCY INJECTION
# =============================================================================


def get_storage() -> PostgreSQLSensorStorage:
    """Get storage instance."""
    return PostgreSQLSensorStorage.get_instance()


# =============================================================================
# AUTHENTICATION HELPERS
# =============================================================================


def verify_hmac_signature(
    body: bytes,
    signature: str,
    secret: str,
    algorithm: str = "sha256",
) -> bool:
    """
    Verify HMAC signature.

    Supports GitHub-style (sha256=xxx) and plain signatures.
    """
    if not signature or not secret:
        return False

    # Handle GitHub-style signatures (sha256=xxx)
    if "=" in signature:
        algo, sig_value = signature.split("=", 1)
        algorithm = algo
    else:
        sig_value = signature

    # Compute expected signature
    if algorithm == "sha256":
        expected = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    elif algorithm == "sha1":
        expected = hmac.new(secret.encode(), body, hashlib.sha1).hexdigest()
    else:
        return False

    return hmac.compare_digest(expected, sig_value)


def verify_api_key(provided_key: str, expected_key: str) -> bool:
    """Verify API key authentication."""
    if not provided_key or not expected_key:
        return False
    return hmac.compare_digest(provided_key, expected_key)


def verify_basic_auth(auth_header: str, expected_secret: str) -> bool:
    """Verify Basic authentication."""
    if not auth_header or not auth_header.startswith("Basic "):
        return False

    import base64

    try:
        encoded = auth_header.split(" ", 1)[1]
        decoded = base64.b64decode(encoded).decode()
        # Expected format: username:password (we just check the whole thing)
        return hmac.compare_digest(decoded, expected_secret)
    except Exception:
        return False


async def authenticate_webhook(
    request: Request,
    registration: Dict[str, Any],
) -> None:
    """
    Authenticate a webhook request.

    Raises WebhookAuthError if authentication fails.
    """
    auth_type = registration.get("auth_type", "none")
    auth_secret = registration.get("auth_secret")

    if auth_type == "none":
        return

    if not auth_secret:
        logger.warning(
            f"Webhook {registration.get('path')} has auth_type={auth_type} "
            "but no auth_secret configured"
        )
        return

    body = await request.body()

    if auth_type == "hmac":
        # Check various HMAC header names
        signature = (
            request.headers.get("x-hub-signature-256")
            or request.headers.get("x-hub-signature")
            or request.headers.get("x-signature")
            or request.headers.get("x-webhook-signature")
        )

        if not signature:
            raise WebhookAuthError("Missing signature header", registration.get("path"))

        if not verify_hmac_signature(body, signature, auth_secret):
            raise WebhookAuthError("Invalid signature", registration.get("path"))

    elif auth_type == "api_key":
        api_key = (
            request.headers.get("x-api-key")
            or request.headers.get("authorization", "").replace("Bearer ", "")
            or request.query_params.get("api_key")
        )

        if not verify_api_key(api_key, auth_secret):
            raise WebhookAuthError("Invalid API key", registration.get("path"))

    elif auth_type == "basic":
        auth_header = request.headers.get("authorization", "")

        if not verify_basic_auth(auth_header, auth_secret):
            raise WebhookAuthError("Invalid basic auth", registration.get("path"))


# =============================================================================
# WEBHOOK ENDPOINT
# =============================================================================


@router.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def receive_webhook(
    path: str,
    request: Request,
    storage: PostgreSQLSensorStorage = Depends(get_storage),
):
    """
    Public webhook endpoint.

    Receives webhooks at /webhooks/{path} and ingests them into event_log.

    The path is used to look up the webhook registration and determine
    which sensor should receive the event.

    Flow:
    1. Look up webhook registration by path
    2. Verify authentication
    3. Parse payload
    4. Write to event_log (durable)
    5. Return 202 Accepted

    The sensor daemon will process the event on its next evaluation cycle.
    """
    # Normalize path
    full_path = f"/{path}" if not path.startswith("/") else path

    # Look up webhook registration
    registration = await storage.get_webhook_by_path(full_path)

    if not registration:
        logger.debug(f"No webhook registration for path: {full_path}")
        raise HTTPException(
            status_code=404,
            detail=f"No webhook registered for path: {full_path}",
        )

    # Check HTTP method
    allowed_methods = registration.get("methods", ["POST"])
    if request.method not in allowed_methods:
        raise HTTPException(
            status_code=405,
            detail=f"Method {request.method} not allowed. Allowed: {allowed_methods}",
        )

    # Authenticate
    try:
        await authenticate_webhook(request, registration)
    except WebhookAuthError as e:
        logger.warning(f"Webhook auth failed for {full_path}: {e}")
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Parse payload
    content_type = request.headers.get("content-type", "")
    body = await request.body()

    if "application/json" in content_type:
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON payload")
    elif "application/x-www-form-urlencoded" in content_type:
        from urllib.parse import parse_qs

        payload = dict(parse_qs(body.decode()))
    else:
        # Store as raw string
        payload = {"raw": body.decode("utf-8", errors="replace")}

    # Extract headers (filter sensitive ones)
    sensitive_headers = {
        "authorization",
        "x-api-key",
        "cookie",
        "set-cookie",
        "x-webhook-secret",
    }
    headers = {
        k: v for k, v in request.headers.items() if k.lower() not in sensitive_headers
    }

    # Generate event_key if not provided
    event_key = None

    # Try to extract from common webhook ID headers
    event_key = (
        request.headers.get("x-github-delivery")
        or request.headers.get("x-request-id")
        or request.headers.get("x-correlation-id")
        or request.headers.get("x-webhook-id")
    )

    # If no ID header, generate from payload hash
    if not event_key:
        from marie.sensors.definitions.webhook_sensor import WebhookSensor

        event_key = WebhookSensor.generate_event_key(payload, full_path)

    # Get sensor external ID from registration
    sensor_external_id = registration.get("sensor_external_id")

    # Write to event_log (MUST complete before returning 202)
    event_id = await storage.insert_event(
        source="webhook",
        payload=payload,
        sensor_external_id=sensor_external_id,
        sensor_type=SensorType.WEBHOOK,
        routing_key=full_path,
        event_key=event_key,
        headers=headers,
    )

    logger.info(
        f"Webhook received: path={full_path}, event_id={event_id}, "
        f"sensor={sensor_external_id}"
    )

    return WebhookResponse(
        event_id=event_id,
        status="accepted",
        message="Event queued for processing",
    )


@router.api_route(
    "/test/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"]
)
async def test_webhook(
    path: str,
    request: Request,
    storage: PostgreSQLSensorStorage = Depends(get_storage),
):
    """
    Test webhook endpoint (dry run).

    Same as the main endpoint but doesn't write to event_log.
    Useful for testing webhook configuration and authentication.
    """
    full_path = f"/{path}" if not path.startswith("/") else path

    # Look up webhook registration
    registration = await storage.get_webhook_by_path(full_path)

    if not registration:
        raise HTTPException(
            status_code=404,
            detail=f"No webhook registered for path: {full_path}",
        )

    # Check HTTP method
    allowed_methods = registration.get("methods", ["POST"])
    if request.method not in allowed_methods:
        raise HTTPException(
            status_code=405,
            detail=f"Method {request.method} not allowed. Allowed: {allowed_methods}",
        )

    # Authenticate
    try:
        await authenticate_webhook(request, registration)
    except WebhookAuthError as e:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Parse payload
    content_type = request.headers.get("content-type", "")
    body = await request.body()

    if "application/json" in content_type:
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON payload")
    else:
        payload = {"raw": body.decode("utf-8", errors="replace")}

    return {
        "status": "valid",
        "message": "Webhook configuration is valid",
        "path": full_path,
        "method": request.method,
        "auth_type": registration.get("auth_type", "none"),
        "payload_preview": str(payload)[:200],
        "sensor_id": registration.get("sensor_id"),
    }


# =============================================================================
# REGISTRATION MANAGEMENT
# =============================================================================


@router.get("/registrations")
async def list_webhook_registrations(
    storage: PostgreSQLSensorStorage = Depends(get_storage),
):
    """List all webhook registrations. Admin only."""
    # TODO: Add admin authentication
    # For now, this is a placeholder that would need proper access control
    raise HTTPException(
        status_code=501,
        detail="Not implemented - requires admin authentication",
    )
