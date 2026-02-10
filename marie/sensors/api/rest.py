"""
REST API endpoints for sensor management.

These endpoints are called by marie-studio to sync trigger configurations
and manage sensor lifecycle.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from marie.sensors.config import SensorSettings
from marie.sensors.exceptions import SensorNotFoundError
from marie.sensors.state.psql_storage import PostgreSQLSensorStorage
from marie.sensors.types import SensorStatus, SensorType, TickStatus

router = APIRouter(prefix="/api/v1/sensors", tags=["sensors"])


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================


class SensorCreateRequest(BaseModel):
    """Create sensor from marie-studio trigger_config."""

    external_id: str = Field(..., description="trigger_config.id from marie_studio")
    name: str = Field(..., min_length=1, max_length=255)
    sensor_type: str = Field(..., description="Sensor type (schedule, webhook, etc.)")
    config: Dict[str, Any] = Field(default_factory=dict)
    target_job_name: Optional[str] = None
    target_dag_id: Optional[str] = None
    minimum_interval_seconds: int = Field(default=30, ge=10)


class SensorUpdateRequest(BaseModel):
    """Update sensor configuration."""

    name: Optional[str] = Field(None, min_length=1, max_length=255)
    config: Optional[Dict[str, Any]] = None
    target_job_name: Optional[str] = None
    target_dag_id: Optional[str] = None
    minimum_interval_seconds: Optional[int] = Field(None, ge=10)


class SensorResponse(BaseModel):
    """Sensor state response."""

    id: str
    external_id: str
    name: str
    sensor_type: str
    status: str
    config: Dict[str, Any]
    target_job_name: Optional[str]
    target_dag_id: Optional[str]
    cursor: Optional[str]
    last_tick_at: Optional[datetime]
    last_run_key: Optional[str]
    failure_count: int
    last_error: Optional[str]
    minimum_interval_seconds: int
    created_at: datetime
    updated_at: datetime


class SensorTickResponse(BaseModel):
    """Sensor tick (execution) response."""

    id: str
    sensor_id: str
    status: str
    cursor: Optional[str]
    run_ids: List[str]
    skip_reason: Optional[str]
    error_message: Optional[str]
    started_at: datetime
    completed_at: Optional[datetime]
    duration_ms: Optional[int]


class SensorListResponse(BaseModel):
    """Paginated sensor list response."""

    sensors: List[SensorResponse]
    total: int
    limit: int
    offset: int


class TickListResponse(BaseModel):
    """Paginated tick list response."""

    ticks: List[SensorTickResponse]
    total: int
    limit: int
    offset: int


class SensorTestRequest(BaseModel):
    """Request to test a sensor."""

    payload: Optional[Dict[str, Any]] = None
    dry_run: bool = Field(default=False, description="If true, don't submit jobs")


class SensorTestResponse(BaseModel):
    """Response from sensor test."""

    success: bool
    run_requests: List[Dict[str, Any]]
    skip_reason: Optional[str]
    cursor: Optional[str]
    message: str


class WebhookUrlResponse(BaseModel):
    """Webhook URL information."""

    path: str
    production_url: str
    test_url: str
    methods: List[str]
    auth_type: Optional[str]


class SensorMetricsResponse(BaseModel):
    """Aggregated sensor metrics."""

    sensor_id: str
    period_hours: int
    tick_counts: Dict[str, int]  # status -> count
    total_ticks: int
    avg_duration_ms: Optional[float]
    p95_duration_ms: Optional[float]
    total_jobs_triggered: int


# =============================================================================
# DEPENDENCY INJECTION
# =============================================================================


def get_storage() -> PostgreSQLSensorStorage:
    """Get storage instance. Override in tests."""
    return PostgreSQLSensorStorage.get_instance()


def get_settings() -> SensorSettings:
    """Get sensor settings."""
    return SensorSettings()


# =============================================================================
# SENSOR CRUD ENDPOINTS
# =============================================================================


@router.post("", response_model=SensorResponse, status_code=201)
async def create_sensor(
    request: SensorCreateRequest,
    storage: PostgreSQLSensorStorage = Depends(get_storage),
):
    """
    Create a new sensor from marie-studio trigger configuration.

    Called when a user creates a trigger in the UI.
    """
    # Validate sensor type
    try:
        SensorType(request.sensor_type)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid sensor_type: {request.sensor_type}",
        )

    # Check if external_id already exists
    existing = await storage.get_sensor_by_external_id(request.external_id)
    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"Sensor with external_id {request.external_id} already exists",
        )

    sensor_id = await storage.create_sensor(request.model_dump())

    sensor = await storage.get_sensor(sensor_id)
    return SensorResponse(**sensor)


@router.get("/{sensor_id}", response_model=SensorResponse)
async def get_sensor(
    sensor_id: str,
    storage: PostgreSQLSensorStorage = Depends(get_storage),
):
    """Get sensor by ID."""
    sensor = await storage.get_sensor(sensor_id)
    if not sensor:
        raise HTTPException(status_code=404, detail="Sensor not found")
    return SensorResponse(**sensor)


@router.get("/by-external/{external_id}", response_model=SensorResponse)
async def get_sensor_by_external_id(
    external_id: str,
    storage: PostgreSQLSensorStorage = Depends(get_storage),
):
    """Get sensor by external_id (marie_studio trigger_config.id)."""
    sensor = await storage.get_sensor_by_external_id(external_id)
    if not sensor:
        raise HTTPException(status_code=404, detail="Sensor not found")
    return SensorResponse(**sensor)


@router.patch("/{sensor_id}", response_model=SensorResponse)
async def update_sensor(
    sensor_id: str,
    request: SensorUpdateRequest,
    storage: PostgreSQLSensorStorage = Depends(get_storage),
):
    """Update sensor configuration."""
    updates = request.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=400, detail="No updates provided")

    sensor = await storage.update_sensor(sensor_id, updates)
    if not sensor:
        raise HTTPException(status_code=404, detail="Sensor not found")

    return SensorResponse(**sensor)


@router.delete("/{sensor_id}", status_code=204)
async def delete_sensor(
    sensor_id: str,
    storage: PostgreSQLSensorStorage = Depends(get_storage),
):
    """Delete sensor (called when trigger deleted in UI)."""
    deleted = await storage.delete_sensor(sensor_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Sensor not found")


# =============================================================================
# SENSOR OPERATIONS
# =============================================================================


@router.post("/{sensor_id}/activate", response_model=SensorResponse)
async def activate_sensor(
    sensor_id: str,
    storage: PostgreSQLSensorStorage = Depends(get_storage),
):
    """Activate sensor (start evaluating)."""
    updated = await storage.update_sensor_status(sensor_id, SensorStatus.ACTIVE)
    if not updated:
        raise HTTPException(status_code=404, detail="Sensor not found")

    sensor = await storage.get_sensor(sensor_id)
    return SensorResponse(**sensor)


@router.post("/{sensor_id}/deactivate", response_model=SensorResponse)
async def deactivate_sensor(
    sensor_id: str,
    storage: PostgreSQLSensorStorage = Depends(get_storage),
):
    """Deactivate sensor (stop evaluating)."""
    updated = await storage.update_sensor_status(sensor_id, SensorStatus.INACTIVE)
    if not updated:
        raise HTTPException(status_code=404, detail="Sensor not found")

    sensor = await storage.get_sensor(sensor_id)
    return SensorResponse(**sensor)


@router.post("/{sensor_id}/test", response_model=SensorTestResponse)
async def test_sensor(
    sensor_id: str,
    request: SensorTestRequest = None,
    storage: PostgreSQLSensorStorage = Depends(get_storage),
):
    """
    Manually trigger sensor for testing.

    If dry_run is True, evaluates the sensor but doesn't submit jobs.
    """
    from marie.sensors.context import SensorEvaluationContext
    from marie.sensors.registry import SensorRegistry

    sensor = await storage.get_sensor(sensor_id)
    if not sensor:
        raise HTTPException(status_code=404, detail="Sensor not found")

    sensor_type = SensorType(sensor.get("sensor_type"))
    cursor = await storage.get_cursor(sensor_id)

    # Build test context
    context = SensorEvaluationContext(
        sensor_id=sensor_id,
        sensor_name=sensor.get("name", ""),
        sensor_type=sensor_type,
        cursor=cursor,
        last_tick_at=sensor.get("last_tick_at"),
        config=sensor.get("config", {}),
        target_job_name=sensor.get("target_job_name"),
        target_dag_id=sensor.get("target_dag_id"),
    )

    # Add test payload if provided
    if request and request.payload:
        context.request_body = request.payload

    # Get evaluator and evaluate
    registry = SensorRegistry.get_instance()
    evaluator_class = registry.get_evaluator(sensor_type)
    evaluator = evaluator_class(sensor)

    try:
        result = await evaluator.evaluate(context)

        run_requests = [rr.to_dict() for rr in result.run_requests]
        skip_reason = result.skip_reason.message if result.skip_reason else None

        return SensorTestResponse(
            success=True,
            run_requests=run_requests,
            skip_reason=skip_reason,
            cursor=result.cursor,
            message=f"Evaluation completed: {len(run_requests)} run requests generated",
        )

    except Exception as e:
        return SensorTestResponse(
            success=False,
            run_requests=[],
            skip_reason=None,
            cursor=None,
            message=f"Evaluation failed: {str(e)}",
        )


@router.get("/{sensor_id}/webhook-url", response_model=WebhookUrlResponse)
async def get_webhook_url(
    sensor_id: str,
    storage: PostgreSQLSensorStorage = Depends(get_storage),
    settings: SensorSettings = Depends(get_settings),
):
    """Get webhook URL for webhook-type sensors."""
    sensor = await storage.get_sensor(sensor_id)
    if not sensor:
        raise HTTPException(status_code=404, detail="Sensor not found")

    if sensor.get("sensor_type") != SensorType.WEBHOOK.value:
        raise HTTPException(
            status_code=400,
            detail="Sensor is not a webhook type",
        )

    config = sensor.get("config", {})
    path = config.get("path", f"/sensors/{sensor_id}")
    methods = config.get("methods", ["POST"])
    auth_type = config.get("auth_type")

    # Build URLs from config (should be set by deployment)
    base_url = "https://api.example.com"  # TODO: Get from settings

    return WebhookUrlResponse(
        path=path,
        production_url=f"{base_url}/webhooks{path}",
        test_url=f"{base_url}/webhooks/test{path}",
        methods=methods,
        auth_type=auth_type,
    )


# =============================================================================
# HISTORY & METRICS
# =============================================================================


@router.get("/{sensor_id}/ticks", response_model=TickListResponse)
async def get_sensor_ticks(
    sensor_id: str,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    status: Optional[str] = None,
    storage: PostgreSQLSensorStorage = Depends(get_storage),
):
    """Get sensor execution history (ticks)."""
    # Verify sensor exists
    sensor = await storage.get_sensor(sensor_id)
    if not sensor:
        raise HTTPException(status_code=404, detail="Sensor not found")

    # Parse status filter
    tick_status = None
    if status:
        try:
            tick_status = TickStatus(status)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status: {status}",
            )

    ticks = await storage.get_ticks(
        sensor_id=sensor_id,
        limit=limit,
        offset=offset,
        status=tick_status,
    )

    # Get total count (TODO: Add count method to storage)
    total = len(ticks)  # Approximate for now

    return TickListResponse(
        ticks=[SensorTickResponse(**t) for t in ticks],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{sensor_id}/events")
async def get_sensor_events(
    sensor_id: str,
    limit: int = Query(100, ge=1, le=1000),
    cursor: Optional[str] = None,
    storage: PostgreSQLSensorStorage = Depends(get_storage),
):
    """Get events from event_log for this sensor."""
    sensor = await storage.get_sensor(sensor_id)
    if not sensor:
        raise HTTPException(status_code=404, detail="Sensor not found")

    external_id = sensor.get("external_id")
    events = await storage.get_pending_events(
        sensor_external_id=external_id,
        cursor=cursor,
        limit=limit,
    )

    return {
        "events": events,
        "count": len(events),
        "cursor": events[-1].get("event_log_id") if events else cursor,
    }


@router.get("/{sensor_id}/metrics", response_model=SensorMetricsResponse)
async def get_sensor_metrics(
    sensor_id: str,
    hours: int = Query(24, ge=1, le=168),
    storage: PostgreSQLSensorStorage = Depends(get_storage),
):
    """Get aggregated metrics for a sensor."""
    sensor = await storage.get_sensor(sensor_id)
    if not sensor:
        raise HTTPException(status_code=404, detail="Sensor not found")

    # Get recent ticks for metrics calculation
    ticks = await storage.get_ticks(sensor_id=sensor_id, limit=500)

    # Calculate metrics
    tick_counts = {"success": 0, "skipped": 0, "failed": 0, "started": 0}
    durations = []
    total_jobs = 0

    from datetime import timedelta, timezone

    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

    for tick in ticks:
        started_at = tick.get("started_at")
        if started_at and started_at.replace(tzinfo=timezone.utc) < cutoff:
            continue

        status = tick.get("status", "unknown")
        if status in tick_counts:
            tick_counts[status] += 1

        duration = tick.get("duration_ms")
        if duration:
            durations.append(duration)

        run_ids = tick.get("run_ids", [])
        total_jobs += len(run_ids)

    # Calculate duration percentiles
    avg_duration = sum(durations) / len(durations) if durations else None
    p95_duration = None
    if durations:
        sorted_durations = sorted(durations)
        p95_idx = int(len(sorted_durations) * 0.95)
        p95_duration = sorted_durations[min(p95_idx, len(sorted_durations) - 1)]

    return SensorMetricsResponse(
        sensor_id=sensor_id,
        period_hours=hours,
        tick_counts=tick_counts,
        total_ticks=sum(tick_counts.values()),
        avg_duration_ms=avg_duration,
        p95_duration_ms=p95_duration,
        total_jobs_triggered=total_jobs,
    )


# =============================================================================
# EVENT INGESTION
# =============================================================================


class EventIngestRequest(BaseModel):
    """Request to ingest an event."""

    source: str = Field(..., description="Event source (webhook, rabbitmq, etc.)")
    payload: Dict[str, Any]
    sensor_external_id: Optional[str] = None
    sensor_type: Optional[str] = None
    routing_key: Optional[str] = None
    event_key: Optional[str] = None
    headers: Optional[Dict[str, Any]] = None


class EventIngestResponse(BaseModel):
    """Response from event ingestion."""

    event_id: str
    event_log_id: Optional[int]


@router.post("/events", response_model=EventIngestResponse, status_code=202)
async def ingest_event(
    request: EventIngestRequest,
    storage: PostgreSQLSensorStorage = Depends(get_storage),
):
    """
    Internal event ingestion endpoint.

    Used to write events to the event_log from internal sources.
    External webhooks should use the webhook_receiver endpoint instead.
    """
    sensor_type = None
    if request.sensor_type:
        try:
            sensor_type = SensorType(request.sensor_type)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid sensor_type: {request.sensor_type}",
            )

    event_id = await storage.insert_event(
        source=request.source,
        payload=request.payload,
        sensor_external_id=request.sensor_external_id,
        sensor_type=sensor_type,
        routing_key=request.routing_key,
        event_key=request.event_key,
        headers=request.headers,
    )

    return EventIngestResponse(
        event_id=event_id,
        event_log_id=None,  # TODO: Return event_log_id from insert
    )
