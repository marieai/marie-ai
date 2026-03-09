"""Workflow Coordination API routes.

This module provides REST API endpoints for multi-agent workflow coordination,
including workflow management, status monitoring, and audit log queries.

Endpoints:
    POST /api/workflows                    - Start a new workflow
    GET  /api/workflows                    - List workflows
    GET  /api/workflows/{id}               - Get workflow status
    POST /api/workflows/{id}/cancel        - Cancel running workflow
    GET  /api/workflows/{id}/timeline      - Get execution timeline
    GET  /api/workflows/{id}/audit         - Query audit logs
    GET  /api/workflows/checkpoints        - List checkpoints
    POST /api/workflows/{id}/restore       - Restore from checkpoint
    GET  /api/workflows/sse                - SSE event stream
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from pydantic import BaseModel, Field

from marie.agent.coordination.state import AgentWorkflowStatus
from marie.logging_core.logger import MarieLogger

if TYPE_CHECKING:
    from marie.agent.base import BaseAgent
    from marie.agent.coordination import (
        AuditLogger,
        CheckpointStore,
        WorkflowCoordinator,
    )

logger = MarieLogger("marie.api.routes.workflow")


# -------------------------------------------------------------------------
# Request/Response Models
# -------------------------------------------------------------------------


class AgentStatusInfo(BaseModel):
    """Status information for a single agent in a workflow."""

    name: str
    status: str = "idle"
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_ms: Optional[float] = None
    output: Optional[str] = None
    error: Optional[str] = None


class AgentWorkflowInfo(BaseModel):
    """Summary information about an agent workflow."""

    workflow_id: str
    status: AgentWorkflowStatus
    goal: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    total_steps: int = 0
    current_agent: Optional[str] = None
    agents: List[str] = Field(default_factory=list)


class AgentWorkflowDetailResponse(BaseModel):
    """Detailed workflow information including agent status."""

    workflow_id: str
    status: AgentWorkflowStatus
    goal: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    total_steps: int = 0
    current_agent: Optional[str] = None
    agents: List[AgentStatusInfo] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StartAgentWorkflowRequest(BaseModel):
    """Request to start a new agent workflow."""

    goal: str = Field(..., description="Workflow objective")
    messages: List[Dict[str, Any]] = Field(
        default_factory=list, description="Initial messages"
    )
    initial_agent: Optional[str] = Field(
        default=None, description="Starting agent name"
    )
    agents: List[str] = Field(
        default_factory=list, description="Agent names to include"
    )
    config: Optional[Dict[str, Any]] = Field(
        default=None, description="Coordination config overrides"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Custom metadata"
    )


class StartAgentWorkflowResponse(BaseModel):
    """Response after starting an agent workflow."""

    workflow_id: str
    status: AgentWorkflowStatus
    message: str


class ListAgentWorkflowsResponse(BaseModel):
    """Response listing agent workflows."""

    workflows: List[AgentWorkflowInfo]
    total: int
    limit: int
    offset: int


class CancelAgentWorkflowResponse(BaseModel):
    """Response after cancelling an agent workflow."""

    workflow_id: str
    status: AgentWorkflowStatus
    message: str


class AgentWorkflowTimelineEvent(BaseModel):
    """A single event in the workflow timeline."""

    timestamp: datetime
    event_type: str
    agent_name: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)


class AgentWorkflowTimelineResponse(BaseModel):
    """Workflow execution timeline."""

    workflow_id: str
    events: List[AgentWorkflowTimelineEvent]
    total_duration_ms: Optional[float] = None


class AgentWorkflowAuditEntry(BaseModel):
    """A single audit log entry."""

    event_id: str
    timestamp: datetime
    event_type: str
    workflow_id: str
    agent_name: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)


class AgentWorkflowAuditResponse(BaseModel):
    """Audit log query response."""

    entries: List[AgentWorkflowAuditEntry]
    total: int
    limit: int
    offset: int


class AgentWorkflowCheckpointInfo(BaseModel):
    """Information about a workflow checkpoint."""

    workflow_id: str
    created_at: datetime
    updated_at: datetime
    status: AgentWorkflowStatus
    step_count: int


class ListAgentWorkflowCheckpointsResponse(BaseModel):
    """Response listing checkpoints."""

    checkpoints: List[AgentWorkflowCheckpointInfo]
    total: int


class RestoreAgentWorkflowCheckpointResponse(BaseModel):
    """Response after restoring from checkpoint."""

    workflow_id: str
    status: AgentWorkflowStatus
    restored_step: int
    message: str


# -------------------------------------------------------------------------
# SSE Event Models
# -------------------------------------------------------------------------


class AgentWorkflowSSEEvent(BaseModel):
    """Server-sent event for workflow updates."""

    event_type: str
    workflow_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    data: Dict[str, Any] = Field(default_factory=dict)

    def to_sse(self) -> str:
        """Format as SSE message."""
        import json

        return f"event: {self.event_type}\ndata: {json.dumps(self.model_dump(), default=str)}\n\n"


# -------------------------------------------------------------------------
# Workflow Router Class
# -------------------------------------------------------------------------


class AgentWorkflowRouter:
    """Router for agent workflow coordination API endpoints.

    Provides workflow management, monitoring, and audit capabilities.
    Designed to be mounted in a FastAPI application.

    Example:
        ```python
        from fastapi import FastAPI
        from marie.api.routes.workflow import create_agent_workflow_router

        app = FastAPI()
        workflow_router = create_agent_workflow_router(
            checkpoint_store=my_store,
            audit_logger=my_logger,
        )
        app.include_router(workflow_router, prefix="/api/agent-workflows")
        ```
    """

    def __init__(
        self,
        checkpoint_store: Optional["CheckpointStore"] = None,
        audit_logger: Optional["AuditLogger"] = None,
        agent_registry: Optional[Dict[str, "BaseAgent"]] = None,
    ):
        """Initialize workflow router.

        Args:
            checkpoint_store: Store for workflow checkpoints
            audit_logger: Logger for audit events
            agent_registry: Registry of available agents by name
        """
        self._checkpoint_store = checkpoint_store
        self._audit_logger = audit_logger
        self._agent_registry = agent_registry or {}
        self._active_workflows: Dict[str, "WorkflowCoordinator"] = {}
        self._event_subscribers: Dict[str, asyncio.Queue] = {}

    async def start_workflow(
        self,
        request: StartAgentWorkflowRequest,
        user_id: str,
    ) -> StartAgentWorkflowResponse:
        """Start a new workflow execution.

        Args:
            request: Workflow start parameters
            user_id: ID of the requesting user

        Returns:
            StartAgentWorkflowResponse with workflow ID and status
        """
        from marie.agent.config import CoordinationConfig
        from marie.agent.coordination import WorkflowCoordinator
        from marie.agent.message import Message

        workflow_id = f"wf-{uuid.uuid4().hex[:12]}"

        # Build config
        config_dict = request.config or {}
        config = CoordinationConfig(
            topology="workflow",
            max_steps=config_dict.get("max_steps", 20),
            max_retries_per_agent=config_dict.get("max_retries", 3),
            timeout=config_dict.get("timeout", 30.0),
        )

        # Create coordinator
        coordinator = WorkflowCoordinator(
            config,
            checkpoint_store=self._checkpoint_store,
            audit_logger=self._audit_logger,
        )

        # Add requested agents
        for agent_name in request.agents:
            if agent_name in self._agent_registry:
                coordinator.add_agent(self._agent_registry[agent_name])
            else:
                logger.warning(f"Agent '{agent_name}' not found in registry")

        # Store active workflow
        self._active_workflows[workflow_id] = coordinator

        # Build messages
        messages = [
            Message(**msg) if isinstance(msg, dict) else msg for msg in request.messages
        ] or [Message.user(request.goal)]

        # Start workflow in background
        asyncio.create_task(
            self._run_workflow(
                workflow_id=workflow_id,
                coordinator=coordinator,
                messages=messages,
                goal=request.goal,
                initial_agent=request.initial_agent,
                metadata=request.metadata,
            )
        )

        return StartAgentWorkflowResponse(
            workflow_id=workflow_id,
            status=AgentWorkflowStatus.RUNNING,
            message="Workflow started",
        )

    async def _run_workflow(
        self,
        workflow_id: str,
        coordinator: "WorkflowCoordinator",
        messages: List[Any],
        goal: str,
        initial_agent: Optional[str],
        metadata: Optional[Dict[str, Any]],
    ) -> None:
        """Execute workflow and broadcast events."""
        try:
            # Broadcast start event
            await self._broadcast_event(
                AgentWorkflowSSEEvent(
                    event_type="workflow_started",
                    workflow_id=workflow_id,
                    data={"goal": goal, "initial_agent": initial_agent},
                )
            )

            # Run workflow
            result = await coordinator.run(
                messages,
                workflow_id=workflow_id,
                goal=goal,
                initial_agent=initial_agent,
                **(metadata or {}),
            )

            # Broadcast completion
            await self._broadcast_event(
                AgentWorkflowSSEEvent(
                    event_type="workflow_completed",
                    workflow_id=workflow_id,
                    data={
                        "status": AgentWorkflowStatus.COMPLETED.value,
                        "total_steps": len(result.results),
                        "duration_ms": result.total_duration_ms,
                    },
                )
            )

        except Exception as e:
            logger.error(f"Workflow {workflow_id} failed: {e}")
            await self._broadcast_event(
                AgentWorkflowSSEEvent(
                    event_type="workflow_failed",
                    workflow_id=workflow_id,
                    data={"error": str(e)},
                )
            )

        finally:
            # Cleanup
            self._active_workflows.pop(workflow_id, None)

    async def get_workflow(
        self,
        workflow_id: str,
        user_id: str,
    ) -> AgentWorkflowDetailResponse:
        """Get detailed workflow status.

        Args:
            workflow_id: Workflow identifier
            user_id: ID of the requesting user

        Returns:
            AgentWorkflowDetailResponse with full status

        Raises:
            ValueError: If workflow not found
        """
        # Check active workflows first
        if workflow_id in self._active_workflows:
            coordinator = self._active_workflows[workflow_id]
            state = coordinator.state

            if state:
                agents_info = []
                for agent in coordinator.agents:
                    agent_info = AgentStatusInfo(
                        name=agent.name,
                        status=(
                            "completed"
                            if agent.name in state.step_history
                            else "pending"
                        ),
                    )
                    agents_info.append(agent_info)

                return AgentWorkflowDetailResponse(
                    workflow_id=state.workflow_id,
                    status=state.status,
                    goal=state.goal,
                    created_at=state.created_at,
                    updated_at=state.updated_at,
                    total_steps=state.step,
                    current_agent=state.active_agent,
                    agents=agents_info,
                    errors=state.errors,
                )

        # Check checkpoint store
        if self._checkpoint_store:
            state = await self._checkpoint_store.load(workflow_id)
            if state:
                return AgentWorkflowDetailResponse(
                    workflow_id=state.workflow_id,
                    status=state.status,
                    goal=state.goal,
                    created_at=state.created_at,
                    updated_at=state.updated_at,
                    total_steps=state.step,
                    current_agent=state.active_agent,
                    errors=state.errors,
                )

        raise ValueError(f"Workflow '{workflow_id}' not found")

    async def list_workflows(
        self,
        user_id: str,
        status: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> ListAgentWorkflowsResponse:
        """List workflows with optional filtering.

        Args:
            user_id: ID of the requesting user
            status: Optional status filter
            limit: Max results
            offset: Pagination offset

        Returns:
            ListAgentWorkflowsResponse with workflow list
        """
        workflows = []

        # Active workflows
        for wf_id, coordinator in self._active_workflows.items():
            state = coordinator.state
            if state:
                if status is None or state.status.value == status:
                    workflows.append(
                        AgentWorkflowInfo(
                            workflow_id=state.workflow_id,
                            status=state.status,
                            goal=state.goal,
                            created_at=state.created_at,
                            updated_at=state.updated_at,
                            total_steps=state.step,
                            current_agent=state.active_agent,
                            agents=state.step_history,
                        )
                    )

        # Checkpointed workflows
        if self._checkpoint_store:
            checkpoint_ids = await self._checkpoint_store.list_checkpoints()
            for cp_id in checkpoint_ids:
                if cp_id not in self._active_workflows:
                    state = await self._checkpoint_store.load(cp_id)
                    if state:
                        if status is None or state.status.value == status:
                            workflows.append(
                                AgentWorkflowInfo(
                                    workflow_id=state.workflow_id,
                                    status=state.status,
                                    goal=state.goal,
                                    created_at=state.created_at,
                                    updated_at=state.updated_at,
                                    total_steps=state.step,
                                    current_agent=state.active_agent,
                                    agents=state.step_history,
                                )
                            )

        # Sort by updated_at descending
        workflows.sort(key=lambda w: w.updated_at, reverse=True)

        # Apply pagination
        total = len(workflows)
        workflows = workflows[offset : offset + limit]

        return ListAgentWorkflowsResponse(
            workflows=workflows,
            total=total,
            limit=limit,
            offset=offset,
        )

    async def cancel_workflow(
        self,
        workflow_id: str,
        user_id: str,
    ) -> CancelAgentWorkflowResponse:
        """Cancel a running workflow.

        Args:
            workflow_id: Workflow identifier
            user_id: ID of the requesting user

        Returns:
            CancelAgentWorkflowResponse with result

        Raises:
            ValueError: If workflow not found or not running
        """
        if workflow_id not in self._active_workflows:
            raise ValueError(f"Workflow '{workflow_id}' not found or not running")

        coordinator = self._active_workflows[workflow_id]
        if coordinator.state:
            coordinator.state.cancel()

        # Remove from active
        del self._active_workflows[workflow_id]

        # Broadcast cancellation
        await self._broadcast_event(
            AgentWorkflowSSEEvent(
                event_type="workflow_cancelled",
                workflow_id=workflow_id,
                data={"cancelled_by": user_id},
            )
        )

        return CancelAgentWorkflowResponse(
            workflow_id=workflow_id,
            status=AgentWorkflowStatus.CANCELLED,
            message="Workflow cancelled",
        )

    async def get_timeline(
        self,
        workflow_id: str,
        user_id: str,
    ) -> AgentWorkflowTimelineResponse:
        """Get workflow execution timeline.

        Args:
            workflow_id: Workflow identifier
            user_id: ID of the requesting user

        Returns:
            AgentWorkflowTimelineResponse with event timeline
        """
        events = []

        # Query audit logs for timeline events
        if self._audit_logger:
            audit_events = await self._audit_logger.query(workflow_id=workflow_id)
            for ae in audit_events:
                events.append(
                    AgentWorkflowTimelineEvent(
                        timestamp=ae.timestamp,
                        event_type=ae.event_type.value,
                        agent_name=ae.agent_name,
                        details=ae.details or {},
                    )
                )

        # Sort by timestamp
        events.sort(key=lambda e: e.timestamp)

        # Calculate total duration
        total_duration_ms = None
        if len(events) >= 2:
            start = events[0].timestamp
            end = events[-1].timestamp
            total_duration_ms = (end - start).total_seconds() * 1000

        return AgentWorkflowTimelineResponse(
            workflow_id=workflow_id,
            events=events,
            total_duration_ms=total_duration_ms,
        )

    async def query_audit_logs(
        self,
        workflow_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> AgentWorkflowAuditResponse:
        """Query audit logs with filters.

        Args:
            workflow_id: Filter by workflow ID
            agent_name: Filter by agent name
            event_type: Filter by event type
            limit: Max results
            offset: Pagination offset

        Returns:
            AgentWorkflowAuditResponse with matching entries
        """
        entries = []

        if self._audit_logger:
            from marie.agent.coordination.audit import AuditEventType

            type_filter = None
            if event_type:
                try:
                    type_filter = AuditEventType(event_type)
                except ValueError:
                    pass

            audit_events = await self._audit_logger.query(
                workflow_id=workflow_id,
                agent_name=agent_name,
                event_type=type_filter,
                limit=limit + offset,  # Over-fetch for pagination
            )

            for ae in audit_events[offset:]:
                entries.append(
                    AgentWorkflowAuditEntry(
                        event_id=str(uuid.uuid4()),
                        timestamp=ae.timestamp,
                        event_type=ae.event_type.value,
                        workflow_id=ae.workflow_id,
                        agent_name=ae.agent_name,
                        details=ae.details or {},
                    )
                )

        return AgentWorkflowAuditResponse(
            entries=entries[:limit],
            total=len(entries),
            limit=limit,
            offset=offset,
        )

    async def list_checkpoints(
        self,
        prefix: Optional[str] = None,
    ) -> ListAgentWorkflowCheckpointsResponse:
        """List available checkpoints.

        Args:
            prefix: Optional prefix filter

        Returns:
            ListAgentWorkflowCheckpointsResponse with checkpoint list
        """
        checkpoints = []

        if self._checkpoint_store:
            checkpoint_ids = await self._checkpoint_store.list_checkpoints(prefix)
            for cp_id in checkpoint_ids:
                state = await self._checkpoint_store.load(cp_id)
                if state:
                    checkpoints.append(
                        AgentWorkflowCheckpointInfo(
                            workflow_id=state.workflow_id,
                            created_at=state.created_at,
                            updated_at=state.updated_at,
                            status=state.status,
                            step_count=state.step,
                        )
                    )

        return ListAgentWorkflowCheckpointsResponse(
            checkpoints=checkpoints,
            total=len(checkpoints),
        )

    async def restore_checkpoint(
        self,
        workflow_id: str,
        user_id: str,
    ) -> RestoreAgentWorkflowCheckpointResponse:
        """Restore workflow from checkpoint.

        Args:
            workflow_id: Workflow identifier
            user_id: ID of the requesting user

        Returns:
            RestoreAgentWorkflowCheckpointResponse with result

        Raises:
            ValueError: If checkpoint not found
        """
        if not self._checkpoint_store:
            raise ValueError("Checkpoint store not configured")

        state = await self._checkpoint_store.load(workflow_id)
        if not state:
            raise ValueError(f"Checkpoint '{workflow_id}' not found")

        return RestoreAgentWorkflowCheckpointResponse(
            workflow_id=workflow_id,
            status=state.status,
            restored_step=state.step,
            message=f"Checkpoint restored at step {state.step}",
        )

    # -------------------------------------------------------------------------
    # SSE Event Broadcasting
    # -------------------------------------------------------------------------

    async def subscribe(self, subscriber_id: str) -> asyncio.Queue:
        """Subscribe to workflow events.

        Args:
            subscriber_id: Unique subscriber identifier

        Returns:
            Queue for receiving events
        """
        queue: asyncio.Queue = asyncio.Queue()
        self._event_subscribers[subscriber_id] = queue
        return queue

    def unsubscribe(self, subscriber_id: str) -> None:
        """Unsubscribe from workflow events.

        Args:
            subscriber_id: Subscriber identifier
        """
        self._event_subscribers.pop(subscriber_id, None)

    async def _broadcast_event(self, event: AgentWorkflowSSEEvent) -> None:
        """Broadcast event to all subscribers.

        Args:
            event: Event to broadcast
        """
        for queue in self._event_subscribers.values():
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                pass  # Drop if queue is full


# -------------------------------------------------------------------------
# FastAPI Router Factory
# -------------------------------------------------------------------------


def create_agent_workflow_router(
    checkpoint_store: Optional["CheckpointStore"] = None,
    audit_logger: Optional["AuditLogger"] = None,
    agent_registry: Optional[Dict[str, Any]] = None,
    prefix: str = "",
):
    """Create FastAPI router for agent workflow coordination API.

    Args:
        checkpoint_store: Checkpoint store instance
        audit_logger: Audit logger instance
        agent_registry: Registry of available agents
        prefix: URL prefix for routes

    Returns:
        FastAPI APIRouter instance

    Example:
        ```python
        from fastapi import FastAPI
        from marie.api.routes.workflow import create_agent_workflow_router

        app = FastAPI()
        router = create_agent_workflow_router(
            checkpoint_store=my_store,
            audit_logger=my_logger,
        )
        app.include_router(router, prefix="/api/agent-workflows")
        ```
    """
    try:
        from fastapi import APIRouter, Depends, HTTPException, Query
        from fastapi.responses import StreamingResponse

        api_router = APIRouter(prefix=prefix, tags=["agent-workflows"])
        workflow_router = AgentWorkflowRouter(
            checkpoint_store=checkpoint_store,
            audit_logger=audit_logger,
            agent_registry=agent_registry,
        )

        async def get_current_user() -> str:
            return "default_user"

        @api_router.post("", response_model=StartAgentWorkflowResponse)
        async def start_workflow(
            request: StartAgentWorkflowRequest,
            user_id: str = Depends(get_current_user),
        ):
            """Start a new agent workflow execution."""
            return await workflow_router.start_workflow(request, user_id)

        @api_router.get("", response_model=ListAgentWorkflowsResponse)
        async def list_workflows(
            status: Optional[str] = Query(None),
            limit: int = Query(20, ge=1, le=100),
            offset: int = Query(0, ge=0),
            user_id: str = Depends(get_current_user),
        ):
            """List agent workflows with optional filtering."""
            return await workflow_router.list_workflows(
                user_id, status=status, limit=limit, offset=offset
            )

        @api_router.get(
            "/checkpoints", response_model=ListAgentWorkflowCheckpointsResponse
        )
        async def list_checkpoints(
            prefix: Optional[str] = Query(None),
        ):
            """List available workflow checkpoints."""
            return await workflow_router.list_checkpoints(prefix)

        @api_router.get("/audit", response_model=AgentWorkflowAuditResponse)
        async def query_audit_logs(
            workflow_id: Optional[str] = Query(None),
            agent_name: Optional[str] = Query(None),
            event_type: Optional[str] = Query(None),
            limit: int = Query(100, ge=1, le=1000),
            offset: int = Query(0, ge=0),
        ):
            """Query workflow audit logs."""
            return await workflow_router.query_audit_logs(
                workflow_id=workflow_id,
                agent_name=agent_name,
                event_type=event_type,
                limit=limit,
                offset=offset,
            )

        @api_router.get("/sse")
        async def sse_events(
            user_id: str = Depends(get_current_user),
        ):
            """Server-sent events stream for workflow updates."""
            subscriber_id = f"sse-{uuid.uuid4().hex[:8]}"

            async def event_generator():
                queue = await workflow_router.subscribe(subscriber_id)
                try:
                    # Send initial connection event
                    yield f"event: connected\ndata: {{}}\n\n"

                    while True:
                        try:
                            event = await asyncio.wait_for(queue.get(), timeout=30.0)
                            yield event.to_sse()
                        except asyncio.TimeoutError:
                            # Send keepalive
                            yield ": keepalive\n\n"
                finally:
                    workflow_router.unsubscribe(subscriber_id)

            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        @api_router.get("/{workflow_id}", response_model=AgentWorkflowDetailResponse)
        async def get_workflow(
            workflow_id: str,
            user_id: str = Depends(get_current_user),
        ):
            """Get detailed workflow status."""
            try:
                return await workflow_router.get_workflow(workflow_id, user_id)
            except ValueError as e:
                raise HTTPException(status_code=404, detail=str(e))

        @api_router.post(
            "/{workflow_id}/cancel", response_model=CancelAgentWorkflowResponse
        )
        async def cancel_workflow(
            workflow_id: str,
            user_id: str = Depends(get_current_user),
        ):
            """Cancel a running workflow."""
            try:
                return await workflow_router.cancel_workflow(workflow_id, user_id)
            except ValueError as e:
                raise HTTPException(status_code=404, detail=str(e))

        @api_router.get(
            "/{workflow_id}/timeline", response_model=AgentWorkflowTimelineResponse
        )
        async def get_timeline(
            workflow_id: str,
            user_id: str = Depends(get_current_user),
        ):
            """Get workflow execution timeline."""
            return await workflow_router.get_timeline(workflow_id, user_id)

        @api_router.post(
            "/{workflow_id}/restore",
            response_model=RestoreAgentWorkflowCheckpointResponse,
        )
        async def restore_checkpoint(
            workflow_id: str,
            user_id: str = Depends(get_current_user),
        ):
            """Restore workflow from checkpoint."""
            try:
                return await workflow_router.restore_checkpoint(workflow_id, user_id)
            except ValueError as e:
                raise HTTPException(status_code=404, detail=str(e))

        return api_router

    except ImportError:
        logger.warning("FastAPI not available, workflow router not created")
        return None


# Default router class for export
router = AgentWorkflowRouter
