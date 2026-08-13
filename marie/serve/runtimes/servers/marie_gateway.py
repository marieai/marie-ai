import asyncio
import importlib
import json
import os
import socket
import sys
import time
import traceback
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, AsyncGenerator, AsyncIterator, Dict, Literal, Optional
from urllib.parse import urlparse

import grpc
from docarray import DocList
from docarray.documents import TextDoc
from fastapi import FastAPI, Request
from grpc_health.v1.health_pb2 import HealthCheckResponse
from marie.engine.llm_queue.registry import (
    dispatch_runtime_live_state,
    dispatch_runtime_snapshot,
)
from rich.traceback import install

import marie.helper
from marie._core.definitions.metadata import JsonMetadataValue
from marie.auth.api_key_manager import APIKeyManager
from marie.auth.auth_bearer import TokenBearer
from marie.constants import (
    __cache_path__,
    __config_dir__,
    __marie_home__,
    __model_path__,
)
from marie.excepts import BadConfigSource, RuntimeFailToStart
from marie.helper import get_or_reuse_loop
from marie.jaml import JAML
from marie.job.gateway_job_distributor import GatewayJobDistributor
from marie.job.job_manager import JobManager
from marie.kb.gateway_routes import register_kb_routes
from marie.logging_core.predefined import default_logger as logger
from marie.messaging import Toast, mark_as_accepted, mark_as_failed
from marie.messaging.events import (
    EngineEventData,
    EventMessage,
    MarieEvent,
    MarieEventType,
)
from marie.messaging.grpc_event_broker import GrpcEventBroker
from marie.proto import jina_pb2, jina_pb2_grpc
from marie.sandbox.blueprints.gateway_routes import register_blueprint_routes
from marie.scheduler import PostgreSQLJobScheduler
from marie.scheduler.models import DEFAULT_RETRY_POLICY, JobSubmissionModel, WorkInfo
from marie.scheduler.state import WorkState
from marie.serve.discovery import JsonAddress
from marie.serve.discovery.etcd_manager import (
    close_etcd_client,
    convert_to_etcd_args,
    get_etcd_client,
)
from marie.serve.discovery.registry import _is_known_connection_error
from marie.serve.discovery.resolver import EtcdServiceResolver
from marie.serve.instrumentation import MetricsTimer
from marie.serve.networking.balancer.load_balancer import LoadBalancerType
from marie.serve.networking.utils import get_grpc_channel
from marie.serve.runtimes.gateway.marie.llm_dispatch_runtime import (
    GatewayLlmDispatchRuntime,
)
from marie.serve.runtimes.gateway.request_handling import GatewayRequestHandler
from marie.serve.runtimes.gateway.streamer import GatewayStreamer
from marie.serve.runtimes.servers.cluster_state import ClusterState
from marie.serve.runtimes.servers.composite import CompositeServer
from marie.serve.runtimes.servers.grpc import GRPCServer
from marie.serve.runtimes.servers.wasm_routes import register_wasm_routes
from marie.state.base import is_stale
from marie.state.semaphore_store import SemaphoreStore
from marie.state.slot_capacity_manager import SlotCapacityManager
from marie.state.state_store import (
    DesiredDoc,
    DesiredStore,
    StatusDoc,
    StatusStore,
    _now_iso,
    _status_code,
    _status_name,
)
from marie.storage.kv.psql import PostgreSQLKV
from marie.types_core.request.data import DataRequest, Response
from marie.types_core.request.status import StatusMessage
from marie.utils.scheduler_trace import scheduler_trace
from marie.utils.server_runtime import (
    attach_sensor_worker_scheduler,
    setup_auth,
    setup_llm_tracking,
    setup_sensor_worker,
    setup_storage,
    setup_toast_events,
)
from marie.utils.types import strtobool
from marie.utils.utils import current_milli_time

ROOT = "deployments/"

HEARTBEAT_INTERVAL_S = 10  # worker -> status.heartbeat_at
HEARTBEAT_TIMEOUT_S = 3 * HEARTBEAT_INTERVAL_S  # server considers dead
RESCHEDULE_BACKOFF_S = 5  # server backoff before bumping epoch again

CLAIM_TIMEOUT_S = 30  # how long you wait for a claim per epoch
MAX_MISSES = 3  # after 5 failed epochs -> GC
MAX_AGE_S = 30 * 60  # hard age cap (30 min)
STATUS_DEGRADED_SINCE = "status_degraded_since"
STATUS_DEGRADED_REASON = "status_degraded_reason"
STATUS_DEGRADED_LIVE_MISSING = "live_node_missing_status"
SERVICE_SNAPSHOT_COMPLETE = "snapshot_complete"
DEFAULT_SERVICE_EVENT_WORKERS = 32
CAPACITY_INFO_LOG_INTERVAL_SECONDS = 10.0


class EventKind(str, Enum):
    SERVICE = "SERVICE"
    DESIRED = "DESIRED"
    STATUS = "STATUS"


@dataclass
class ServiceEvent:
    kind: EventKind  # EventKind.SERVICE
    service: str  # resolver’s service name
    ev_type: str  # "put" | "delete" | "snapshot_complete"
    value: dict | None
    key: str  # raw key (so you can log/debug)


@dataclass
class StateEvent:
    kind: EventKind  # EventKind.DESIRED or EventKind.STATUS
    node: str
    deployment: str
    ev_type: str  # "put" | "delete"
    value: dict | None
    key: str  # raw key for debugging


def load_env_file(dotenv_path: Optional[str] = None) -> None:
    from dotenv import load_dotenv

    logger.info(f"Loading env file from {dotenv_path}")
    load_dotenv(dotenv_path=dotenv_path, verbose=True)


def _is_desired_key(key: str) -> bool:
    return key.startswith(ROOT) and key.endswith("/desired")


def _is_status_key(key: str) -> bool:
    return key.startswith(ROOT) and key.endswith("/status")


def handle_exception(exc_type, exc_value, exc_traceback):
    """
    Handle uncaught exceptions
    :param exc_type:
    :param exc_value:
    :param exc_traceback:
    """
    logger.error("exc_type", exc_type)
    logger.error("exc_value", exc_value)
    logger.error("exc_traceback", exc_traceback)
    traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stdout)


def _netloc(addr: str) -> str:
    """
    Accepts 'grpc://host:port' or 'host:port' and returns 'host:port'.
    """
    if "://" in addr:
        p = urlparse(addr)
        return p.netloc or addr
    return addr


def _llm_queue_runtime_config(args: dict[str, Any]) -> dict[str, Any]:
    config = dict(args.get("llm_queue") or {})
    scheduler_config = config.get("scheduler")
    if _has_llm_scheduler_repository_config(scheduler_config):
        return config

    job_scheduler_kwargs = args.get("job_scheduler_kwargs")
    if not isinstance(job_scheduler_kwargs, dict):
        return config

    scheduler = dict(scheduler_config) if isinstance(scheduler_config, dict) else {}
    scheduler["psql"] = dict(job_scheduler_kwargs)
    config["scheduler"] = scheduler
    return config


def _has_llm_scheduler_repository_config(scheduler_config: Any) -> bool:
    if not isinstance(scheduler_config, dict):
        return False

    storage_config = scheduler_config.get("storage")
    if isinstance(storage_config, dict) and isinstance(
        storage_config.get("psql"), dict
    ):
        return True

    return isinstance(scheduler_config.get("psql"), dict)


LLM_DISPATCH_RUNTIME_EVENT = "llm.dispatch.runtime.snapshot"
LLM_DISPATCH_RUNTIME_MARKER = "LLM_DISPATCH_RUNTIME_SNAPSHOT"
LLM_DISPATCH_RUNTIME_SOURCE = "gateway://control-plane"
LLM_DISPATCH_RUNTIME_JOB_ID = "gateway"
LLM_DISPATCH_RUNTIME_COMPONENT = "llm_dispatch_runtime"
LLM_DISPATCH_RUNTIME_IDLE_SNAPSHOT_INTERVAL_S = 5.0


def _llm_dispatch_runtime_event_fingerprint(snapshot: dict[str, Any]) -> str:
    return json.dumps(snapshot, sort_keys=True, separators=(",", ":"), default=str)


def _llm_dispatch_runtime_event_result(snapshot: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(snapshot, sort_keys=True, default=str))


def _should_publish_llm_dispatch_runtime_event(
    *,
    fingerprint: str,
    last_fingerprint: Optional[str],
    last_published_at: float,
    now: float,
    unchanged_interval_s: float,
) -> bool:
    if fingerprint != last_fingerprint:
        return True

    if last_published_at <= 0:
        return True

    return now - last_published_at >= unchanged_interval_s


def _llm_dispatch_runtime_event_message(
    *,
    snapshot: dict[str, Any],
    queue_config: Any,
) -> EventMessage:
    fabric_group_id = str(getattr(queue_config, "fabric_group_id", "") or "")
    gateway_id = str(getattr(queue_config, "gateway_id", "") or "")
    pool_id = str(getattr(queue_config, "pool_id", "") or "")

    result = _llm_dispatch_runtime_event_result(snapshot)
    event = MarieEvent.engine_event(
        LLM_DISPATCH_RUNTIME_SOURCE,
        "LLM dispatch runtime snapshot updated",
        EngineEventData(
            metadata={
                "llm_dispatch_runtime": JsonMetadataValue(result),
            },
            marker_start=LLM_DISPATCH_RUNTIME_MARKER,
        ),
    )

    return Toast.marie_event_to_message(
        event,
        api_key="system:gateway",
        node="gateway",
        jobid=LLM_DISPATCH_RUNTIME_JOB_ID,
        extra_payload={
            "component": LLM_DISPATCH_RUNTIME_COMPONENT,
            "event_type": LLM_DISPATCH_RUNTIME_EVENT,
            "fabric_group_id": fabric_group_id,
            "gateway_id": gateway_id,
            "pool_id": pool_id,
            "result": result,
        },
    )


class MarieServerGateway(CompositeServer):
    """A custom Gateway for Marie server. Effectively we are providing a custom implementation of the Gateway class
    that providers communication between individual executors and the server.

    This utilizes service discovery(ETCD) to find deployed Executors from discovered gateways that could have spawned them(Flow/Deployment).
    Ref : https://docs.jina.ai/v3.14.0/concepts/gateway/customization/#custom-gateway
    """

    def __init__(self, **kwargs):
        """Main entry point for the Marie server
        :param yml_config:
        :param env:
        :param env_file:
        """
        super().__init__(**kwargs)

        # install handler for exceptions
        sys.excepthook = handle_exception
        install(show_locals=True)

        self.logger.info(f"Setting up MarieServerGateway")
        self._loop = get_or_reuse_loop()
        self.deployment_nodes = {}
        self.event_queue = asyncio.Queue(maxsize=512)
        self.ready_event = asyncio.Event()
        configured_gateway_instance_id = kwargs.get("gateway_instance_id")
        if configured_gateway_instance_id is None:
            configured_gateway_instance_id = getattr(self, "args", {}).get(
                "gateway_instance_id"
            )
        self.gateway_instance_id = str(
            configured_gateway_instance_id or f"{socket.gethostname()}:{uuid.uuid4()}"
        )

        # OTel metrics for gateway request tracking
        self._gateway_request_seconds = None
        # Observable gauge observations — updated by refresh, read by OTel callbacks
        self._slot_observations = {"capacity": {}, "used": {}, "available": {}}
        self._node_observations = {
            "active_requests": {},
            "slot_capacity": {},
            "accepting_traffic": {},
            "selection_count": {},
        }
        self._last_capacity_info_log_at = 0.0
        if self.meter:
            self._gateway_request_seconds = self.meter.create_histogram(
                name="marie_gateway_request_seconds",
                description="Time spent processing gateway API requests",
                unit="s",
            )
            # Slot capacity metrics (observable gauges — callback-based for
            # compatibility with all OTel exporters including ClickHouse/Prometheus)
            from opentelemetry.metrics import Observation

            self.meter.create_observable_gauge(
                name="marie_executor_slot_capacity",
                callbacks=[
                    lambda _: [
                        Observation(v, {"executor": k})
                        for k, v in self._slot_observations["capacity"].items()
                    ]
                ],
                description="Total slot capacity per executor",
                unit="{slots}",
            )
            self.meter.create_observable_gauge(
                name="marie_executor_slot_used",
                callbacks=[
                    lambda _: [
                        Observation(v, {"executor": k})
                        for k, v in self._slot_observations["used"].items()
                    ]
                ],
                description="Currently used slots per executor",
                unit="{slots}",
            )
            self.meter.create_observable_gauge(
                name="marie_executor_slot_available",
                callbacks=[
                    lambda _: [
                        Observation(v, {"executor": k})
                        for k, v in self._slot_observations["available"].items()
                    ]
                ],
                description="Available slots per executor",
                unit="{slots}",
            )
            self.meter.create_observable_gauge(
                name="marie_executor_node_inflight_requests",
                callbacks=[
                    lambda _: [
                        Observation(
                            value,
                            {"executor": key[0], "address": key[1]},
                        )
                        for key, value in self._node_observations[
                            "active_requests"
                        ].items()
                    ]
                ],
                description="In-flight gateway requests per executor node",
                unit="{requests}",
            )
            self.meter.create_observable_gauge(
                name="marie_executor_node_slot_capacity",
                callbacks=[
                    lambda _: [
                        Observation(
                            value,
                            {"executor": key[0], "address": key[1]},
                        )
                        for key, value in self._node_observations[
                            "slot_capacity"
                        ].items()
                    ]
                ],
                description="Configured slot capacity per executor node",
                unit="{slots}",
            )
            self.meter.create_observable_gauge(
                name="marie_executor_node_accepting_traffic",
                callbacks=[
                    lambda _: [
                        Observation(
                            value,
                            {"executor": key[0], "address": key[1]},
                        )
                        for key, value in self._node_observations[
                            "accepting_traffic"
                        ].items()
                    ]
                ],
                description="Whether an executor node can receive routed traffic",
                unit="{node}",
            )
            self.meter.create_observable_gauge(
                name="marie_executor_node_selection_count",
                callbacks=[
                    lambda _: [
                        Observation(
                            value,
                            {"executor": key[0], "address": key[1]},
                        )
                        for key, value in self._node_observations[
                            "selection_count"
                        ].items()
                    ]
                ],
                description="Gateway selections per executor node",
                unit="{selections}",
            )

        self.desired_map: Dict[tuple[str, str], DesiredDoc] = {}
        self.status_map: Dict[tuple[str, str], StatusDoc] = {}

        self.args = {**vars(self.runtime_args), **kwargs}
        yml_config = self.args.get("uses")

        if "env_file" not in kwargs:
            env_file = os.path.join(__config_dir__, ".env")
        else:
            env_file = kwargs["env_file"]
        load_env_file(dotenv_path=env_file)

        context = {}
        for k, v in os.environ.items():
            context[k] = v

        self.logger.info(f"Debugging information:")
        self.logger.info(f"__model_path__ = {__model_path__}")
        self.logger.info(f"__config_dir__ = {__config_dir__}")
        self.logger.info(f"__marie_home__ = {__marie_home__}")
        self.logger.info(f"__cache_path__ = {__cache_path__}")
        self.logger.info(f"yml_config = {yml_config}")
        self.logger.info(f"env_file = {env_file}")

        # Load the config file and inject the environment variables, we do this here because we need to pass the context
        # Another option is to modify the core BaseGateway.load_config method to accept context with environment variables
        self.args = JAML.expand_dict(self.args, context)

        if "kv_store_kwargs" not in self.args:
            raise BadConfigSource("Missing kv_store_kwargs in config")

        kv_store_kwargs = self.args["kv_store_kwargs"]
        expected_keys = [
            "provider",
            "hostname",
            "port",
            "username",
            "password",
            "database",
        ]
        if not all(key in kv_store_kwargs for key in expected_keys):
            raise ValueError(
                f"kv_store_kwargs must contain the following keys: {expected_keys}"
            )

        if "job_scheduler_kwargs" not in self.args:
            raise BadConfigSource("Missing job_scheduler_kwargs in config")

        job_scheduler_kwargs = dict(self.args["job_scheduler_kwargs"])
        if not all(key in job_scheduler_kwargs for key in expected_keys):
            raise ValueError(
                f"job_scheduler_kwargs must contain the following keys: {expected_keys}"
            )
        job_scheduler_kwargs["gateway_instance_id"] = self.gateway_instance_id
        self.args["job_scheduler_kwargs"] = job_scheduler_kwargs

        self.distributor = GatewayJobDistributor(
            gateway_streamer=None,
            deployment_nodes=None,
            logger=self.logger,
            ready_event=self.ready_event,
        )

        self.grpc_broker: Optional[GrpcEventBroker] = None
        # FIXME : We need to get etcd host and port from the config
        # we should start job scheduler after the gateway server is started
        storage = PostgreSQLKV(config=kv_store_kwargs, reset=False)
        self.etcd_client = get_etcd_client(convert_to_etcd_args(self.args))
        self.desired_store = DesiredStore(self.etcd_client)
        self.status_store = StatusStore(self.etcd_client)
        self.semaphore_store = SemaphoreStore(self.etcd_client, default_lease_ttl=30)
        self.capacity_manager = SlotCapacityManager(
            semaphore_store=self.semaphore_store,
            logger=self.logger,
            # Optional mapping if slot types differ from executor names:
            # slot_type_resolver=lambda executor: {"extract_executor": "ocr.gpu"}.get(executor, executor),
        )
        self.service_events_queue = asyncio.Queue(maxsize=512)
        self.state_events_queue = asyncio.Queue(maxsize=2048)  # tends to be chattier
        llm_queue_config = _llm_queue_runtime_config(self.args)
        self.llm_dispatch_runtime = GatewayLlmDispatchRuntime(
            logger=self.logger,
            config=llm_queue_config,
        )
        self._last_llm_dispatch_event_fingerprint: Optional[str] = None
        self._last_llm_dispatch_event_monotonic: float = 0.0
        self._background_services_shutdown = False
        self._background_services_lock = asyncio.Lock()
        self._control_plane_tasks: set[asyncio.Task[Any]] = set()

        self.job_manager = JobManager(
            storage=storage,
            job_distributor=self.distributor,
            etcd_client=self.etcd_client,
            desired_state_worker_count=int(
                job_scheduler_kwargs.get("desired_state_worker_count", 16)
            ),
            desired_state_max_pending=int(
                job_scheduler_kwargs.get("desired_state_max_pending", 128)
            ),
            job_event_worker_count=int(
                job_scheduler_kwargs.get("job_event_worker_count", 8)
            ),
            job_event_queue_size=int(
                job_scheduler_kwargs.get("job_event_queue_size", 1024)
            ),
        )
        self.job_scheduler = PostgreSQLJobScheduler(
            config=job_scheduler_kwargs,
            job_manager=self.job_manager,
            gateway_ready_event=self.ready_event,
        )

        # perform monkey patching
        GatewayRequestHandler.stream = self.custom_stream
        GatewayRequestHandler.Call = (
            self.custom_stream
        )  # Call is an alias for stream in GatewayRequestHandler
        GatewayRequestHandler.dry_run = self.custom_dry_run

        # FIXME : The resolver watch_service is not implemented correctly
        self.resolver = None

        self._rebuild_task: asyncio.Task | None = None
        self._rebuild_requested = False
        self._streamer_update_lock = asyncio.Lock()
        self._service_retry_tasks: dict[str, asyncio.Task] = {}
        self._service_retry_attempts: dict[str, int] = {}
        self._service_readiness: dict[str, dict[str, Any]] = {}
        self._debounce_s = 0.05

        def _extend_rest_function(app: 'FastAPI'):
            from fastapi import Depends, Header, HTTPException, Query, Request, Response
            from fastapi.responses import JSONResponse, StreamingResponse

            @app.exception_handler(Exception)
            async def global_exception_handler(request: Request, exc: Exception):
                self.logger.error(f"Unhandled exception: {exc}")
                return JSONResponse(
                    status_code=500,
                    content={
                        "status": "error",
                        "message": "Internal server error",
                        "detail": (str(exc)),
                    },
                )

            @app.exception_handler(HTTPException)
            async def http_exception_handler(request: Request, exc: HTTPException):
                return JSONResponse(
                    status_code=exc.status_code,
                    content={"status": "error", "message": exc.detail},
                )

            @app.on_event("shutdown")
            async def _shutdown():
                self.logger.info("Shutting down")
                await self._shutdown_background_services()

            @app.get("/check")
            async def get_health(text: str):
                self.logger.info(f"Received request at {datetime.now(timezone.utc)}")
                return {"result": "ok"}

            @app.api_route(
                path="/api/debug",
                methods=["GET"],
                summary="Get scheduler debug information /api/debug",
            )
            async def get_debug_info():
                """
                Get debug information from the job scheduler.
                :return:
                """
                self.logger.info(
                    f"Debug info requested at {datetime.now(timezone.utc)}"
                )
                try:
                    debug_data = await self.job_scheduler.debug_info()
                    debug_data["llm_dispatch"] = dispatch_runtime_live_state(
                        limit_per_pool=50
                    )
                    return {"status": "OK", "result": debug_data}
                except Exception as e:
                    self.logger.error(f"Error getting debug info: {str(e)}")
                    return {
                        "status": "error",
                        "result": f"Failed to get debug info: {str(e)}",
                    }

            @app.api_route(
                path="/api/llm-dispatch/runtime",
                methods=["GET"],
                summary="Get live LLM dispatch runtime information /api/llm-dispatch/runtime",
            )
            async def get_llm_dispatch_runtime(
                limit: int = Query(default=50, ge=1, le=250),
            ):
                self.logger.info(
                    f"LLM dispatch runtime requested at {datetime.now(timezone.utc)}"
                )
                try:
                    runtime_data = dispatch_runtime_live_state(limit_per_pool=limit)
                    return {"status": "OK", "result": runtime_data}
                except Exception as e:
                    self.logger.error(
                        f"Error getting LLM dispatch runtime info: {str(e)}"
                    )
                    return {
                        "status": "error",
                        "result": f"Failed to get LLM dispatch runtime info: {str(e)}",
                    }

            @app.api_route(
                path="/api/debug/reset-dags",
                methods=["POST"],
                summary="Reset active DAGs /api/debug/reset-dags",
            )
            async def reset_active_dags():
                """
                Reset the active DAGs in the job scheduler.
                :return:
                """
                self.logger.info(
                    f"Reset active DAGs requested at {datetime.now(timezone.utc)}"
                )
                try:
                    result = await self.job_scheduler.reset_active_dags()
                    if result["success"]:
                        return {"status": "OK", "result": result}
                    else:
                        return {"status": "error", "result": result}
                except Exception as e:
                    self.logger.error(f"Error resetting active DAGs: {str(e)}")
                    return {
                        "status": "error",
                        "result": f"Failed to reset active DAGs: {str(e)}",
                    }

            @app.api_route(
                path="/api/scheduler/pause",
                methods=["POST"],
                summary="Pause the scheduler — stops dispatching new jobs",
            )
            async def pause_scheduler():
                self.logger.info(
                    f"Scheduler pause requested at {datetime.now(timezone.utc)}"
                )
                try:
                    result = self.job_scheduler.pause()
                    return {"status": "OK", "result": result}
                except Exception as e:
                    self.logger.error(f"Error pausing scheduler: {str(e)}")
                    return {
                        "status": "error",
                        "result": f"Failed to pause scheduler: {str(e)}",
                    }

            @app.api_route(
                path="/api/scheduler/unpause",
                methods=["POST"],
                summary="Unpause the scheduler — resumes job dispatching",
            )
            async def unpause_scheduler():
                self.logger.info(
                    f"Scheduler unpause requested at {datetime.now(timezone.utc)}"
                )
                try:
                    result = self.job_scheduler.unpause()
                    return {"status": "OK", "result": result}
                except Exception as e:
                    self.logger.error(f"Error unpausing scheduler: {str(e)}")
                    return {
                        "status": "error",
                        "result": f"Failed to unpause scheduler: {str(e)}",
                    }

            def _operational_states(value: Optional[str]) -> list[str] | None:
                if value is None:
                    return None
                states = [state.strip().lower() for state in value.split(",")]
                if not states or any(not state for state in states):
                    raise HTTPException(
                        status_code=400,
                        detail="state must be a comma-separated list of job states",
                    )
                invalid = [
                    state
                    for state in states
                    if state not in {work_state.value for work_state in WorkState}
                ]
                if invalid:
                    raise HTTPException(
                        status_code=400,
                        detail=f"unsupported state: {', '.join(invalid)}",
                    )
                return states

            def _operational_attempt_states(value: Optional[str]) -> list[str] | None:
                if value is None:
                    return None
                states = [state.strip().lower() for state in value.split(",")]
                if not states or any(not state for state in states):
                    raise HTTPException(
                        status_code=400,
                        detail="state must be a comma-separated list of attempt states",
                    )
                return states

            @app.get(
                "/api/operations/events",
                summary="List cursor-paged operational lifecycle events",
            )
            async def list_operational_events(
                limit: int = Query(default=25, ge=1, le=100),
                before_at: Optional[datetime] = Query(default=None),
                before_id: Optional[str] = Query(default=None, max_length=160),
                window_seconds: int = Query(default=900, ge=60, le=86_400),
                severity: Optional[Literal["info", "warning", "bad"]] = Query(
                    default=None
                ),
                component: Optional[str] = Query(default=None, max_length=128),
                search: Optional[str] = Query(default=None, max_length=128),
            ):
                try:
                    result = await self.job_scheduler.diagnostics.events(
                        limit=limit,
                        before_at=before_at,
                        before_id=before_id,
                        window_seconds=window_seconds,
                        severity=severity,
                        component=component.strip() if component else None,
                        search=search.strip() if search else None,
                    )
                    return {"status": "OK", "result": result}
                except ValueError as error:
                    raise HTTPException(status_code=400, detail=str(error)) from error
                except Exception as error:
                    self.logger.error(
                        f"Operational event query failed: {error}", exc_info=True
                    )
                    raise HTTPException(
                        status_code=503,
                        detail="Operational event data is unavailable",
                    ) from error

            @app.get(
                "/api/operations/attempts",
                summary="List a bounded page of execution attempts",
            )
            async def list_operational_attempts(
                limit: int = Query(default=25, ge=1, le=100),
                offset: int = Query(default=0, ge=0),
                state: Optional[str] = Query(default=None, max_length=256),
                attention: Literal[
                    "any",
                    "active_too_long",
                    "stale_update",
                    "recovered",
                    "terminal_rejected",
                    "terminal_mismatch",
                    "owner_mismatch",
                ] = "any",
                gateway: Optional[str] = Query(default=None, max_length=256),
                executor: Optional[str] = Query(default=None, max_length=256),
                search: Optional[str] = Query(default=None, max_length=128),
                sort: Literal["attention", "newest", "oldest", "updated"] = (
                    "attention"
                ),
            ):
                try:
                    result = await self.job_scheduler.diagnostics.attempts(
                        limit=limit,
                        offset=offset,
                        states=_operational_attempt_states(state),
                        attention=attention,
                        gateway=gateway.strip() if gateway else None,
                        executor=executor.strip() if executor else None,
                        search=search.strip() if search else None,
                        sort=sort,
                    )
                    return {"status": "OK", "result": result}
                except HTTPException:
                    raise
                except ValueError as error:
                    raise HTTPException(status_code=400, detail=str(error)) from error
                except Exception as error:
                    self.logger.error(
                        f"Operational attempt query failed: {error}", exc_info=True
                    )
                    raise HTTPException(
                        status_code=503,
                        detail="Operational attempt data is unavailable",
                    ) from error

            @app.get(
                "/api/operations/flow",
                summary="Get scheduler flow pressure and stage latency",
            )
            async def get_operational_flow(
                window_seconds: int = Query(default=900, ge=60, le=86_400),
                queue: Optional[str] = Query(default=None, max_length=128),
                queue_limit: int = Query(default=25, ge=1, le=100),
            ):
                try:
                    result = await self.job_scheduler.diagnostics.flow(
                        window_seconds=window_seconds,
                        queue=queue.strip() if queue else None,
                        queue_limit=queue_limit,
                    )
                    return {"status": "OK", "result": result}
                except ValueError as error:
                    raise HTTPException(status_code=400, detail=str(error)) from error
                except Exception as error:
                    self.logger.error(
                        f"Operational flow query failed: {error}", exc_info=True
                    )
                    raise HTTPException(
                        status_code=503,
                        detail="Operational flow data is unavailable",
                    ) from error

            @app.get(
                "/api/operations/health",
                summary="Get read-only runtime dependency health",
            )
            async def get_operational_health():
                return {
                    "status": "OK",
                    "result": await self._operational_health_snapshot(),
                }

            @app.get(
                "/api/operations/jobs",
                summary="List a bounded page of operational job metadata",
            )
            async def list_operational_jobs(
                limit: int = Query(default=25, ge=1, le=100),
                offset: int = Query(default=0, ge=0),
                state: Optional[str] = Query(default=None, max_length=128),
                attention: Literal[
                    "any",
                    "queued_too_long",
                    "running_too_long",
                    "stale_update",
                    "retrying",
                    "failed",
                    "terminal_mismatch",
                ] = "any",
                queue: Optional[str] = Query(default=None, max_length=128),
                search: Optional[str] = Query(default=None, max_length=128),
                sort: Literal["attention", "newest", "oldest", "updated"] = (
                    "attention"
                ),
            ):
                try:
                    result = await self.job_scheduler.diagnostics.jobs(
                        limit=limit,
                        offset=offset,
                        states=_operational_states(state),
                        attention=attention,
                        queue=queue.strip() if queue else None,
                        search=search.strip() if search else None,
                        sort=sort,
                    )
                    return {"status": "OK", "result": result}
                except HTTPException:
                    raise
                except ValueError as error:
                    raise HTTPException(status_code=400, detail=str(error)) from error
                except Exception as error:
                    self.logger.error(
                        f"Operational job list query failed: {error}", exc_info=True
                    )
                    raise HTTPException(
                        status_code=503,
                        detail="Operational job data is unavailable",
                    ) from error

            @app.get(
                "/api/operations/jobs/{job_id}",
                summary="Get payload-free lifecycle details for one job",
            )
            async def get_operational_job(job_id: uuid.UUID):
                try:
                    result = await self.job_scheduler.diagnostics.job(str(job_id))
                    if result is None:
                        raise HTTPException(status_code=404, detail="Job not found")
                    return {"status": "OK", "result": result}
                except HTTPException:
                    raise
                except Exception as error:
                    self.logger.error(
                        f"Operational job detail query failed: {error}", exc_info=True
                    )
                    raise HTTPException(
                        status_code=503,
                        detail="Operational job data is unavailable",
                    ) from error

            @app.get(
                "/api/operations/execution-history",
                summary="List bounded worker execution history for a job or DAG",
            )
            async def list_operational_execution_history(
                job_id: Optional[uuid.UUID] = Query(default=None),
                dag_id: Optional[uuid.UUID] = Query(default=None),
                limit: int = Query(default=50, ge=1, le=100),
                offset: int = Query(default=0, ge=0),
            ):
                try:
                    result = await self.job_scheduler.diagnostics.execution_history(
                        job_id=str(job_id) if job_id else None,
                        dag_id=str(dag_id) if dag_id else None,
                        limit=limit,
                        offset=offset,
                    )
                    if result is None:
                        raise HTTPException(
                            status_code=404,
                            detail="Job or DAG not found",
                        )
                    return {"status": "OK", "result": result}
                except HTTPException:
                    raise
                except ValueError as error:
                    raise HTTPException(status_code=400, detail=str(error)) from error
                except Exception as error:
                    self.logger.error(
                        f"Operational execution history query failed: {error}",
                        exc_info=True,
                    )
                    raise HTTPException(
                        status_code=503,
                        detail="Operational execution history is unavailable",
                    ) from error

            @app.get(
                "/api/operations/dags",
                summary="List a bounded page of operational DAG metadata",
            )
            async def list_operational_dags(
                limit: int = Query(default=25, ge=1, le=100),
                offset: int = Query(default=0, ge=0),
                state: Optional[str] = Query(default=None, max_length=128),
                attention: Literal[
                    "any",
                    "queued_too_long",
                    "running_too_long",
                    "stale_update",
                    "retrying",
                    "failed",
                    "terminal_mismatch",
                ] = "any",
                queue: Optional[str] = Query(default=None, max_length=128),
                search: Optional[str] = Query(default=None, max_length=128),
                sort: Literal["attention", "newest", "oldest", "updated"] = (
                    "attention"
                ),
            ):
                try:
                    result = await self.job_scheduler.diagnostics.dags(
                        limit=limit,
                        offset=offset,
                        states=_operational_states(state),
                        attention=attention,
                        queue=queue.strip() if queue else None,
                        search=search.strip() if search else None,
                        sort=sort,
                    )
                    return {"status": "OK", "result": result}
                except HTTPException:
                    raise
                except ValueError as error:
                    raise HTTPException(status_code=400, detail=str(error)) from error
                except Exception as error:
                    self.logger.error(
                        f"Operational DAG list query failed: {error}", exc_info=True
                    )
                    raise HTTPException(
                        status_code=503,
                        detail="Operational DAG data is unavailable",
                    ) from error

            @app.get(
                "/api/operations/dags/{dag_id}",
                summary="Get a DAG and one bounded page of child jobs",
            )
            async def get_operational_dag(
                dag_id: uuid.UUID,
                job_limit: int = Query(default=25, ge=1, le=100),
                job_offset: int = Query(default=0, ge=0),
            ):
                try:
                    result = await self.job_scheduler.diagnostics.dag(
                        str(dag_id),
                        job_limit=job_limit,
                        job_offset=job_offset,
                    )
                    if result is None:
                        raise HTTPException(status_code=404, detail="DAG not found")
                    return {"status": "OK", "result": result}
                except HTTPException:
                    raise
                except Exception as error:
                    self.logger.error(
                        f"Operational DAG detail query failed: {error}", exc_info=True
                    )
                    raise HTTPException(
                        status_code=503,
                        detail="Operational DAG data is unavailable",
                    ) from error

            @app.get(
                "/api/operations/throughput",
                summary="Get bounded scheduler completion-throughput reports",
            )
            async def get_operational_throughput(
                lookback_hours: int = Query(default=24, ge=1, le=720),
                planner: Optional[str] = Query(default=None, max_length=250),
                planner_limit: int = Query(default=25, ge=1, le=100),
                task_limit: int = Query(default=25, ge=1, le=100),
            ):
                try:
                    result = await self.job_scheduler.diagnostics.throughput(
                        lookback_hours=lookback_hours,
                        planner=planner,
                        planner_limit=planner_limit,
                        task_limit=task_limit,
                    )
                    return {"status": "OK", "result": result}
                except ValueError as error:
                    raise HTTPException(status_code=400, detail=str(error)) from error
                except Exception as error:
                    self.logger.error(
                        f"Operational throughput query failed: {error}", exc_info=True
                    )
                    raise HTTPException(
                        status_code=503,
                        detail="Operational throughput data is unavailable",
                    ) from error

            async def list_jobs_handler(request: Request):
                try:
                    self.logger.info(
                        f"Received request at {datetime.now(timezone.utc)}"
                    )
                    params = request.path_params
                    state = params.get("state")

                    if state:
                        jobs = await self.job_scheduler.list_jobs(state=state)
                    else:
                        jobs = await self.job_scheduler.list_jobs()

                    return {"status": "OK", "result": jobs}

                except ValueError as e:
                    self.logger.warning(f"Invalid job state parameter: {e}")
                    return {
                        "status": "error",
                        "message": str(e),
                        "code": "INVALID_STATE",
                    }

                except Exception as e:
                    self.logger.error(f"Error listing jobs: {e}")
                    return {
                        "status": "error",
                        "message": f"Failed to retrieve jobs: {str(e)}",
                        "code": "INTERNAL_ERROR",
                    }

            # allows us to list jobs with or without state parameter and last slash
            app.add_api_route(
                path="/api/jobs",
                endpoint=list_jobs_handler,
                methods=["GET"],
                summary="Job listing endpoint /api/jobs with state filter",
            )

            app.add_api_route(
                path="/api/jobs/{state}",
                endpoint=list_jobs_handler,
                methods=["GET"],
                summary=f"Job listing endpoint /api/jobs",
            )

            @app.api_route(
                path="/api/jobsXX/{job_id}",
                methods=["GET"],
                summary="Stop a job /api/jobs/{job_id}",
            )
            async def get_job_info(request: Request):
                self.logger.info(f"Received request at {datetime.now(timezone.utc)}")
                # params = request.query_params
                params = request.path_params
                job_id = params.get("job_id")
                if not job_id:
                    return {"status": "error", "result": "Invalid job id"}
                job = await self.job_scheduler.get_job(job_id)
                if not job:
                    return {"status": "error", "result": "Job not found"}
                return {"status": "OK", "result": job}

            @app.api_route(
                path="/api/jobs/{job_id}/stop",
                methods=["GET"],
                summary="Stop a job /api/jobs/{job_id}/stop",
            )
            async def stop_job(request: Request):
                self.logger.info(f"Received request at {datetime.now(timezone.utc)}")
                return {"status": "OK", "result": "Job stopped"}

            @app.api_route(
                path="/api/jobs/{job_id}",
                methods=["DELETE"],
                summary="Delete a job /api/jobs/{job_id}/stop",
            )
            async def delete_job(request: Request):
                self.logger.info(f"Received request at {datetime.now(timezone.utc)}")
                return {"status": "OK", "result": "Job deleted"}

            @app.api_route(
                path="/api/v1/invoke",
                methods=["POST"],
                summary="Invoke a new command /api/v1/invoke",
                dependencies=[Depends(TokenBearer())],
            )
            async def invoke_command(
                request: Request, token: str = Depends(TokenBearer())
            ):
                self.logger.info(f"Received request at {datetime.now(timezone.utc)}")
                self.logger.debug(f"Token : {token}")

                metric_labels = {"endpoint": "/api/v1/invoke", "status": "success"}

                with MetricsTimer(self._gateway_request_seconds, metric_labels):
                    # Parse request payload with error handling
                    try:
                        payload = await request.json()
                    except Exception as e:
                        self.logger.error(f"Failed to parse JSON payload: {str(e)}")
                        metric_labels["status"] = "error"
                        raise HTTPException(
                            status_code=400, detail="Invalid JSON payload"
                        )

                    header = payload.get("header", {})
                    message = payload.get("parameters", {})

                    if "api_key" not in message or message["api_key"] is None:
                        message["api_key"] = token

                    req = DataRequest()
                    req.parameters = message

                    async def caller(req: DataRequest):
                        try:
                            decoded = await self.decode_request(req)
                            if isinstance(decoded, AsyncIterator):
                                async for response in decoded:
                                    yield response
                            else:
                                yield decoded
                        except Exception as e:
                            self.logger.error(f"Error in caller function: {str(e)}")
                            raise

                    try:
                        event_generator = caller(req)
                        response = await event_generator.__anext__()

                        # Validate response structure
                        if not hasattr(response, 'parameters'):
                            self.logger.error(
                                "Response object missing parameters attribute"
                            )
                            metric_labels["status"] = "error"
                            raise HTTPException(
                                status_code=500,
                                detail="Invalid response format from processing",
                            )

                        return {
                            "header": {},
                            "parameters": response.parameters,
                            "data": None,
                        }

                    except StopAsyncIteration:
                        self.logger.error("No response generated from event generator")
                        metric_labels["status"] = "error"
                        raise HTTPException(
                            status_code=500,
                            detail="No response generated from processing",
                        )
                    except Exception as e:
                        self.logger.error(f"Error processing request: {str(e)}")
                        metric_labels["status"] = "error"
                        raise HTTPException(
                            status_code=500, detail=f"Processing error: {str(e)}"
                        )

                # event_generator = _gen_dict_documents(caller(req))
                # return EventSourceResponse(event_generator)
                # # ['header', 'parameters', 'routes', 'data'
                # return {"header": {}, "parameters": {}, "data": None}

            # SSE endpoints removed - use gRPC EventStreamService instead
            # Clients should connect via gRPC bidirectional streaming for real-time events

            @app.api_route(
                path="/api/deployments",
                methods=["GET"],
                summary="Get all deployments information /api/deployments",
            )
            async def get_deployments():
                """
                Get registered deployment nodes.
                :return:
                """
                self.logger.info(
                    f"Deployments info requested at {datetime.now(timezone.utc)}"
                )
                try:
                    return {
                        "status": "OK",
                        "result": {
                            "deployment_nodes": self.deployment_nodes,
                        },
                    }
                except Exception as e:
                    self.logger.error(f"Error getting deployments info: {str(e)}")
                    return {
                        "status": "error",
                        "result": f"Failed to get deployments info: {str(e)}",
                    }

            @app.api_route(
                path="/api/deployment-nodes",
                methods=["GET"],
                summary="Get deployment nodes /api/deployment-nodes",
            )
            async def get_deployment_nodes():
                """
                Get all deployment nodes information.
                :return:
                """
                self.logger.info(
                    f"Deployment nodes info requested at {datetime.now(timezone.utc)}"
                )
                try:
                    return {
                        "status": "OK",
                        "result": self.deployment_nodes,
                    }
                except Exception as e:
                    self.logger.error(f"Error getting deployment nodes: {str(e)}")
                    return {
                        "status": "error",
                        "result": f"Failed to get deployment nodes: {str(e)}",
                    }

            @app.api_route(
                path="/api/deployment-status",
                methods=["GET"],
                summary="Get deployment status /api/deployment-status",
            )
            async def get_deployment_status():
                """
                Get deployment status including desired and status maps.
                :return:
                """
                self.logger.info(
                    f"Deployment status requested at {datetime.now(timezone.utc)}"
                )
                try:
                    # Convert the maps to a serializable format
                    desired_data = {}
                    for (node, deployment), desired in self.desired_map.items():
                        key = f"{node}/{deployment}"
                        desired_data[key] = {
                            "phase": desired.phase,
                            "epoch": desired.epoch,
                            "params": desired.params,
                            "updated_at": desired.updated_at,
                        }

                    status_data = {}
                    for (node, deployment), status in self.status_map.items():
                        key = f"{node}/{deployment}"
                        status_data[key] = {
                            "status_code": status.status_code,
                            "status_name": status.status_name,
                            "owner": status.owner,
                            "epoch": status.epoch,
                            "updated_at": status.updated_at,
                            "heartbeat_at": status.heartbeat_at,
                            "details": status.details,
                        }

                    return {
                        "status": "OK",
                        "result": {
                            "desired": desired_data,
                            "status": status_data,
                        },
                    }
                except Exception as e:
                    self.logger.error(f"Error getting deployment status: {str(e)}")
                    return {
                        "status": "error",
                        "result": f"Failed to get deployment status: {str(e)}",
                    }

            @app.api_route(
                path="/api/capacity",
                methods=["GET"],
                summary="Get capacity and slot information /api/capacity",
            )
            async def get_capacity():
                """
                Get capacity and slot information from the capacity manager.
                :return:
                """
                self.logger.info(
                    f"Capacity info requested at {datetime.now(timezone.utc)}"
                )
                try:
                    # Get current capacity snapshot
                    rows, totals = (
                        self.capacity_manager.compute_summary_rows_and_totals()
                    )
                    nodes = self._node_capacity_snapshot()
                    self._set_node_observations(nodes)

                    # Convert rows to dict format for easier consumption
                    slots = []
                    for row in rows:
                        slot, cap, tgt, used, avail, holders, notes = row
                        slots.append(
                            {
                                "name": slot,
                                "capacity": cap,
                                "target": tgt,
                                "used": used,
                                "available": avail,
                                "holders": holders,
                                "notes": notes,
                            }
                        )

                    return {
                        "status": "OK",
                        "result": {
                            "slots": slots,
                            "totals": totals,
                            "nodes": nodes,
                        },
                    }
                except Exception as e:
                    self.logger.error(f"Error getting capacity info: {str(e)}")
                    return {
                        "status": "error",
                        "result": f"Failed to get capacity info: {str(e)}",
                    }

            @app.get(
                path="/api/discovery/readiness",
                summary="Get service discovery readiness",
                tags=["Operations"],
            )
            async def get_discovery_readiness(
                state: Optional[
                    Literal["ready", "checking", "retrying", "error", "unready"]
                ] = Query(default=None),
            ):
                return {
                    "status": "OK",
                    "result": self._discovery_readiness_snapshot(state),
                }

            # Query Planner Management Endpoints
            @app.get(
                path="/api/planners",
                summary="List all registered query planners",
                tags=["Query Planners"],
            )
            async def list_planners():
                """
                Get a list of all registered query planners with their metadata.

                Returns:
                    List of planner metadata dictionaries
                """
                from marie.query_planner.base import QueryPlanRegistry

                planners = QueryPlanRegistry.list_planners_with_metadata()
                return {"planners": planners, "total": len(planners)}

            @app.get(
                path="/api/planners/{planner_id}",
                summary="Get a specific query planner by ID",
                tags=["Query Planners"],
            )
            async def get_planner(planner_id: str, response: Response):
                """
                Get metadata and plan definition for a specific query planner by ID.

                Args:
                    planner_id: ID of the query planner

                Returns:
                    Planner metadata including plan definition if available
                """
                from fastapi import status

                from marie.query_planner.base import QueryPlanRegistry

                metadata = QueryPlanRegistry.get_metadata_by_id(planner_id)

                if metadata is None:
                    response.status_code = status.HTTP_404_NOT_FOUND
                    return {"error": f"Planner with ID '{planner_id}' not found"}

                return metadata.model_dump()

            @app.post(
                path="/api/planners",
                summary="Register a new query planner from JSON",
                tags=["Query Planners"],
                status_code=201,
            )
            async def register_planner(
                name: str,
                plan: Dict[str, Any],
                description: Optional[str] = None,
                version: str = "1.0.0",
                tags: Optional[list] = None,
                category: Optional[str] = None,
                response: Response = None,
            ):
                """
                Register a new query planner from a JSON plan definition.

                This endpoint allows Marie Studio to publish query plan templates.

                Args:
                    name: Unique name for the planner
                    plan: JSON plan definition (QueryPlan structure)
                    description: Optional description
                    version: Version string (default: "1.0.0")
                    tags: Optional list of tags
                    category: Optional category

                Returns:
                    Success message with planner metadata
                """
                from marie.query_planner.base import QueryPlanRegistry

                success = QueryPlanRegistry.register_from_json(
                    name=name,
                    plan_definition=plan,
                    description=description,
                    version=version,
                    tags=tags,
                    category=category,
                )

                if success:
                    metadata = QueryPlanRegistry.get_metadata(name)
                    return {
                        "success": True,
                        "message": f"Planner '{name}' registered successfully",
                        "planner": metadata.model_dump() if metadata else None,
                    }
                else:
                    response.status_code = 400
                    return {
                        "success": False,
                        "error": f"Failed to register planner '{name}'",
                    }

            @app.delete(
                path="/api/planners/{planner_id}",
                summary="Unregister a query planner by ID",
                tags=["Query Planners"],
            )
            async def unregister_planner(planner_id: str, response: Response):
                """
                Unregister a query planner by ID.

                Args:
                    planner_id: ID of the planner to unregister

                Returns:
                    Success message
                """
                from marie.query_planner.base import QueryPlanRegistry

                success = QueryPlanRegistry.unregister_by_id(planner_id)

                if success:
                    return {
                        "success": True,
                        "message": f"Planner unregistered successfully",
                    }
                else:
                    response.status_code = 404
                    return {
                        "success": False,
                        "error": f"Planner with ID '{planner_id}' not found",
                    }

            @app.api_route(
                path="/api/registry",
                methods=["GET"],
                summary="Get component registry and query planner information",
            )
            async def get_registry_info():
                self.logger.info(
                    f"Registry info requested at {datetime.now(timezone.utc)}"
                )
                try:
                    from marie.extract.registry import component_registry
                    from marie.query_planner.base import QueryPlanRegistry

                    registry_data = component_registry.get_registry_info()
                    planner_info = QueryPlanRegistry.get_planner_info()
                    planner_details = QueryPlanRegistry.list_planners_with_metadata()

                    return {
                        "status": "OK",
                        "result": {
                            "components": registry_data,
                            "planners": {
                                **planner_info,
                                "planner_details": planner_details,
                            },
                        },
                    }
                except Exception as e:
                    self.logger.error(f"Error getting registry info: {str(e)}")
                    return {
                        "status": "error",
                        "result": f"Failed to get registry info: {str(e)}",
                    }

            # Register Wasm compilation routes
            register_wasm_routes(app)

            # Register sandbox blueprint-import and plugin-install routes
            register_blueprint_routes(app)

            # Register the knowledge-base API extension (search/stats/delete);
            # streamer resolved lazily — it does not exist at registration time
            register_kb_routes(app, self.args.get("kb"), lambda: self.streamer)

            return app

        marie.helper.extend_rest_interface = _extend_rest_function

    async def custom_stream(
        self, request_iterator, context=None, *args, **kwargs
    ) -> AsyncIterator["Request"]:
        """
        Intercept the stream of requests and process them.

        :param request_iterator: An asynchronous iterator that provides the request objects.
        :param context: The context of the API request. Defaults to None.
        :param args: Additional positional arguments.
        :param kwargs: Additional keyword arguments.
        :return: An asynchronous iterator that yields the response objects.

        """
        self.logger.debug(f"intercepting stream : custom_stream")
        async for request in request_iterator:
            metric_labels = {"endpoint": "grpc", "status": "success"}
            with MetricsTimer(self._gateway_request_seconds, metric_labels):
                try:
                    decoded = await self.decode_request(request)
                    if isinstance(decoded, AsyncIterator):
                        async for response in decoded:
                            yield response
                    else:
                        yield decoded
                except Exception:
                    metric_labels["status"] = "error"
                    raise

    async def decode_request(
        self, request: Request
    ) -> Response | AsyncGenerator[Request, None]:
        """
        Decode the request and return a response.
        :param request: The request to decode.
        :return: The response.
        """
        message = request.parameters
        self.logger.debug(f"Message details : {message}")

        if "invoke_action" not in message:
            response = Response()
            response.parameters = {"error": "Invalid request, missing invoke_action"}
            return response

        invoke_action = message["invoke_action"]

        if "api_key" not in invoke_action or invoke_action["api_key"] is None:
            response = Response()
            response.parameters = {"error": "Invalid request, missing api_key"}
            return response

        if not APIKeyManager.is_valid(invoke_action["api_key"]):
            response = Response()
            response.parameters = {"error": "Invalid or expired token"}
            return response

        command = invoke_action.get("command")  # job

        if command == "job":
            return self.handle_job_command(invoke_action)
        elif command == "nodes":
            return self.handle_nodes_command(invoke_action)
        else:
            return self.error_response(
                f"Command not recognized or not implemented : {command}", None
            )

    async def handle_nodes_command(
        self, message: dict
    ) -> AsyncGenerator[Request, None]:
        """
        Handle nodes command based on the action provided in the message.

        :param message: Dictionary containing the job command details.
                        It should have the "action" key specifying the action to perform.
        :return: Response object containing the result of the nodes command.

        :raises ValueError: If the action provided in the message is not recognized.
        """
        #
        action = message.get("action")  # list
        self.logger.info(f"Handling nodes action : {action}")
        if action == "list":
            docs = DocList[TextDoc]()
            unique_nodes = set()

            for executor, nodes in self.deployment_nodes.items():
                for node in nodes:
                    if node["address"] not in unique_nodes:
                        unique_nodes.add(node["address"])
                        docs.append(TextDoc(text=node["address"]))

            req = DataRequest()
            req.document_array_cls = DocList[TextDoc]
            req.data.docs = docs
            req.parameters = {
                "status": "ok",
                "msg": "Received nodes list request",
            }
            yield req
        else:
            yield self.error_response(f"Action not recognized : {action}", None)

    async def handle_job_command(self, message: dict) -> AsyncGenerator[Request, None]:
        """
        Handle job command based on the action provided in the message.

        :param message: Dictionary containing the job command details.
                        It should have the "action" key specifying the action to perform.
        :return: Response object containing the result of the job command.

        :raises ValueError: If the action provided in the message is not recognized.
        """

        action = message.get("action")  # status, submit, logs, stop
        self.logger.debug(f"Handling job action : {action}")

        if action == "status":
            response = Response()
            response.parameters = {
                "status": "ok",
                "msg": "Received status request",
            }
            yield response
        elif action == "submit":
            yield await self.handle_job_submit_command(message)
        elif action == "logs":
            response = Response()
            response.parameters = {
                "status": "ok",
                "msg": "Received logs request",
            }
            yield response
            for i in range(0, 10):
                response = Response()
                response.parameters = {
                    "msg": f"log message #{i}",
                }
                yield response
                await asyncio.sleep(1)
        elif action == "events":
            response = Response()
            response.parameters = {
                "status": "ok",
                "msg": "Received events request",
            }
            yield response
        else:
            yield self.error_response(f"Action not recognized : {action}")

    async def handle_job_submit_command(self, message: Dict[str, Any]) -> Request:
        """
        Handle job submission command.

        :param message: The message containing the job information.
        :return: The response with the submission result.
        """
        start_time = time.time()
        self.logger.debug(f"Handling job submit command : {message}")
        silence_exceptions = strtobool(
            os.environ.get("MARIE_SILENCE_EXCEPTIONS", False)
        )

        api_key = message["api_key"]

        now = datetime.now(timezone.utc)
        submission_model = JobSubmissionModel(**message)
        metadata = submission_model.metadata
        project_id = metadata.get("project_id", None)
        ref_type = metadata.get("ref_type", None)
        ref_id = metadata.get("ref_id", None)
        submission_policy = metadata.get("policy", None)
        retry = DEFAULT_RETRY_POLICY
        event_name = submission_model.name

        scheduler_owns_failure_notification = False
        try:
            priority = self._parse_priority(metadata.get("priority", 0))
            soft_sla, hard_sla = self._normalize_slas(
                now,
                metadata.get("soft_sla"),
                metadata.get("hard_sla"),
            )
            publish_accepted_event = strtobool(
                os.environ.get("MARIE_GATEWAY_PUBLISH_ACCEPTED_EVENT", False)
            )
        except ValueError as exc:
            return self.error_response(str(exc), None, silence_exceptions)

        # ensure that project_id, ref_type, ref_id are int  metadata of the submission model
        # we need this as this what we will use for Toast events
        if not ref_type or not ref_id or not project_id:
            return self.error_response(
                "Project ID , Reference Type and Reference ID are required in the metadata",
                None,
            )

        # Event name is the name of the job, and it will be used to generate the toast event
        if (
            not event_name
            or (any(not (c.isalnum() or c in '-_.') for c in event_name))
            or event_name.startswith('amq.')
        ):
            return self.error_response(
                "Event name can only contain letters, digits, hyphen, underscore and period",
                None,
            )
        if len(event_name.encode()) > 255:
            return self.error_response(
                "Event name cannot exceed 255 bytes in length", None
            )

        work_info = WorkInfo(
            name=event_name,
            # Persisted priority is the operator override. SLA urgency is derived later in the planner.
            priority=priority,
            data=message,
            state=WorkState.CREATED,
            retry_limit=retry.retry_limit,
            retry_delay=retry.retry_delay,
            retry_backoff=retry.retry_backoff,
            start_after=now,
            expire_in_seconds=0,
            keep_until=now + timedelta(days=2),
            policy=submission_policy,
            soft_sla=soft_sla,
            hard_sla=hard_sla,
        )

        try:
            scheduler_trace(
                "gateway_submit_received",
                job_id=work_info.id,
                dag_id=work_info.id,
                event_name=event_name,
                ref_id=ref_id,
                ref_type=ref_type,
                project_id=project_id,
                planner=metadata.get("planner"),
                priority=priority,
                soft_sla=soft_sla.isoformat() if soft_sla else None,
                hard_sla=hard_sla.isoformat() if hard_sla else None,
                gateway_instance_id=self.gateway_instance_id,
            )
            scheduler_owns_failure_notification = True
            job_id = await self.job_scheduler.submit_job(work_info)
            scheduler_trace(
                "gateway_submit_accepted",
                dag_id=job_id,
                event_name=event_name,
                ref_id=ref_id,
                ref_type=ref_type,
                planner=metadata.get("planner"),
                gateway_instance_id=self.gateway_instance_id,
            )

            # Tag the active ASGI span with session.id = job_id (the dag_id)
            # so ClickHouse materializes oi_session_id for session grouping.
            from opentelemetry import trace as otel_trace

            active_span = otel_trace.get_current_span()
            if active_span and active_span.is_recording():
                active_span.set_attribute("session.id", job_id)

            response = Response()
            response.parameters = {
                "status": "ok",
                "msg": f"job submitted with id {job_id}",
                "job_id": job_id,
            }
            self.logger.info(f"Job submitted with id {job_id}")
            if publish_accepted_event:
                try:
                    published = await mark_as_accepted(
                        api_key=api_key,
                        job_id=job_id,
                        event_name=event_name,
                        job_tag=ref_type,
                        status="OK",
                        timestamp=current_milli_time(),
                        payload=metadata,
                    )
                    scheduler_trace(
                        "gateway_submit_accepted_notified",
                        dag_id=job_id,
                        event_name=event_name,
                        ref_id=ref_id,
                        ref_type=ref_type,
                        published=published,
                        gateway_instance_id=self.gateway_instance_id,
                    )
                except Exception as notification_error:
                    self.logger.error(
                        f"Failed to publish accepted event for durable job "
                        f"{job_id}: {notification_error}"
                    )

            return response
        except BaseException as ex:
            scheduler_trace(
                "gateway_submit_failed",
                event_name=event_name,
                ref_id=ref_id,
                ref_type=ref_type,
                error=repr(ex),
                gateway_instance_id=self.gateway_instance_id,
            )
            response = self.error_response(
                f"Failed to submit job. {ex}", ex, silence_exceptions
            )
            if scheduler_owns_failure_notification:
                return response
            try:
                exc_msg = response.parameters.get("exception", "Unknown error")
                job_key = f"failed/{ref_type}/{ref_id}"

                self.logger.error(f"Marking job as failed: {job_key}")
                await mark_as_failed(
                    api_key=project_id,
                    job_id=job_key,
                    event_name=work_info.name,
                    job_tag=ref_type,
                    status="FAILED",
                    timestamp=current_milli_time(),
                    payload=exc_msg,
                )
            except Exception as e:
                self.logger.error(f"Failed to mark job as failed: {e}")
            return response
        finally:
            elapsed_time = time.time() - start_time
            self.logger.debug(f"Job submission completed in {elapsed_time:.2f} seconds")

    @staticmethod
    def _parse_priority(raw: Any) -> int:
        try:
            return int(raw if raw is not None else 0)
        except (TypeError, ValueError) as exc:
            raise ValueError("priority must be an integer") from exc

    @staticmethod
    def _parse_optional_datetime(raw: Any) -> Optional[datetime]:
        if raw is None:
            return None
        if isinstance(raw, datetime):
            value = raw
        elif isinstance(raw, str):
            value = datetime.fromisoformat(raw)
        else:
            raise ValueError("SLA values must be datetimes or ISO-8601 strings")

        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    @classmethod
    def _normalize_slas(
        cls,
        now: datetime,
        soft_sla_raw: Any,
        hard_sla_raw: Any,
    ) -> tuple[datetime, datetime]:
        soft_sla = cls._parse_optional_datetime(soft_sla_raw) or now
        hard_sla = cls._parse_optional_datetime(hard_sla_raw) or (
            soft_sla + timedelta(hours=4)
        )
        if soft_sla > hard_sla:
            raise ValueError("Soft SLA must be before Hard SLA")
        return soft_sla, hard_sla

    def error_response(
        self, msg: str, exception: Optional[Exception], silence_exceptions: bool = False
    ) -> Response:
        """
        Set the response parameters to indicate a failure.
        :param msg: A string representing the error message.
        :param exception: An optional exception that triggered the error.
        :param silence_exceptions: A boolean indicating whether to silence the exception.
        :return: The response object with the error parameters set.
        """
        try:
            self.logger.error(f"processing error : {msg} > {exception}", exc_info=True)
            # get the traceback and clear the frames to avoid memory leak
            exc_msg = {"type": "Unknown", "message": "Unknown error"}
            if exception:
                _, val, tb = sys.exc_info()
                traceback.clear_frames(tb)

                filename = tb.tb_frame.f_code.co_filename
                name = tb.tb_frame.f_code.co_name
                line_no = tb.tb_lineno
                # print traceback
                detail = "Internal Server Error - processing error"
                exc_msg = {}

                if not silence_exceptions:
                    detail = exception.__str__()

                exc_msg = {
                    "type": type(exception).__name__,
                    "message": detail,
                    "filename": filename.split("/")[-1],
                    "name": name,
                    "line_no": line_no,
                }

            response = Response()
            response.parameters = {"status": "error", "msg": msg, "exception": exc_msg}
            return response

            # return {"status": "error", "error": {"code": code, "message": detail}}
        except Exception as e:
            logger.error(f"Failure handling exception: {e}", exc_info=True)
            raise e

    async def custom_dry_run(self, empty, context) -> jina_pb2.StatusProto:
        logger.info("Running custom dry run logic")

        status_message = StatusMessage()
        status_message.set_code(jina_pb2.StatusProto.SUCCESS)
        return status_message.proto

    async def setup_server(self):
        """
        setup servers inside CompositeServer
        """
        self.logger.debug(f"Setting up MarieGateway server")
        await super().setup_server()

        self.grpc_broker = setup_toast_events(self.args.get("toast", {}))

        # Register EventStreamService on the gRPC server if broker is configured
        if self.grpc_broker:
            for server in self.servers:
                if hasattr(server, 'register_event_service'):
                    await server.register_event_service(self.grpc_broker)
                    await self.grpc_broker.start()
                    break
        storage_config = self.args.get("storage", {})
        setup_storage(storage_config)
        setup_auth(self.args.get("auth", {}))
        setup_llm_tracking(self.args.get("llm_tracking", {}))

        await self.setup_service_discovery(
            etcd_host=self.args["discovery_host"],
            etcd_port=self.args["discovery_port"],
            service_name=self.args["discovery_service_name"],
        )

    async def _start_gateway_background_runtimes(self) -> None:
        await self.llm_dispatch_runtime.start()

    async def _stop_control_plane_tasks(self) -> None:
        tasks = {task for task in self._control_plane_tasks if not task.done()}
        if self._rebuild_task is not None and not self._rebuild_task.done():
            tasks.add(self._rebuild_task)

        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        self._control_plane_tasks.clear()
        self._rebuild_task = None

    async def _shutdown_background_services(self) -> None:
        async with self._background_services_lock:
            if self._background_services_shutdown:
                return
            self._background_services_shutdown = True

            resolver = self.resolver
            if resolver is not None:
                try:
                    resolver.stop()
                except Exception as exc:
                    self.logger.error("Failed stopping service resolver: %s", exc)
                self.resolver = None

            await self._stop_control_plane_tasks()

            try:
                await self.job_scheduler.stop()
            except Exception as exc:
                self.logger.error("Failed stopping job scheduler: %s", exc)

            try:
                await self.job_manager.shutdown()
            except Exception as exc:
                self.logger.error("Failed stopping job manager: %s", exc)

            if self.grpc_broker:
                try:
                    await self.grpc_broker.stop()
                except Exception as exc:
                    self.logger.error("Failed stopping gRPC broker: %s", exc)

            try:
                await self.llm_dispatch_runtime.stop()
            except Exception as exc:
                self.logger.error("Failed stopping LLM dispatch runtime: %s", exc)

            etcd_client = self.etcd_client
            if etcd_client is not None:
                try:
                    close_etcd_client(etcd_client)
                except Exception as exc:
                    self.logger.error("Failed closing etcd client: %s", exc)

    async def shutdown(self):
        self.logger.debug("Shutting down Marie gateway")
        await self._shutdown_background_services()
        await super().shutdown()

    async def run_server(self):
        """Run servers inside CompositeServer forever"""
        run_server_tasks = []
        for server in self.servers:
            run_server_tasks.append(asyncio.create_task(server.run_server()))

        self._control_plane_tasks = {
            asyncio.create_task(
                self.process_service_events(max_errors=5),
                name="gateway-service-events",
            ),
            asyncio.create_task(
                self.process_state_events(max_errors=10),
                name="gateway-state-events",
            ),
            asyncio.create_task(
                self.wait_and_start_scheduler(timeout=5),
                name="gateway-scheduler-start",
            ),
            asyncio.create_task(
                self._reconcile_loop(interval_s=10),
                name="gateway-reconcile",
            ),
            asyncio.create_task(
                self._capacity_broadcast_loop(interval_s=5),
                name="gateway-capacity-broadcast",
            ),
        }
        if self.grpc_broker:
            self._control_plane_tasks.add(
                asyncio.create_task(
                    self._llm_dispatch_broadcast_loop(interval_s=1.0),
                    name="gateway-llm-dispatch-broadcast",
                )
            )
        run_server_tasks.extend(self._control_plane_tasks)

        try:
            await asyncio.gather(*run_server_tasks)
        except asyncio.CancelledError:
            if not self._background_services_shutdown:
                raise
            await asyncio.gather(*run_server_tasks, return_exceptions=True)

    def _node_capacity_snapshot(self) -> list[dict[str, Any]]:
        if not getattr(self, "streamer", None):
            return []

        gateways_by_node: dict[tuple[str, str], set[str]] = {}
        for executor, nodes in self._routable_deployment_nodes().items():
            for node in nodes:
                address = node.get("address") or ""
                parsed_address = urlparse(address)
                normalized_address = parsed_address.netloc or address
                gateway = node.get("gateway")
                if normalized_address and gateway:
                    gateways_by_node.setdefault(
                        (executor, normalized_address), set()
                    ).add(gateway)

        snapshot = []
        for node in self.streamer.get_node_stats():
            executor = node["executor"]
            address = node["address"]
            active_requests = int(node["active_requests"])
            slot_capacity = self.capacity_manager.slots_per_node(executor)
            accepting_traffic = bool(node["accepting_traffic"])

            if not accepting_traffic:
                routing_state = "unavailable"
            elif slot_capacity <= 0:
                routing_state = "disabled"
            elif active_requests >= slot_capacity:
                routing_state = "saturated"
            elif active_requests == 0:
                routing_state = "idle"
            else:
                routing_state = "active"

            snapshot.append(
                {
                    **node,
                    "gateways": sorted(
                        gateways_by_node.get((executor, address), set())
                    ),
                    "slot_capacity": slot_capacity,
                    "slot_available": max(slot_capacity - active_requests, 0),
                    "utilization_pct": (
                        round(active_requests * 100 / slot_capacity, 1)
                        if slot_capacity > 0
                        else None
                    ),
                    "routing_state": routing_state,
                }
            )

        return snapshot

    def _set_node_observations(self, nodes: list[dict[str, Any]]) -> None:
        self._node_observations = {
            "active_requests": {
                (node["executor"], node["address"]): node["active_requests"]
                for node in nodes
            },
            "slot_capacity": {
                (node["executor"], node["address"]): node["slot_capacity"]
                for node in nodes
            },
            "accepting_traffic": {
                (node["executor"], node["address"]): int(node["accepting_traffic"])
                for node in nodes
            },
            "selection_count": {
                (node["executor"], node["address"]): node["selection_count"]
                for node in nodes
            },
        }

    async def _publish_capacity_event(self) -> None:
        """Publish current capacity state as an event."""
        try:
            rows, totals, summary = self.capacity_manager.refresh_from_nodes(
                self._routable_deployment_nodes()
            )
            now = time.monotonic()
            if (
                now - self._last_capacity_info_log_at
                >= CAPACITY_INFO_LOG_INTERVAL_SECONDS
            ):
                self.logger.info(summary)
                self._last_capacity_info_log_at = now
            else:
                self.logger.debug(summary)

            # Update observable gauge observations (read by OTel callbacks)
            # rows: [(slot, capacity, target, used, available, holders, notes), ...]
            if self._slot_observations is not None:
                obs = {"capacity": {}, "used": {}, "available": {}}
                for row in rows:
                    slot, cap, tgt, used, avail, holders, notes = row
                    obs["capacity"][slot] = cap
                    obs["used"][slot] = used
                    obs["available"][slot] = avail
                obs["capacity"]["_total"] = totals["capacity"]
                obs["used"]["_total"] = totals["used"]
                obs["available"]["_total"] = totals["available"]
                self._slot_observations = obs

            node_stats = self._node_capacity_snapshot()
            self._set_node_observations(node_stats)

            capacity_stats = (rows, totals)
            self.logger.debug(f"Publishing capacity stats: {capacity_stats}")
            event = MarieEvent.engine_event(
                "gateway://control-plane",
                "Cluster capacity updated",
                EngineEventData(
                    metadata={
                        "stats": JsonMetadataValue(capacity_stats),
                        "capacity": JsonMetadataValue(capacity_stats),
                    },
                    marker_start=MarieEventType.RESOURCE_EXECUTOR_UPDATED,
                ),
            )
            await Toast.notify(
                event,
                api_key="system:gateway",
                node="gateway",
            )
        except Exception as ex:
            if _is_known_connection_error(ex):
                self.logger.debug(
                    f"Capacity event skipped (etcd unavailable): {type(ex).__name__}"
                )
            else:
                self.logger.error(
                    f"Failed to publish capacity event: {ex}", exc_info=True
                )

    async def _capacity_broadcast_loop(self, interval_s: float = 5.0) -> None:
        """Periodically broadcast capacity state to connected clients."""
        self.logger.info(f"Starting capacity broadcast loop (interval={interval_s}s)")
        # Wait for gateway to be ready before starting broadcasts
        await asyncio.sleep(interval_s)

        while True:
            try:
                await self._publish_capacity_event()
            except Exception as ex:
                if _is_known_connection_error(ex):
                    self.logger.debug(f"Capacity broadcast skipped (etcd unavailable)")
                else:
                    self.logger.error(f"Capacity broadcast error: {ex}", exc_info=True)
            await asyncio.sleep(interval_s)

    async def _publish_llm_dispatch_runtime_event(
        self,
        *,
        unchanged_interval_s: float = LLM_DISPATCH_RUNTIME_IDLE_SNAPSHOT_INTERVAL_S,
    ) -> None:
        snapshot = dispatch_runtime_live_state(limit_per_pool=50)
        fingerprint = _llm_dispatch_runtime_event_fingerprint(snapshot)
        now = time.monotonic()
        if not _should_publish_llm_dispatch_runtime_event(
            fingerprint=fingerprint,
            last_fingerprint=self._last_llm_dispatch_event_fingerprint,
            last_published_at=self._last_llm_dispatch_event_monotonic,
            now=now,
            unchanged_interval_s=unchanged_interval_s,
        ):
            return

        event = _llm_dispatch_runtime_event_message(
            snapshot=snapshot,
            queue_config=self.llm_dispatch_runtime.config,
        )
        await Toast.notify(event.event, event)
        self._last_llm_dispatch_event_fingerprint = fingerprint
        self._last_llm_dispatch_event_monotonic = now

    async def _llm_dispatch_broadcast_loop(self, interval_s: float = 1.0) -> None:
        self.logger.info(
            "Starting LLM dispatch broadcast loop "
            f"(poll={interval_s}s, idle_snapshot={LLM_DISPATCH_RUNTIME_IDLE_SNAPSHOT_INTERVAL_S}s)"
        )
        await asyncio.sleep(interval_s)

        while True:
            try:
                await self._publish_llm_dispatch_runtime_event()
            except Exception as exc:
                self.logger.error(
                    "LLM dispatch broadcast error: %s",
                    exc,
                    exc_info=True,
                )
            await asyncio.sleep(interval_s)

    async def wait_and_start_scheduler(self, timeout: int = 5):
        """Start the scheduler after the discovery watch and initial enumeration."""
        self.logger.info("Waiting for initial service discovery enumeration")

        while not self.ready_event.is_set():
            try:
                await asyncio.wait_for(self.ready_event.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                self.logger.warning(
                    "Initial service discovery is still in progress; scheduler remains paused"
                )

        await self.job_scheduler.start()

        await self._start_gateway_background_runtimes()
        setup_sensor_worker(
            self.args.get("sensors", {}), self.args.get("kv_store_kwargs", {})
        )

        attached = attach_sensor_worker_scheduler(self.job_scheduler)
        self.logger.info(
            f"SensorWorker job scheduler attachment: {'attached' if attached else 'skipped (no SensorWorker)'}"
        )

    async def setup_service_discovery(
        self,
        etcd_host: str,
        etcd_port: int,
        service_name: str,
        watchdog_interval: int = 1,
    ):
        """
         Setup service discovery for the gateway.

        :param etcd_host: str - The host address of the ETCD service. Default is "0.0.0.0".
        :param etcd_port: int - The port of the ETCD service. Default is 2379.
        :param service_name: str - The name of the service to discover.
        :param watchdog_interval: int - The interval in seconds between each service address check. Default is 5.
        :return: None

        """
        self.logger.info(
            f"Setting up service discovery : {etcd_host}:{etcd_port}/{service_name}"
        )

        if not service_name:
            raise BadConfigSource("Service name must be provided for service discovery")

        async def _start_watcher():
            try:
                self.resolver = EtcdServiceResolver(
                    etcd_client=self.etcd_client,
                    namespace="marie",
                    start_listener=False,
                    listen_timeout=watchdog_interval,
                )

                # watch services
                self.resolver.watch_service(
                    service_name,
                    self._on_service_event,
                    notify_on_start=True,
                    initial_snapshot_callback=self._on_service_snapshot_complete,
                )

                # watch node status changes
                # self.resolver.watch_service(
                #     ROOT, self._on_state_event, notify_on_start=True
                # )
                # start the state watch
                await self._start_state_watch()
            except Exception as e:
                self.logger.error(
                    f"Failed to initialize etcd client on {etcd_host}:{etcd_port}"
                )
                if isinstance(e, RuntimeFailToStart):
                    raise e
                raise RuntimeFailToStart(
                    f"Initialize etcd client failed on {etcd_host}:{etcd_port}, ensure the etcd server is running.",
                )

        task = asyncio.create_task(_start_watcher())
        try:
            await task  # This raises an exception if the task had an exception
        except Exception as e:
            self.logger.error(f"Task watcher failed: {e}")
            if isinstance(e, RuntimeFailToStart):
                raise e
            raise RuntimeFailToStart(
                f"Unexpected error during service discovery setup for etcd client on {etcd_host}:{etcd_port}",
                details=str(e),
            )

    def handle_discovery_event(self, service: str, event: str) -> None:
        """
        Enqueue the event to be processed.

        :param service: The name of the service that is available.
        :param event: The event that triggered the method.
        :return: None
        """

        asyncio.run_coroutine_threadsafe(
            self.event_queue.put((service, event)), self._loop
        )

    def _on_service_event(self, service: str, event) -> None:
        # event has .key .event .value (resolver contract)
        se = ServiceEvent(
            kind=EventKind.SERVICE,
            service=service,
            ev_type=event.event,
            value=event.value,
            key=event.key,
        )
        asyncio.run_coroutine_threadsafe(self.service_events_queue.put(se), self._loop)

    def _on_service_snapshot_complete(self, service: str, event_count: int) -> None:
        self.logger.info(
            "Initial service snapshot enumerated: service=%s events=%d",
            service,
            event_count,
        )
        event = ServiceEvent(
            kind=EventKind.SERVICE,
            service=service,
            ev_type=SERVICE_SNAPSHOT_COMPLETE,
            value=None,
            key=service,
        )
        asyncio.run_coroutine_threadsafe(
            self.service_events_queue.put(event), self._loop
        )

    def _on_state_event(self, service: str, event) -> None:
        key = event.key
        if _is_desired_key(key):
            kind = EventKind.DESIRED
        elif _is_status_key(key):
            kind = EventKind.STATUS
        else:
            # raise ValueError(f"Unexpected state key: {key}")
            self.logger.warning(f"Unexpected state key: {key}")
            return

        try:
            node, depl = self._parse_kv_key(key)
        except Exception as e:
            self.logger.warning(f"Unexpected state key: {key}")
            return

        se = StateEvent(
            kind=kind,
            node=node,
            deployment=depl,
            ev_type=event.event,
            value=event.value,
            key=key,
        )
        asyncio.run_coroutine_threadsafe(self.state_events_queue.put(se), self._loop)

    def _schedule_rebuild(self, update_gateway: bool = True) -> None:
        """Coalesce discovery changes without cancelling an active update."""
        self._rebuild_requested = True
        if self._rebuild_task and not self._rebuild_task.done():
            return

        async def _rebuilder():
            try:
                while self._rebuild_requested:
                    await asyncio.sleep(self._debounce_s)
                    self._rebuild_requested = False
                    await self._refresh_discovery_routing(update_gateway)
            except asyncio.CancelledError:
                raise
            except Exception as ex:
                if _is_known_connection_error(ex):
                    self.logger.debug(
                        f"Rebuild deferred (etcd unavailable): {type(ex).__name__}"
                    )
                else:
                    self.logger.error(f"Rebuild error: {ex}", exc_info=True)

        self._rebuild_task = asyncio.create_task(_rebuilder())

    async def _refresh_discovery_routing(self, update_gateway: bool) -> None:
        async with self._streamer_update_lock:
            self.logger.debug("Refreshing deployment routing...")
            ClusterState.deployment_nodes = self._routable_deployment_nodes()
            ClusterState.notify_deployment_update()
            if update_gateway:
                await self.update_gateway_streamer()

    async def process_service_events(self, max_errors=5) -> None:
        configured_workers = int(
            getattr(self, "args", {}).get(
                "discovery_event_worker_count", DEFAULT_SERVICE_EVENT_WORKERS
            )
        )
        worker_count = max(1, min(configured_workers, 64))
        self.logger.info(
            "Starting service discovery event workers: count=%d", worker_count
        )
        worker_queues = [asyncio.Queue(maxsize=64) for _ in range(worker_count)]
        workers = [
            asyncio.create_task(
                self._service_event_worker(queue, max_errors),
                name=f"gateway-service-event-{index}",
            )
            for index, queue in enumerate(worker_queues)
        ]

        try:
            while True:
                event: ServiceEvent = await self.service_events_queue.get()
                try:
                    if event.ev_type == SERVICE_SNAPSHOT_COMPLETE:
                        await self._refresh_discovery_routing(True)
                        await self._publish_capacity_event()
                        if not self.ready_event.is_set():
                            self.ready_event.set()
                            self.logger.info(
                                "Initial service discovery enumerated; gateway control plane is ready"
                            )
                        continue

                    worker_index = hash(event.key) % worker_count
                    await worker_queues[worker_index].put(event)
                finally:
                    self.service_events_queue.task_done()
        finally:
            for worker in workers:
                worker.cancel()
            retry_tasks = list(self._service_retry_tasks.values())
            for retry_task in retry_tasks:
                retry_task.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            await asyncio.gather(*retry_tasks, return_exceptions=True)

    async def _service_event_worker(
        self, queue: asyncio.Queue, max_errors: int
    ) -> None:
        error_count = 0
        while True:
            event: ServiceEvent = await queue.get()
            readiness = None
            routing_nodes_before = self._routable_deployment_nodes()
            try:
                if event.ev_type == "put":
                    readiness = self._service_readiness_entry(event)
                    readiness["state"] = "checking"
                    result = await self.gateway_server_online(
                        event.service, event.value
                    )
                    if result is None:
                        readiness.update(
                            {
                                "ready": False,
                                "state": "unready",
                                "last_checked_at": _now_iso(),
                                "last_error": "gRPC health check did not return SERVING",
                            }
                        )
                        self._schedule_service_retry(event)
                        changed = False
                    else:
                        checked_at = _now_iso()
                        readiness.update(
                            {
                                "ready": True,
                                "state": "ready",
                                "retry_attempt": 0,
                                "last_checked_at": checked_at,
                                "last_ready_at": checked_at,
                                "next_retry_at": None,
                                "last_error": None,
                            }
                        )
                        self._cancel_service_retry(event.key)
                        changed = result
                elif event.ev_type == "delete":
                    self._cancel_service_retry(event.key)
                    self._service_readiness.pop(event.key, None)
                    changed = await self.gateway_server_offline(event.key, event.value)
                else:
                    self.logger.warning(f"Unknown service ev_type: {event.ev_type}")
                    changed = False

                if self.ready_event.is_set() and (
                    changed or routing_nodes_before != self._routable_deployment_nodes()
                ):
                    self._schedule_rebuild(True)
                    await self._publish_capacity_event()
                error_count = 0
            except asyncio.CancelledError:
                raise
            except Exception as ex:
                if readiness is not None:
                    readiness.update(
                        {
                            "ready": False,
                            "state": "error",
                            "last_checked_at": _now_iso(),
                            "next_retry_at": None,
                            "last_error": str(ex),
                        }
                    )
                if (
                    self.ready_event.is_set()
                    and routing_nodes_before != self._routable_deployment_nodes()
                ):
                    self._schedule_rebuild(True)
                    await self._publish_capacity_event()
                self.logger.error(f"Service event error: {ex}", exc_info=True)
                error_count += 1
                if error_count >= max_errors:
                    self.logger.error(
                        f"Service worker reached {max_errors} consecutive errors"
                    )
                    error_count = 0
                await asyncio.sleep(1)
            finally:
                queue.task_done()

    def _schedule_service_retry(self, event: ServiceEvent) -> None:
        existing = self._service_retry_tasks.get(event.key)
        if existing is not None and not existing.done():
            existing.cancel()

        attempt = self._service_retry_attempts.get(event.key, 0) + 1
        self._service_retry_attempts[event.key] = attempt
        delay = min(5 * (2 ** min(attempt - 1, 4)), 60)
        delay += (abs(hash(event.key)) % 1000) / 1000
        next_retry_at = (
            (datetime.now(timezone.utc) + timedelta(seconds=delay))
            .isoformat()
            .replace("+00:00", "Z")
        )

        readiness = self._service_readiness_entry(event)
        readiness.update(
            {
                "ready": False,
                "state": "retrying",
                "retry_attempt": attempt,
                "next_retry_at": next_retry_at,
            }
        )

        async def _retry() -> None:
            try:
                await asyncio.sleep(delay)
                await self.service_events_queue.put(event)
            finally:
                current = self._service_retry_tasks.get(event.key)
                if current is asyncio.current_task():
                    self._service_retry_tasks.pop(event.key, None)

        self.logger.info(
            "Scheduling service readiness retry: key=%s attempt=%d delay=%.1fs",
            event.key,
            attempt,
            delay,
        )
        self._service_retry_tasks[event.key] = asyncio.create_task(_retry())

    def _service_readiness_entry(self, event: ServiceEvent) -> dict[str, Any]:
        entry = self._service_readiness.get(event.key)
        if entry is not None:
            return entry

        json_address = JsonAddress.from_value(event.value)
        address = _netloc(json_address._addr)
        metadata = json.loads(json_address._metadata)
        parsed_address = urlparse(f"//{address}")
        entry = {
            "key": event.key,
            "service": event.service,
            "address": address,
            "host": parsed_address.hostname,
            "port": parsed_address.port,
            "registered": True,
            "ready": False,
            "state": "checking",
            "retry_attempt": self._service_retry_attempts.get(event.key, 0),
            "probe_attempt_limit": 3,
            "first_seen_at": _now_iso(),
            "last_checked_at": None,
            "last_ready_at": None,
            "next_retry_at": None,
            "last_error": None,
            "executors": sorted(metadata),
        }
        self._service_readiness[event.key] = entry
        return entry

    def _discovery_readiness_snapshot(
        self,
        state: Optional[str] = None,
    ) -> dict[str, Any]:
        gateways = [dict(entry) for entry in self._service_readiness.values()]
        ready_count = sum(1 for entry in gateways if entry["ready"])
        summary = {
            "registered": len(gateways),
            "ready": ready_count,
            "unready": len(gateways) - ready_count,
            "checking": sum(1 for entry in gateways if entry["state"] == "checking"),
            "retrying": sum(1 for entry in gateways if entry["state"] == "retrying"),
            "error": sum(1 for entry in gateways if entry["state"] == "error"),
        }

        if state == "unready":
            gateways = [entry for entry in gateways if not entry["ready"]]
        elif state is not None:
            gateways = [entry for entry in gateways if entry["state"] == state]

        if not self.ready_event.is_set():
            readiness = "initializing"
        elif summary["unready"]:
            readiness = "degraded"
        else:
            readiness = "ready"

        gateways.sort(key=lambda entry: (entry["address"], entry["key"]))
        return {
            "readiness": readiness,
            "control_plane_ready": self.ready_event.is_set(),
            "observed_at": _now_iso(),
            "summary": summary,
            "gateways": gateways,
        }

    async def _operational_health_snapshot(self) -> dict[str, Any]:
        observed_at = _now_iso()
        dependencies: list[dict[str, Any]] = []

        try:
            database = await self.job_scheduler.diagnostics.database_health()
            pool = database["pool"]
            waiters = pool.get("waiters") or 0
            blocked = database.get("blocked_sessions") or 0
            database_state = "degraded" if waiters or blocked else "ok"
            dependencies.append(
                {
                    "name": "postgresql",
                    "state": database_state,
                    "latency_ms": database["latency_ms"],
                    "observed_at": observed_at,
                    "summary": (
                        "connection pool or database sessions are waiting"
                        if database_state == "degraded"
                        else "scheduler database is reachable"
                    ),
                    "details": {
                        "schema_version": database["schema_version"],
                        "pool_used": pool.get("used"),
                        "pool_size": pool.get("size"),
                        "pool_maximum": pool.get("maximum"),
                        "pool_waiters": pool.get("waiters"),
                        "active_sessions": database.get("active_sessions"),
                        "blocked_sessions": database.get("blocked_sessions"),
                        "oldest_transaction_seconds": database.get(
                            "oldest_transaction_seconds"
                        ),
                    },
                }
            )
        except Exception as error:
            self.logger.warning(
                "Operational PostgreSQL health probe failed: %s", type(error).__name__
            )
            dependencies.append(
                {
                    "name": "postgresql",
                    "state": "bad",
                    "latency_ms": None,
                    "observed_at": observed_at,
                    "summary": "scheduler database health probe failed",
                    "details": {},
                }
            )

        etcd_state_value = getattr(
            self.etcd_client.get_connection_state(), "value", None
        )
        etcd_state = str(etcd_state_value or "unknown").lower()
        watch_stats = self.etcd_client.get_watch_stats()
        etcd_health = {
            "connected": "ok",
            "reconnecting": "degraded",
            "disconnected": "degraded",
            "failed": "bad",
        }.get(etcd_state, "degraded")
        dependencies.append(
            {
                "name": "etcd",
                "state": etcd_health,
                "latency_ms": None,
                "observed_at": observed_at,
                "summary": f"etcd client is {etcd_state}",
                "details": {
                    "connection_state": etcd_state,
                    "active_watches": watch_stats.get("active_watches"),
                    "event_queue_size": watch_stats.get("event_queue_size"),
                    "reconnect_attempts": getattr(
                        self.etcd_client, "_reconnect_attempts", None
                    ),
                    "last_success_age_seconds": max(
                        0.0,
                        time.time()
                        - float(
                            getattr(
                                self.etcd_client,
                                "_last_successful_operation",
                                time.time(),
                            )
                        ),
                    ),
                },
            }
        )

        discovery = self._discovery_readiness_snapshot()
        discovery_summary = discovery["summary"]
        discovery_state = {
            "ready": "ok",
            "degraded": "degraded",
            "initializing": "degraded",
        }.get(discovery["readiness"], "bad")
        dependencies.append(
            {
                "name": "discovery",
                "state": discovery_state,
                "latency_ms": None,
                "observed_at": discovery["observed_at"],
                "summary": f"service discovery is {discovery['readiness']}",
                "details": {
                    "control_plane_ready": discovery["control_plane_ready"],
                    "registered": discovery_summary["registered"],
                    "ready": discovery_summary["ready"],
                    "unready": discovery_summary["unready"],
                    "checking": discovery_summary["checking"],
                    "retrying": discovery_summary["retrying"],
                    "errors": discovery_summary["error"],
                },
            }
        )

        gateway_state = "ok"
        if not self.job_scheduler.running:
            gateway_state = "bad"
        elif not self.ready_event.is_set() or self.job_scheduler.paused:
            gateway_state = "degraded"
        dependencies.append(
            {
                "name": "gateway",
                "state": gateway_state,
                "latency_ms": None,
                "observed_at": observed_at,
                "summary": (
                    "gateway control plane is ready"
                    if gateway_state == "ok"
                    else "gateway control plane is not fully available"
                ),
                "details": {
                    "gateway_instance_id": self.gateway_instance_id,
                    "control_plane_ready": self.ready_event.is_set(),
                    "scheduler_running": self.job_scheduler.running,
                    "scheduler_paused": self.job_scheduler.paused,
                    "service_event_queue_size": self.service_events_queue.qsize(),
                    "state_event_queue_size": self.state_events_queue.qsize(),
                },
            }
        )

        rank = {"ok": 0, "degraded": 1, "bad": 2}
        overall = max(dependencies, key=lambda item: rank[item["state"]])["state"]
        return {
            "schema_version": "1.0",
            "generated_at": observed_at,
            "overall_state": overall,
            "partial": any(item["state"] == "bad" for item in dependencies),
            "dependencies": dependencies,
        }

    def _cancel_service_retry(self, key: str) -> None:
        retry_task = self._service_retry_tasks.pop(key, None)
        if retry_task is not None and not retry_task.done():
            retry_task.cancel()
        self._service_retry_attempts.pop(key, None)

    async def process_state_events(self, max_errors=10) -> None:
        error_counter = 0
        while True:
            ev: StateEvent = await self.state_events_queue.get()
            try:
                routing_nodes_before = self._routable_deployment_nodes()
                status_changed = False
                if ev.kind == EventKind.DESIRED:
                    if ev.ev_type == "delete":
                        self.desired_map.pop((ev.node, ev.deployment), None)
                    else:
                        self.desired_map[(ev.node, ev.deployment)] = (
                            self._normalize_desired_event(
                                ev.node, ev.deployment, ev.value or {}
                            )
                        )
                    ClusterState.desired = self.desired_map

                elif ev.kind == EventKind.STATUS:
                    key = (ev.node, ev.deployment)
                    previous_status = self.status_map.get(key)
                    if ev.ev_type == "delete":
                        self.status_map.pop(key, None)
                        current_status = None
                    else:
                        current_status = self._normalize_status_event(
                            ev.node, ev.deployment, ev.value or {}
                        )
                        self.status_map[key] = current_status
                    status_changed = self._status_identity(
                        previous_status
                    ) != self._status_identity(current_status)
                    ClusterState.status = self.status_map

                else:
                    self.logger.warning(f"Ignoring unexpected state kind: {ev.kind}")
                    raise ValueError(f"Unexpected state kind : {ev.kind}")

                if status_changed:
                    ClusterState.notify_deployment_update()
                if routing_nodes_before != self._routable_deployment_nodes():
                    self._schedule_rebuild(True)
                    await self._publish_capacity_event()
                error_counter = 0

            except Exception as ex:
                self.logger.error(f"State event error: {ex}", exc_info=True)
                error_counter += 1
                if error_counter >= max_errors:
                    self.logger.error(f"State loop reached max errors: {max_errors}")
                    break
                await asyncio.sleep(0.5)
            finally:
                self.state_events_queue.task_done()

    def _normalize_desired_event(self, node: str, depl: str, value: dict) -> DesiredDoc:
        """
        Incoming DESIRED event can be a dict or JSON string at:
          { "<node>": { "<depl>": { ...DesiredDoc... } } }
        """
        inner = (value or {}).get(node, {})
        raw = inner.get(depl, inner.get("desired", inner))

        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode()
        if isinstance(raw, str):
            try:
                data = json.loads(raw)
            except Exception:
                data = {}
        elif isinstance(raw, dict):
            data = raw
        else:
            data = {}

        return DesiredDoc(
            phase=str(data.get("phase") or "SCHEDULED"),
            epoch=int(data.get("epoch") or 0),
            params=dict(data.get("params") or {}),
            updated_at=str(data.get("updated_at") or _now_iso()),
        )

    def _normalize_status_event(self, node: str, depl: str, value: dict) -> StatusDoc:
        """
        Incoming STATUS event shape (examples):
          { "<node>": { "<depl>": { "status": "<json-string>" } } }
          { "<node>": { "<depl>": { "status": { ... } } } }
        We always return a StatusDoc.
        """
        inner = (value or {}).get(node, {})
        exec_map = inner.get(depl, {})
        raw_status = exec_map.get("status", exec_map)  # tolerate both shapes

        if isinstance(raw_status, (bytes, bytearray)):
            raw_status = raw_status.decode()
        if isinstance(raw_status, str):
            try:
                payload = json.loads(raw_status)
            except Exception:
                payload = {}
        elif isinstance(raw_status, dict):
            payload = raw_status
        else:
            payload = {}

        # Fill required StatusDoc fields with safe defaults if missing
        status_code = payload.get("status_code")
        if status_code is None and "status_name" in payload:
            status_code = _status_code(payload["status_name"])
        if status_code is None:
            status_code = HealthCheckResponse.ServingStatus.SERVICE_UNKNOWN

        status_name = payload.get("status_name") or _status_name(status_code)

        return StatusDoc(
            status_code=int(status_code),
            status_name=str(status_name),
            owner=str(payload.get("owner") or ""),
            epoch=int(payload.get("epoch") or 0),
            updated_at=str(payload.get("updated_at") or _now_iso()),
            heartbeat_at=str(payload.get("heartbeat_at") or _now_iso()),
            details=payload.get("details"),
        )

    @staticmethod
    def _status_identity(status: StatusDoc | dict | None) -> tuple[object, ...]:
        if status is None:
            return ()
        if isinstance(status, StatusDoc):
            return (
                status.status_code,
                status.status_name,
                status.owner,
                status.epoch,
            )
        return (
            status.get("status_code"),
            status.get("status_name"),
            status.get("owner"),
            status.get("epoch"),
        )

    async def gateway_server_online(
        self, service: str, event_value: dict[str, Any]
    ) -> bool | None:
        """
        Handle the event when a gateway server comes online.

        :param service: The name of the service that is available.
        :param event_value: The value of the event that triggered the method.
        :return: Whether discovery changed, or ``None`` when the gateway is not ready.

        This method is used to handle the event when a gateway server comes online. It checks if the gateway server is ready and then discovers all executors from the gateway. It updates the gateway streamer with the discovered nodes.

        """
        self.logger.info(f"Service is available : {service} @ {event_value}")

        # convert event_value to JsonAddress
        json_address = JsonAddress.from_value(event_value)
        ctrl_address = json_address._addr
        metadata = json.loads(json_address._metadata)

        max_tries = 3
        tries = 0
        is_ready = False

        while tries < max_tries:
            self.logger.info(
                f"checking is ready at {ctrl_address}  (try {tries + 1}/{max_tries})"
            )
            is_ready = await GRPCServer.async_is_ready(ctrl_address)
            self.logger.info(f"gateway status: {is_ready}")
            if is_ready:
                break
            await asyncio.sleep(1)
            tries += 1

        if is_ready is False:
            self.logger.warning(
                f"Gateway is not ready at {ctrl_address} after {max_tries} attempts"
            )
            return None

        self.logger.info(f"Gateway is ready at {ctrl_address}")
        # discover all executors from the gateway
        # stub =  jina_pb2_grpc.JinaDiscoverEndpointsRPCStub(GRPCServer.get_channel(ctrl_address))
        TLS_PROTOCOL_SCHEMES = ["grpcs"]

        parsed_address = urlparse(ctrl_address)
        address = parsed_address.netloc if parsed_address.netloc else ctrl_address
        use_tls = parsed_address.scheme in TLS_PROTOCOL_SCHEMES
        channel_options = None
        timeout = 1

        changed = False
        for executor, deployment_addresses in metadata.items():
            for deployment_address in deployment_addresses:
                endpoints = []
                tries = 0
                while tries < max_tries:
                    try:
                        async with get_grpc_channel(
                            address,
                            tls=use_tls,
                            root_certificates=None,
                            options=channel_options,
                            asyncio=True,
                        ) as channel:
                            stub = jina_pb2_grpc.JinaDiscoverEndpointsRPCStub(channel)
                            response = await stub.endpoint_discovery(
                                jina_pb2.google_dot_protobuf_dot_empty__pb2.Empty(),
                                timeout=timeout,
                                metadata=(),
                            )
                            self.logger.info(f"response: {response.endpoints}")
                            endpoints = response.endpoints
                            break
                    except grpc.RpcError as e:
                        await asyncio.sleep(1)
                        tries += 1
                        if (
                            e.code()
                            not in (
                                grpc.StatusCode.UNAVAILABLE,
                                grpc.StatusCode.DEADLINE_EXCEEDED,
                            )
                            or tries >= max_tries
                        ):
                            raise

                for endpoint in endpoints:
                    if executor not in self.deployment_nodes:
                        self.deployment_nodes[executor] = []
                    deployment_details = {
                        "address": deployment_address,
                        "endpoint": endpoint,
                        "executor": executor,
                        "gateway": ctrl_address,
                    }
                    # Prevent duplicate entries on reconnection
                    existing = self.deployment_nodes[executor]
                    is_duplicate = any(
                        n.get("address") == deployment_address
                        and n.get("endpoint") == endpoint
                        for n in existing
                    )
                    if is_duplicate:
                        self.logger.debug(
                            f"Skipping duplicate endpoint: {executor} : {deployment_details}"
                        )
                        continue
                    self.deployment_nodes[executor].append(deployment_details)
                    changed = True
                    self.logger.debug(
                        f"Discovered endpoint: {executor} : {deployment_details}"
                    )

        for executor, nodes in self.deployment_nodes.items():
            self.logger.debug(
                f"Discovered nodes for executor : {executor}, {len(nodes)}"
            )
            for node in nodes:
                self.logger.debug(f"\tNode : {node}")

        return changed

    async def update_gateway_streamer(self):
        """Update the gateway streamer with the discovered executors."""
        self.logger.info("Updating gateway streamer")
        # TODO : We can only do one Executor for now, need to update this to handle multiple executors
        # Graph here is just a simple start-gateway -> executor -> end-gateway representation of the deployment
        # it does not care if the executor is a Flow or a Deployment or if nodes are present in the executor
        # this allows us to use same gateway streamer for all types of deployments

        # {
        #     "start-gateway": ["executor0","extract_executor"],
        #     "executor0": ["end-gateway"],
        #     "extract_executor": ["end-gateway"]
        # }

        routing_nodes = self._routable_deployment_nodes()
        executors_ = sorted(routing_nodes)
        graph_description = {
            "start-gateway": executors_,
        }
        for executor in executors_:
            graph_description[executor] = ["end-gateway"]

        self.logger.info(f"graph_description: {graph_description}")

        deployments_metadata = {"deployment0": {"key": "value"}}
        deployments_addresses = {}
        for executor, nodes in routing_nodes.items():
            connections = []
            for node in nodes:
                address = node["address"]
                parsed_address = urlparse(address)
                port = parsed_address.port
                host = parsed_address.hostname
                connections.append(f"{host}:{port}")
            deployments_addresses[executor] = sorted(set(connections))

        self.logger.info(f"graph_description: {graph_description}")
        self.logger.info(f"deployments_addresses: {deployments_addresses}")

        # Check if we can do an incremental update
        if self._can_update_incrementally(graph_description):
            await self._apply_incremental_updates(deployments_addresses)
            self.logger.info("Applied incremental update to gateway streamer")
        else:
            # Full recreation required (topology changed or first time)
            await self._create_new_gateway_streamer(
                graph_description, deployments_addresses, deployments_metadata
            )
            self.logger.info("Created new gateway streamer (full rebuild)")

        self._last_graph_description = graph_description
        self._last_deployments_addresses = deployments_addresses

        self.distributor.deployment_nodes = routing_nodes

        self.logger.debug(f'topology_graph : {self.streamer.topology_graph}')
        self.logger.debug("-----------------------------")
        for node in self.streamer.topology_graph.all_nodes:
            self.logger.debug(node)
            for outgoing in node.outgoing_nodes:
                self.logger.debug(f"\t{outgoing}")

    def _can_update_incrementally(self, new_graph: dict) -> bool:
        """
        Check if we can update incrementally (same topology, different addresses).

        :param new_graph: New graph description
        :return: True if incremental update is possible
        """
        # Need existing streamer
        if not hasattr(self, "streamer") or self.streamer is None:
            return False

        # Need previous graph description
        if not hasattr(self, "_last_graph_description"):
            return False

        # Same executor set = incremental possible
        # Compare executor names (ignoring start-gateway and end-gateway)
        old_executors = set(self._last_graph_description.get("start-gateway", []))
        new_executors = set(new_graph.get("start-gateway", []))

        return old_executors == new_executors

    async def _apply_incremental_updates(self, new_addresses: dict) -> None:
        """
        Apply address changes incrementally without recreating the streamer.

        :param new_addresses: New deployment addresses mapping
        """
        # Update existing deployments
        for deployment, addresses in new_addresses.items():
            previous = set(
                getattr(self, "_last_deployments_addresses", {}).get(deployment, [])
            )
            if previous == set(addresses):
                continue
            await self.streamer.update_executor_addresses(deployment, addresses)

        # Handle removed deployments
        if hasattr(self, "_last_deployments_addresses"):
            current_deployments = set(self._last_deployments_addresses.keys())
            new_deployments = set(new_addresses.keys())

            for removed in current_deployments - new_deployments:
                self.logger.info(f"Removing deployment: {removed}")
                for addr in self._last_deployments_addresses[removed]:
                    await self.streamer.remove_connection(removed, addr)

    async def _create_new_gateway_streamer(
        self,
        graph_description: dict,
        deployments_addresses: dict,
        deployments_metadata: dict,
    ) -> None:
        """
        Create a new gateway streamer (full rebuild).

        :param graph_description: Graph topology description
        :param deployments_addresses: Deployment addresses mapping
        :param deployments_metadata: Deployment metadata
        """
        old_streamer = getattr(self, "streamer", None)
        streamer = GatewayStreamer(
            graph_representation=graph_description,
            executor_addresses=deployments_addresses,
            deployments_metadata=deployments_metadata,
            load_balancer_type=LoadBalancerType.LEAST_CONNECTION.name,
            grpc_channel_options=(
                self.runtime_args.grpc_channel_options
                if hasattr(self.runtime_args, "grpc_channel_options")
                else None
            ),
        )

        self.streamer = streamer
        self.distributor.streamer = streamer

        if old_streamer is not None:
            try:
                await old_streamer.close()
            except Exception as e:
                self.logger.warning(f"Error closing old streamer: {e}")

    async def gateway_server_offline(self, service_key: str, ev_value) -> bool:
        """
        Handle the event when a gateway server goes offline.

        :param service_key: The full service key (e.g., "gateway/marie/192.168.106.75:55698")
        :param ev_value: The value representing the offline gateway (usually None for DELETE).
        """
        # Extract address from full key: "gateway/marie/192.168.106.75:55698" -> "192.168.106.75:55698"
        ctrl_address = service_key.split("/")[-1]
        self.logger.info(
            f"Service {service_key} is offline @ {ctrl_address}, removing nodes"
        )

        removed_count = 0
        for executor, nodes in self.deployment_nodes.items():
            before_len = len(nodes)
            self.deployment_nodes[executor] = [
                node for node in nodes if node["gateway"] != ctrl_address
            ]
            removed_count += before_len - len(self.deployment_nodes[executor])

        if removed_count > 0:
            self.logger.info(
                f"Removed {removed_count} nodes for offline gateway {ctrl_address}"
            )
        else:
            self.logger.debug(f"No nodes found for offline gateway {ctrl_address}")

        return removed_count > 0

    def _parse_kv_key(self, key: str) -> tuple[str, str]:
        # returns (node, deployment)
        # key prefix guaranteed by is_desired_key/is_status_key
        parts = key[len(ROOT) :].split("/")
        # parts = [node, deployment, "desired" | "status"]
        if len(parts) < 3:
            raise ValueError(f"Unexpected key format: {key}")
        node, deployment = parts[0], parts[1]
        return node, deployment

    async def desired_changed(self, ev_key: str, ev_type: str, ev_value: dict):
        try:
            node, deployment = self._parse_kv_key(ev_key)
            key = (node, deployment)
            if ev_type == "delete":
                self.desired_map.pop(key, None)
            else:
                # value is JSON from DesiredStore (e.g., {"phase":"SCHEDULED","epoch":123,...})
                self.desired_map[key] = ev_value
            ClusterState.desired = self.desired_map
        except Exception as e:
            self.logger.error(f"desired_changed error for {ev_key}: {e}")

    async def status_changed(self, ev_key: str, ev_type: str, ev_value: dict):
        try:
            node, deployment = self._parse_kv_key(ev_key)
            key = (node, deployment)
            if ev_type == "delete":
                self.status_map.pop(key, None)
            else:
                # value is JSON from StatusStore (e.g., {"owner":"w1@node","epoch":123,"status":"SERVING","ts":...})
                self.status_map[key] = ev_value
            ClusterState.status = self.status_map
        except Exception as e:
            self.logger.error(f"status_changed error for {ev_key}: {e}")

    def _address_is_registered(self, node: str, depl: str) -> bool:
        for registered_depl, nodes in self.deployment_nodes.items():
            if registered_depl != depl:
                continue
            for n in nodes:
                if _netloc(n.get("address") or "") == node:
                    return True
        return False

    def _desired_params(
        self,
        node: str,
        depl: str,
        desired: Optional[DesiredDoc] = None,
    ) -> Dict[str, Any]:
        doc = desired if desired is not None else self.desired_map.get((node, depl))
        if isinstance(doc, DesiredDoc):
            return doc.params or {}
        if isinstance(doc, dict):
            params = doc.get("params")
            return params if isinstance(params, dict) else {}
        return {}

    def _address_is_live(
        self,
        node: str,
        depl: str,
        desired: Optional[DesiredDoc] = None,
        status: Optional[StatusDoc] = None,
    ) -> bool:
        if not self._address_is_registered(node, depl):
            return False
        if self._desired_params(node, depl, desired).get(STATUS_DEGRADED_SINCE):
            return False

        status_doc = status or self.status_map.get((node, depl))
        if status_doc is None:
            return False

        desired_doc = desired or self.desired_map.get((node, depl))
        desired_epoch = getattr(desired_doc, "epoch", None)
        status_epoch = getattr(status_doc, "epoch", None)
        if isinstance(desired_doc, dict):
            desired_epoch = desired_doc.get("epoch")
        if isinstance(status_doc, dict):
            status_epoch = status_doc.get("epoch")
        return desired_epoch is None or status_epoch == desired_epoch

    def _routable_deployment_nodes(self) -> Dict[str, list[dict[str, Any]]]:
        unready_gateways = {
            entry["address"]
            for entry in self._service_readiness.values()
            if not entry["ready"]
        }
        routable: Dict[str, list[dict[str, Any]]] = {}
        for executor, nodes in self.deployment_nodes.items():
            routable[executor] = [
                node
                for node in nodes
                if (
                    _netloc(node.get("gateway") or "") not in unready_gateways
                    and not self._desired_params(
                        _netloc(node.get("address") or ""), executor
                    ).get(STATUS_DEGRADED_SINCE)
                )
            ]
        return routable

    def _incr_miss_and_maybe_gc(
        self,
        node: str,
        depl: str,
        bump_context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Returns True if GC happened (entry removed)."""

        def incr(params: Dict[str, Any]) -> Dict[str, Any]:
            now = _now_iso()
            params["misses"] = int(params.get("misses", 0)) + 1
            params["missing_since"] = params.get("missing_since") or now
            return params

        updated = self.desired_store.update_params(node, depl, incr)
        if updated is None:
            return False

        params = updated.params or {}
        misses = int(params.get("misses", 0))
        missing_since = str(params.get("missing_since") or updated.updated_at)
        too_old = is_stale(missing_since, MAX_AGE_S)
        node_registered = self._address_is_registered(node, depl)
        node_live = self._address_is_live(node, depl, updated)
        log_extra = {
            **(bump_context or {}),
            "event_type": (
                "gateway_status_reconcile_bump"
                if bump_context
                else "gateway_status_reconcile_miss"
            ),
            "node": node,
            "deployment": depl,
            "current_desired_epoch": updated.epoch,
            "misses": misses,
            "missing_since": missing_since,
            "node_live": node_live,
            "node_registered": node_registered,
            "too_old": too_old,
            "degraded_live_node": bool(params.get(STATUS_DEGRADED_SINCE)),
        }
        self.logger.warning(
            "Status reconcile recorded missing or stale status",
            extra=log_extra,
        )

        if misses >= MAX_MISSES or too_old:
            if node_registered:
                if not params.get(STATUS_DEGRADED_SINCE):

                    def mark_degraded(params: Dict[str, Any]) -> Dict[str, Any]:
                        params[STATUS_DEGRADED_SINCE] = _now_iso()
                        params[STATUS_DEGRADED_REASON] = STATUS_DEGRADED_LIVE_MISSING
                        return params

                    degraded_doc = self.desired_store.update_params(
                        node, depl, mark_degraded
                    )
                    degraded_params = (
                        degraded_doc.params if degraded_doc else {}
                    ) or {}
                    self.logger.warning(
                        "Status degraded for registered node; quarantining from routing",
                        extra={
                            **log_extra,
                            "event_type": "gateway_status_degraded_live_node",
                            "degraded_live_node": True,
                            "status_degraded_since": degraded_params.get(
                                STATUS_DEGRADED_SINCE
                            ),
                            "status_degraded_reason": degraded_params.get(
                                STATUS_DEGRADED_REASON
                            ),
                        },
                    )
                return False
            self.logger.warning(
                f"GC misses={misses}, too_old={too_old}. Deleting desired+status subtree : {node}/{depl}",
                extra={
                    **log_extra,
                    "event_type": "gateway_status_gc_missing_node",
                },
            )
            # remove both desired and status for that node/depl
            base = f"deployments/{node}/{depl}"
            self.etcd_client.delete_prefix(base)
            # bookkeeping maps
            self.desired_map.pop((node, depl), None)
            self.status_map.pop((node, depl), None)
            ClusterState.desired = self.desired_map
            ClusterState.status = self.status_map
            return True
        return False

    def _status_bump_suppressed(
        self,
        node: str,
        depl: str,
        d: DesiredDoc,
        st: Optional[StatusDoc] = None,
        reason: str = "status_missing",
    ) -> bool:
        suppressed = bool(
            self._address_is_registered(node, depl)
            and (d.params or {}).get(STATUS_DEGRADED_SINCE)
        )
        if suppressed:
            params = d.params or {}
            self.logger.debug(
                "Status reconcile bump suppressed for degraded live node",
                extra={
                    "event_type": "gateway_status_bump_suppressed",
                    "node": node,
                    "deployment": depl,
                    "reason": reason,
                    "desired_epoch": d.epoch,
                    "status_epoch": st.epoch if st else None,
                    "misses": int(params.get("misses", 0)),
                    "node_live": False,
                    "node_registered": True,
                    "degraded_live_node": True,
                    "status_degraded_since": params.get(STATUS_DEGRADED_SINCE),
                    "status_degraded_reason": params.get(STATUS_DEGRADED_REASON),
                },
            )
        return suppressed

    def _bump_epoch_for_status_miss(
        self,
        node: str,
        depl: str,
        d: DesiredDoc,
        st: Optional[StatusDoc],
        reason: str,
    ) -> bool:
        params = d.params or {}
        context = {
            "reason": reason,
            "old_desired_epoch": d.epoch,
            "desired_epoch": d.epoch,
            "status_epoch": st.epoch if st else None,
            "misses": int(params.get("misses", 0)),
            "node_live": self._address_is_live(node, depl, d, st),
            "node_registered": self._address_is_registered(node, depl),
            "degraded_live_node": bool(params.get(STATUS_DEGRADED_SINCE)),
            "status_degraded_since": params.get(STATUS_DEGRADED_SINCE),
            "status_degraded_reason": params.get(STATUS_DEGRADED_REASON),
        }
        bumped = self.desired_store.bump_epoch(node, depl)
        if not bumped:
            self.logger.warning(
                "Status reconcile failed to bump desired epoch",
                extra={
                    **context,
                    "event_type": "gateway_status_reconcile_bump_failed",
                    "node": node,
                    "deployment": depl,
                    "new_desired_epoch": None,
                },
            )
            return False

        context["new_desired_epoch"] = bumped.epoch
        return self._incr_miss_and_maybe_gc(node, depl, context)

    def _reset_miss_metadata(self, node: str, depl: str) -> None:
        before = self.desired_store.get(node, depl)
        before_params = (before.params if before else {}) or {}

        def reset(params: Dict[str, Any]) -> Dict[str, Any]:
            params.pop("misses", None)
            params.pop("missing_since", None)
            params.pop(STATUS_DEGRADED_SINCE, None)
            params.pop(STATUS_DEGRADED_REASON, None)
            return params

        updated = self.desired_store.update_params(node, depl, reset)
        if updated:
            self.logger.info(
                "Status reconcile recovered; reset miss metadata",
                extra={
                    "event_type": "gateway_status_reconcile_recovered",
                    "node": node,
                    "deployment": depl,
                    "desired_epoch": updated.epoch,
                    "misses": int(before_params.get("misses", 0)),
                    "node_live": self._address_is_live(node, depl, updated),
                    "node_registered": self._address_is_registered(node, depl),
                    "degraded_live_node": bool(
                        before_params.get(STATUS_DEGRADED_SINCE)
                    ),
                    "status_degraded_since": before_params.get(STATUS_DEGRADED_SINCE),
                    "status_degraded_reason": before_params.get(STATUS_DEGRADED_REASON),
                },
            )

    async def _reconcile_loop(self, interval_s: int = 10) -> None:
        self.logger.debug("Reconcile loop starting (interval=%ss)", interval_s)
        first_pass = True
        while True:
            try:
                self.logger.debug("Reconciling")
                try:
                    label = "boot" if first_pass else "periodic"
                    self.logger.debug(
                        f"[sem] {label} reconcile_all: "
                        f"deleting orphans and fixing counters"
                    )
                    summary = self.semaphore_store.reconcile_all(
                        delete_orphan_holders=True,
                        fix_counters=True,
                    )
                    if first_pass:
                        self.logger.debug(f"[sem] boot reconcile summary: {summary}")
                    else:
                        self.logger.debug(
                            f"[sem] periodic reconcile summary: {summary}"
                        )
                except Exception as e:
                    self.logger.warning(f"[sem] reconcile_all failed (non-fatal): {e}")
                finally:
                    first_pass = False

                self.logger.debug("reconciling: desire/status")

                for node, depl in self.desired_store.list_pairs():
                    self.logger.debug(f" - reconciling {node}/{depl}")
                    d = self.desired_store.get(node, depl)
                    if not d or d.phase != "SCHEDULED":
                        continue

                    st = self.status_store.read(node, depl)
                    if not st:
                        if self._status_bump_suppressed(
                            node, depl, d, reason="status_missing"
                        ):
                            continue
                        if is_stale(d.updated_at, CLAIM_TIMEOUT_S):
                            if self._bump_epoch_for_status_miss(
                                node, depl, d, None, "status_missing"
                            ):
                                continue
                        continue

                    if st.epoch != d.epoch:
                        if self._status_bump_suppressed(
                            node, depl, d, st, "status_epoch_mismatch"
                        ):
                            continue
                        if is_stale(d.updated_at, CLAIM_TIMEOUT_S):
                            if self._bump_epoch_for_status_miss(
                                node, depl, d, st, "status_epoch_mismatch"
                            ):
                                continue
                        continue

                    if is_stale(st.heartbeat_at, HEARTBEAT_TIMEOUT_S):
                        if self._status_bump_suppressed(
                            node, depl, d, st, "status_heartbeat_stale"
                        ):
                            continue
                        if self._bump_epoch_for_status_miss(
                            node, depl, d, st, "status_heartbeat_stale"
                        ):
                            continue
                        continue

                    params = d.params or {}
                    if (
                        "misses" in params
                        or "missing_since" in params
                        or STATUS_DEGRADED_SINCE in params
                        or STATUS_DEGRADED_REASON in params
                    ):
                        self._reset_miss_metadata(node, depl)

            except Exception as e:
                if _is_known_connection_error(e):
                    self.logger.debug(f"Reconcile loop deferred (etcd unavailable)")
                else:
                    self.logger.error(f"Reconcile loop error: {e}", exc_info=True)
            finally:
                await asyncio.sleep(interval_s)

    async def _start_state_watch(self):
        """
        Watch deployments/<node>/<depl>/(desired|status) directly,
        seed current state once, and stream updates into state_events_queue.
        """
        self.logger.info("Seeding initial state into state_events_queue")
        # Initial seed (best effort
        try:
            for node, depl in self.desired_store.list_pairs():
                d = self.desired_store.get(node, depl)
                if d:
                    await self.state_events_queue.put(
                        StateEvent(
                            kind=EventKind.DESIRED,
                            node=node,
                            deployment=depl,
                            ev_type="put",
                            value={node: {depl: asdict(d)}},
                            key=f"{ROOT}{node}/{depl}/desired",
                        )
                    )
            for node, depl in self.status_store.list_pairs():
                s = self.status_store.read(node, depl)
                if s:
                    await self.state_events_queue.put(
                        StateEvent(
                            kind=EventKind.STATUS,
                            node=node,
                            deployment=depl,
                            ev_type="put",
                            value={node: {depl: {"status": asdict(s)}}},
                            key=f"{ROOT}{node}/{depl}/status",
                        )
                    )
        except Exception as e:
            self.logger.warning(f"Initial state seed failed: {e}")

        def _cb(_svc_unused: str, ev):
            try:
                k = self.etcd_client._demangle_key(ev.key)
                if not (
                    k.startswith(ROOT)
                    and (k.endswith("/desired") or k.endswith("/status"))
                ):
                    return  # ignore anything else under deployments/

                node, depl = self._parse_kv_key(k)
                payload = (
                    next(iter(ev.values.values()))
                    if getattr(ev, "values", None)
                    else {}
                )
                if k.endswith("/desired"):
                    se = StateEvent(
                        kind=EventKind.DESIRED,
                        node=node,
                        deployment=depl,
                        ev_type=ev.event,
                        value={node: {depl: payload}},
                        key=k,
                    )
                else:
                    se = StateEvent(
                        kind=EventKind.STATUS,
                        node=node,
                        deployment=depl,
                        ev_type=ev.event,
                        value={node: {depl: {"status": payload}}},
                        key=k,
                    )
                asyncio.run_coroutine_threadsafe(
                    self.state_events_queue.put(se), self._loop
                )
            except Exception as e:
                self.logger.error(
                    f"state-watch callback error for {getattr(ev, 'key', '?')}: {e}",
                    exc_info=True,
                )

        # Start watching with revision tracking
        self.resolver.watch_service(
            ROOT,
            _cb,
            notify_on_start=False,  # we do our own seeding
        )
