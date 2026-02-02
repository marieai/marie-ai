import asyncio
import json
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, AsyncGenerator, AsyncIterator, Dict, Optional
from urllib.parse import urlparse

import grpc
from docarray import DocList
from docarray.documents import TextDoc
from fastapi import FastAPI, Request
from grpc_health.v1.health_pb2 import HealthCheckResponse
from rich.traceback import install

import marie
import marie.helper
from marie._core.definitions.metadata import JsonMetadataValue
from marie.auth.api_key_manager import APIKeyManager
from marie.auth.auth_bearer import TokenBearer
from marie.constants import (
    DEPLOYMENT_STATUS_PREFIX,
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
from marie.logging_core.predefined import default_logger as logger
from marie.messaging import Toast, mark_as_failed, mark_as_scheduled
from marie.messaging.events import EngineEventData, MarieEvent, MarieEventType
from marie.messaging.grpc_event_broker import GrpcEventBroker
from marie.proto import jina_pb2, jina_pb2_grpc
from marie.scheduler import PostgreSQLJobScheduler
from marie.scheduler.models import DEFAULT_RETRY_POLICY, JobSubmissionModel, WorkInfo
from marie.scheduler.state import WorkState
from marie.scheduler.util import available_slots_by_executor
from marie.serve.discovery import JsonAddress
from marie.serve.discovery.etcd_manager import convert_to_etcd_args, get_etcd_client
from marie.serve.discovery.resolver import EtcdServiceResolver
from marie.serve.networking.balancer.load_balancer import LoadBalancerType
from marie.serve.networking.utils import get_grpc_channel
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
from marie.utils.server_runtime import (
    setup_auth,
    setup_llm_tracking,
    setup_storage,
    setup_toast_events,
)
from marie.utils.types import strtobool

ROOT = "deployments/"

HEARTBEAT_INTERVAL_S = 10  # worker -> status.heartbeat_at
HEARTBEAT_TIMEOUT_S = 3 * HEARTBEAT_INTERVAL_S  # server considers dead
RESCHEDULE_BACKOFF_S = 5  # server backoff before bumping epoch again

CLAIM_TIMEOUT_S = 30  # how long you wait for a claim per epoch
MAX_MISSES = 3  # after 5 failed epochs -> GC
MAX_AGE_S = 30 * 60  # hard age cap (30 min)


class EventKind(str, Enum):
    SERVICE = "SERVICE"
    DESIRED = "DESIRED"
    STATUS = "STATUS"


@dataclass
class ServiceEvent:
    kind: EventKind  # EventKind.SERVICE
    service: str  # resolver’s service name
    ev_type: str  # "put" | "delete"
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
        self.deployments = {}
        self._deployments_lock = asyncio.Lock()
        self.event_queue = asyncio.Queue(maxsize=512)
        self.ready_event = asyncio.Event()

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

        job_scheduler_kwargs = self.args["job_scheduler_kwargs"]
        if not all(key in job_scheduler_kwargs for key in expected_keys):
            raise ValueError(
                f"job_scheduler_kwargs must contain the following keys: {expected_keys}"
            )

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

        job_manager = JobManager(
            storage=storage,
            job_distributor=self.distributor,
            etcd_client=self.etcd_client,
        )
        self.job_scheduler = PostgreSQLJobScheduler(
            config=job_scheduler_kwargs,
            job_manager=job_manager,
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
                await self.job_scheduler.stop()

            @app.api_route(
                path="/job/submit",
                methods=["GET"],
                summary=f"Submit a job /api/submit",
            )
            async def job_submit(text: str):
                now = datetime.now(timezone.utc)
                self.logger.info(f"Received request at {now}")
                work_info = WorkInfo(
                    name="extract",
                    priority=0,
                    data={},
                    state=WorkState.CREATED,
                    retry_limit=0,
                    retry_delay=0,
                    retry_backoff=False,
                    start_after=now,
                    expire_in_seconds=0,
                    keep_until=now + timedelta(days=2),
                    soft_sla=now,
                    hard_sla=now + timedelta(hours=4),
                )

                result = await self.job_scheduler.submit_job(work_info)
                return {"result": result}

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
                    debug_data = self.job_scheduler.debug_info()
                    return {"status": "OK", "result": debug_data}
                except Exception as e:
                    self.logger.error(f"Error getting debug info: {str(e)}")
                    return {
                        "status": "error",
                        "result": f"Failed to get debug info: {str(e)}",
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
                # For testing purposes, we can return a mock response
                # if False:
                #     return {"header": {}, "parameters": {
                #         "job_id" : "12345",
                #     }, "data": None}

                # Parse request payload with error handling
                try:
                    payload = await request.json()
                except Exception as e:
                    self.logger.error(f"Failed to parse JSON payload: {str(e)}")
                    raise HTTPException(status_code=400, detail="Invalid JSON payload")

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
                    raise HTTPException(
                        status_code=500, detail="No response generated from processing"
                    )
                except Exception as e:
                    self.logger.error(f"Error processing request: {str(e)}")
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
                Get all deployments information including status and nodes.
                :return:
                """
                self.logger.info(
                    f"Deployments info requested at {datetime.now(timezone.utc)}"
                )
                try:
                    return {
                        "status": "OK",
                        "result": {
                            "deployments": self.deployments,
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
                        },
                    }
                except Exception as e:
                    self.logger.error(f"Error getting capacity info: {str(e)}")
                    return {
                        "status": "error",
                        "result": f"Failed to get capacity info: {str(e)}",
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

            # Register Wasm compilation routes
            register_wasm_routes(app)

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
            decoded = await self.decode_request(request)
            if isinstance(decoded, AsyncIterator):
                async for response in decoded:
                    yield response
            else:
                yield decoded

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
        print('invoke_action : ', invoke_action)

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
        soft_sla = metadata.get("soft_sla", None)
        hard_sla = metadata.get("hard_sla", None)
        retry = DEFAULT_RETRY_POLICY
        event_name = submission_model.name

        if soft_sla is None:
            soft_sla = now
            hard_sla = now + timedelta(hours=4)
        else:
            if isinstance(soft_sla, str):
                soft_sla = datetime.fromisoformat(soft_sla)
                # Ensure timezone-aware: if naive, assume UTC
                if soft_sla.tzinfo is None:
                    soft_sla = soft_sla.replace(tzinfo=timezone.utc)
            if isinstance(hard_sla, str):
                hard_sla = datetime.fromisoformat(hard_sla)
                # Ensure timezone-aware: if naive, assume UTC
                if hard_sla.tzinfo is None:
                    hard_sla = hard_sla.replace(tzinfo=timezone.utc)

        if soft_sla > hard_sla:
            return self.error_response(
                "Soft SLA must be before Hard SLA", None, silence_exceptions
            )

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
            priority=0,  # calculated based of the sla criteria and updated via cron
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
            job_id = await self.job_scheduler.submit_job(work_info)

            response = Response()
            response.parameters = {
                "status": "ok",
                "msg": f"job submitted with id {job_id}",
                "job_id": job_id,
            }
            self.logger.info(f"Job submitted with id {job_id}")
            await mark_as_scheduled(
                api_key=api_key,  # project_id,
                job_id=job_id,
                event_name=event_name,
                job_tag=ref_type,
                status="OK",
                timestamp=int(time.time()),
                payload=metadata,
            )

            return response
        except BaseException as ex:
            response = self.error_response(
                f"Failed to submit job. {ex}", ex, silence_exceptions
            )
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
                    timestamp=int(time.time()),
                    payload=exc_msg,
                )
            except Exception as e:
                self.logger.error(f"Failed to mark job as failed: {e}")
            return response
        finally:
            elapsed_time = time.time() - start_time
            self.logger.debug(f"Job submission completed in {elapsed_time:.2f} seconds")

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
        setup_llm_tracking(self.args.get("llm_tracking", {}), storage_config)

        await self.setup_service_discovery(
            etcd_host=self.args["discovery_host"],
            etcd_port=self.args["discovery_port"],
            service_name=self.args["discovery_service_name"],
        )

    async def run_server(self):
        """Run servers inside CompositeServer forever"""
        run_server_tasks = []
        for server in self.servers:
            run_server_tasks.append(asyncio.create_task(server.run_server()))

        # task for processing events and scheduler
        run_server_tasks.append(
            asyncio.create_task(self.process_service_events(max_errors=5))
        )
        run_server_tasks.append(
            asyncio.create_task(self.process_state_events(max_errors=10))
        )
        run_server_tasks.append(
            asyncio.create_task(self.wait_and_start_scheduler(timeout=5))
        )
        run_server_tasks.append(
            asyncio.create_task(self._reconcile_loop(interval_s=10))
        )
        run_server_tasks.append(
            asyncio.create_task(self._capacity_broadcast_loop(interval_s=5))
        )

        await asyncio.gather(*run_server_tasks)

    async def _publish_capacity_event(self) -> None:
        """Publish current capacity state as an event."""
        try:
            rows, totals, summary = self.capacity_manager.refresh_from_nodes(
                self.deployment_nodes
            )
            self.logger.info(summary)

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
            self.logger.error(f"Failed to publish capacity event: {ex}", exc_info=True)

    async def _capacity_broadcast_loop(self, interval_s: float = 5.0) -> None:
        """Periodically broadcast capacity state to connected clients."""
        self.logger.info(f"Starting capacity broadcast loop (interval={interval_s}s)")
        # Wait for gateway to be ready before starting broadcasts
        await asyncio.sleep(interval_s)

        while True:
            try:
                await self._publish_capacity_event()
            except Exception as ex:
                self.logger.error(f"Capacity broadcast error: {ex}", exc_info=True)
            await asyncio.sleep(interval_s)

    async def wait_and_start_scheduler(self, timeout: int = 5):
        """Waits for the service discovery to start and then starts the job scheduler."""
        self.logger.info(f"Waiting for ready_event with a timeout of {timeout} seconds")

        for remaining in range(timeout, 0, -1):
            if self.ready_event.is_set():
                self.logger.info(
                    f"ready_event set, starting job scheduler early (time remaining: {remaining}s)"
                )
                break
            self.logger.info(f"Time remaining: {remaining} seconds")
            await asyncio.sleep(1)
        else:
            if not self.ready_event.is_set():
                self.logger.warning(
                    "Timeout waiting for ready_event, starting scheduler anyway"
                )

        await self.job_scheduler.start()

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
                    service_name, self._on_service_event, notify_on_start=True
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
        """
        Schedule a rebuild of the deployments projection with debouncing.
        Cancels any pending rebuild task and schedules a fresh one.
        :raises asyncio.CancelledError: Raised if the task is canceled during execution.
        """

        # Cancel any pending rebuild and schedule a fresh one
        if self._rebuild_task and not self._rebuild_task.done():
            self.logger.info("Rebuilding deployments canceled...")
            self._rebuild_task.cancel()

        async def _rebuilder():
            try:
                await asyncio.sleep(self._debounce_s)  # coalesce bursts
                # THIS IS THE CRITICAL SECTION AND IT IS MESSYYY
                self.logger.debug("Rebuilding deployments projection...")
                ClusterState.deployment_nodes = self.deployment_nodes
                self._rebuild_deployments_projection()

                if update_gateway:
                    await self.update_gateway_streamer()
                self.ready_event.set()
            except asyncio.CancelledError:
                pass
            except Exception as ex:
                self.logger.error(f"Rebuild error: {ex}", exc_info=True)

        self._rebuild_task = asyncio.create_task(_rebuilder())

    async def process_service_events(self, max_errors=5) -> None:
        error_counter = 0
        while True:
            ev: ServiceEvent = await self.service_events_queue.get()
            try:
                if ev.ev_type == "put":
                    await self.gateway_server_online(ev.service, ev.value)
                elif ev.ev_type == "delete":
                    await self.gateway_server_offline(ev.service, ev.value)
                else:
                    self.logger.warning(f"Unknown service ev_type: {ev.ev_type}")

                # Always schedule a (debounced) rebuild
                self._schedule_rebuild(True)
                # Publish capacity update immediately on service change
                await self._publish_capacity_event()
                error_counter = 0
            except Exception as ex:
                self.logger.error(f"Service event error: {ex}", exc_info=True)
                error_counter += 1
                if error_counter >= max_errors:
                    self.logger.error(f"Service loop reached max errors: {max_errors}")
                    break
                await asyncio.sleep(1)
            finally:
                self.service_events_queue.task_done()

    async def process_state_events(self, max_errors=10) -> None:
        error_counter = 0
        while True:
            ev: StateEvent = await self.state_events_queue.get()
            try:
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
                    if ev.ev_type == "delete":
                        self.status_map.pop((ev.node, ev.deployment), None)
                    else:
                        self.status_map[(ev.node, ev.deployment)] = (
                            self._normalize_status_event(
                                ev.node, ev.deployment, ev.value or {}
                            )
                        )
                    ClusterState.status = self.status_map

                else:
                    self.logger.warning(f"Ignoring unexpected state kind: {ev.kind}")
                    raise ValueError(f"Unexpected state kind : {ev.kind}")

                # Keep legacy projections in sync
                self._rebuild_deployments_projection()
                # self._schedule_rebuild(False)
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

    def parse_deployment_details(self, address: str, ev_value: dict) -> dict:
        # get the first executor from ev_value, we might have multiple executors in the future
        worked_node = list(ev_value.keys())[0]
        worker_info = ev_value[worked_node]
        executor, status = next(iter(worker_info.items()))

        return {
            "prefix": DEPLOYMENT_STATUS_PREFIX,
            "address": address,
            "executor": executor,
            "status": status,
        }

    async def gateway_server_online(self, service: str, event_value: dict[str, Any]):
        """
        Handle the event when a gateway server comes online.

        :param service: The name of the service that is available.
        :param event_value: The value of the event that triggered the method.
        :return: None

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
            time.sleep(1)
            tries += 1

        if is_ready is False:
            self.logger.warning(
                f"Gateway is not ready at {ctrl_address} after {max_tries}, will retry on next event"
            )
            return

        self.logger.info(f"Gateway is ready at {ctrl_address}")
        # discover all executors from the gateway
        # stub =  jina_pb2_grpc.JinaDiscoverEndpointsRPCStub(GRPCServer.get_channel(ctrl_address))
        TLS_PROTOCOL_SCHEMES = ["grpcs"]

        parsed_address = urlparse(ctrl_address)
        address = parsed_address.netloc if parsed_address.netloc else ctrl_address
        use_tls = parsed_address.scheme in TLS_PROTOCOL_SCHEMES
        channel_options = None
        timeout = 1

        for executor, deployment_addresses in metadata.items():
            for deployment_address in deployment_addresses:
                endpoints = []
                tries = 0
                while tries < max_tries:
                    try:
                        with get_grpc_channel(
                            address,
                            tls=use_tls,
                            root_certificates=None,
                            options=channel_options,
                        ) as channel:
                            metadata = ()
                            stub = jina_pb2_grpc.JinaDiscoverEndpointsRPCStub(channel)
                            response, call = stub.endpoint_discovery.with_call(
                                jina_pb2.google_dot_protobuf_dot_empty__pb2.Empty(),
                                timeout=timeout,
                                metadata=metadata,
                            )
                            self.logger.info(f"response: {response.endpoints}")
                            endpoints = response.endpoints
                            break
                    except grpc.RpcError as e:
                        time.sleep(1)
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
                    self.deployment_nodes[executor].append(deployment_details)
                    self.logger.debug(
                        f"Discovered endpoint: {executor} : {deployment_details}"
                    )

        for executor, nodes in self.deployment_nodes.items():
            self.logger.debug(
                f"Discovered nodes for executor : {executor}, {len(nodes)}"
            )
            for node in nodes:
                self.logger.debug(f"\tNode : {node}")

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

        executors_ = list(self.deployment_nodes.keys())
        graph_description = {
            "start-gateway": executors_,
        }
        for executor in executors_:
            graph_description[executor] = ["end-gateway"]

        self.logger.info(f"graph_description: {graph_description}")

        deployments_metadata = {"deployment0": {"key": "value"}}
        deployments_addresses = {}
        for i, (executor, nodes) in enumerate(self.deployment_nodes.items()):
            connections = []
            for node in nodes:
                address = node["address"]
                parsed_address = urlparse(address)
                port = parsed_address.port
                host = parsed_address.hostname
                connections.append(f"{host}:{port}")
            deployments_addresses[executor] = list(set(connections))

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

        # Store for future incremental update checks
        self._last_graph_description = graph_description
        self._last_deployments_addresses = deployments_addresses

        self.distributor.deployment_nodes = self.deployment_nodes

        # Lets get the new deployment topology
        self.logger.info(f'topology_graph : {self.streamer.topology_graph}')
        self.logger.info("-----------------------------")
        for node in self.streamer.topology_graph.all_nodes:
            self.logger.info(node)
            for outgoing in node.outgoing_nodes:
                self.logger.info(f"\t{outgoing}")

        # FIXME : this was a bad idea, we need to use the same deployment
        ClusterState.deployments = self.deployments
        ClusterState.deployment_nodes = self.deployment_nodes

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
        # Close old streamer if exists
        if hasattr(self, "streamer") and self.streamer is not None:
            try:
                await self.streamer.close()
            except Exception as e:
                self.logger.warning(f"Error closing old streamer: {e}")

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

    async def gateway_server_offline(self, service: str, ev_value):
        """
        Handle the event when a gateway server goes offline.

        :param service: The name of the service.
        :param ev_value: The value representing the offline gateway.
        :return: None
        """
        ctrl_address = service.split("/")[-1]
        self.logger.info(
            f"Service {service} is offline @ {ctrl_address}, removing nodes"
        )
        for executor, nodes in self.deployment_nodes.items():
            self.deployment_nodes[executor] = [
                node for node in nodes if node["gateway"] != ctrl_address
            ]

    async def deployment_changed(self, ev_key: str, ev_type: str, ev_value: dict):
        self.logger.info(f"Deployment changed : {ev_key}, {ev_type}, {ev_value}")
        if ev_key == DEPLOYMENT_STATUS_PREFIX:
            return  # ignore the root key

        suffix = ev_key[len(DEPLOYMENT_STATUS_PREFIX) :].lstrip("/")
        address = suffix.split("/", 1)[0]

        # serialize all update
        async with self._deployments_lock:
            new_deployments = self.deployments.copy()
            if ev_type == "delete":
                new_deployments.pop(address, None)
            else:
                new_deployments[address] = self.parse_deployment_details(
                    address, ev_value
                )

            self.deployments = new_deployments
            ClusterState.deployments = new_deployments

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

    def _address_is_live(self, node: str) -> bool:
        # node is "host:port"
        # check against what your discovery currently sees
        for _, nodes in self.deployment_nodes.items():
            for n in nodes:
                # n["address"] is like "grpc://host:port" or "host:port"
                addr = n.get("address") or ""
                if "://" in addr:
                    addr = addr.split("://", 1)[1]
                if addr == node:
                    return True
        return False

    def _incr_miss_and_maybe_gc(self, node: str, depl: str, d: DesiredDoc) -> bool:
        """Returns True if GC happened (entry removed)."""
        # bump 'misses' counter
        params = dict(d.params or {})
        misses = int(params.get("misses", 0)) + 1
        params["misses"] = misses
        d.params = params
        d.updated_at = _now_iso()
        self.desired_store._put_json(
            self.desired_store._desired_key(node, depl), asdict(d)
        )

        # hard age check
        too_old = is_stale(d.updated_at, MAX_AGE_S)

        if misses >= MAX_MISSES or too_old:
            # only GC if the address is not currently live; this avoids killing an actually active worker that’s lagging
            if not self._address_is_live(node):
                self.logger.warning(
                    f"GC misses={misses}, too_old={too_old}. Deleting desired+status subtree : {node}/{depl}"
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

    async def _reconcile_loop(self, interval_s: int = 10) -> None:
        self.logger.debug("Reconcile loop starting (interval=%ss)", interval_s)
        while True:
            try:
                self.logger.debug("Reconciling")
                try:
                    self.logger.debug(
                        "[sem] boot reconcile_all: deleting orphans and fixing counters"
                    )
                    summary = self.semaphore_store.reconcile_all(
                        delete_orphan_holders=True,
                        fix_counters=True,
                    )
                    self.logger.info(f"[sem] boot reconcile summary: {summary}")
                except Exception as e:
                    self.logger.warning(
                        f"[sem] reconcile_all on boot failed (non-fatal): {e}"
                    )

                # Reconcile desire / status
                self.logger.debug("reconciling: desire/status")

                for node, depl in self.desired_store.list_pairs():
                    self.logger.debug(f" - reconciling {node}/{depl}")
                    d = self.desired_store.get(node, depl)
                    if not d or d.phase != "SCHEDULED":
                        continue

                    st = self.status_store.read(node, depl)
                    if not st:
                        if is_stale(d.updated_at, CLAIM_TIMEOUT_S):
                            self.logger.warning(
                                f"No status for {node}/{depl} epoch {d.epoch}; bumping"
                            )
                            self.desired_store.bump_epoch(node, depl)
                            if self._incr_miss_and_maybe_gc(node, depl, d):
                                continue
                        continue

                    if st.epoch != d.epoch:
                        continue

                    if is_stale(st.heartbeat_at, HEARTBEAT_TIMEOUT_S):
                        self.logger.warning(
                            f"Stale status {node}/{depl} epoch {st.epoch}; bumping"
                        )
                        self.desired_store.bump_epoch(node, depl)
                        if self._incr_miss_and_maybe_gc(node, depl, d):
                            continue

            except Exception as e:
                self.logger.error(f"Reconcile loop error: {e}", exc_info=True)
            finally:
                await asyncio.sleep(interval_s)

    def _choose_status_name(self, docs: list[StatusDoc]) -> str:
        """
        SERVING > NOT_SERVING > SERVICE_UNKNOWN > UNKNOWN
        """
        if not docs:
            return "SERVICE_UNKNOWN"
        names = {d.status_name for d in docs if d}
        if "SERVING" in names:
            return "SERVING"
        if "NOT_SERVING" in names:
            return "NOT_SERVING"
        if "SERVICE_UNKNOWN" in names:
            return "SERVICE_UNKNOWN"
        return "UNKNOWN"

    def _rebuild_deployments_projection(self) -> None:
        """
        Recreate self.deployments with the legacy format:
          { "<host:port>": { 'prefix': 'deployments/status',
                             'address': '<host:port>',
                             'executor': '<executor>',
                             'status': '<STATUS_NAME>' } }
        """
        # Index status docs by deployment/executor name
        by_depl: dict[str, list[StatusDoc]] = {}
        for (_, depl), st in self.status_map.items():
            by_depl.setdefault(depl, []).append(st)

        new_deployments: dict[str, dict] = {}
        for executor, nodes in self.deployment_nodes.items():
            # One status per executor (all addresses for that executor get same status)
            status_name = self._choose_status_name(by_depl.get(executor, []))
            # Many nodes share same address but different endpoints; we want 1 entry per address.
            seen_addrs = set()
            for n in nodes:
                addr_raw = n.get("address", "")
                hostport = _netloc(addr_raw)
                if not hostport or hostport in seen_addrs:
                    continue
                seen_addrs.add(hostport)

                new_deployments[hostport] = {
                    "prefix": "deployments/status",  # keep this literal to match expectations
                    "address": hostport,
                    "executor": executor,
                    "status": status_name,
                }

        # Atomic swap to preserve old behavior
        self.deployments = new_deployments
        ClusterState.deployments = new_deployments
        slots = available_slots_by_executor(self.semaphore_store)
        self.logger.debug(f"Rebuilt deployments; slot summary: {slots}")

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
