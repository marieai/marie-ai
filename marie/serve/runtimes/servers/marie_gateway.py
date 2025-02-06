import asyncio
import json
import os
import sys
import time
import traceback
from datetime import datetime
from typing import Any, AsyncGenerator, AsyncIterator, Callable, Dict, Optional
from urllib.parse import urlparse

import grpc
from docarray import DocList
from docarray.documents import TextDoc
from fastapi import Depends, Request
from rich.traceback import install

import marie
import marie.helper
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
from marie.messaging import mark_as_failed, mark_as_scheduled
from marie.proto import jina_pb2, jina_pb2_grpc
from marie.scheduler import PostgreSQLJobScheduler
from marie.scheduler.models import DEFAULT_RETRY_POLICY, JobSubmissionModel, WorkInfo
from marie.scheduler.state import WorkState
from marie.serve.discovery import JsonAddress
from marie.serve.discovery.resolver import EtcdServiceResolver
from marie.serve.networking.balancer.interceptor import LoadBalancerInterceptor
from marie.serve.networking.balancer.load_balancer import LoadBalancerType
from marie.serve.networking.balancer.round_robin_balancer import RoundRobinLoadBalancer
from marie.serve.networking.connection_stub import _ConnectionStubs
from marie.serve.networking.utils import get_grpc_channel
from marie.serve.runtimes.gateway.request_handling import GatewayRequestHandler
from marie.serve.runtimes.gateway.streamer import GatewayStreamer
from marie.serve.runtimes.servers.cluster_state import ClusterState
from marie.serve.runtimes.servers.composite import CompositeServer
from marie.serve.runtimes.servers.grpc import GRPCServer
from marie.storage.kv.psql import PostgreSQLKV
from marie.types_core.request.data import DataRequest, Response
from marie.types_core.request.status import StatusMessage
from marie.utils.server_runtime import setup_auth, setup_storage, setup_toast_events
from marie.utils.types import strtobool


def create_balancer_interceptor() -> LoadBalancerInterceptor:
    def notify(event, connection):
        # print(f"notify: {event}, {connection}")
        pass

    return GatewayLoadBalancerInterceptor(notifier=notify)


def load_env_file(dotenv_path: Optional[str] = None) -> None:
    from dotenv import load_dotenv

    logger.info(f"Loading env file from {dotenv_path}")
    load_dotenv(dotenv_path=dotenv_path, verbose=True)


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
        self.event_queue = asyncio.Queue()
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

        kv_store_kwargs = kwargs["kv_store_kwargs"]
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
        )

        storage = PostgreSQLKV(config=kv_store_kwargs, reset=False)
        job_manager = JobManager(storage=storage, job_distributor=self.distributor)
        self.job_scheduler = PostgreSQLJobScheduler(
            config=job_scheduler_kwargs, job_manager=job_manager
        )

        # perform monkey patching
        GatewayRequestHandler.stream = self.custom_stream
        GatewayRequestHandler.Call = (
            self.custom_stream
        )  # Call is an alias for stream in GatewayRequestHandler
        GatewayRequestHandler.dry_run = self.custom_dry_run

        # FIXME : The resolver watch_service is not implemented correctly
        self.resolver = None

        def _extend_rest_function(app):
            from fastapi import Request

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
                self.logger.info(f"Received request at {datetime.now}")
                work_info = WorkInfo(
                    name="extract",
                    priority=0,
                    data={},
                    state=WorkState.CREATED,
                    retry_limit=0,
                    retry_delay=0,
                    retry_backoff=False,
                    start_after=datetime.now(),
                    expire_in_seconds=0,
                    keep_until=datetime.now(),
                )

                result = await self.job_scheduler.submit_job(work_info)
                return {"result": result}

            @app.get("/check")
            async def get_health(text: str):
                self.logger.info(f"Received request at {datetime.now()}")
                return {"result": "ok"}

            @app.api_route(
                path="/api/jobs/{state}",
                methods=["GET"],
                summary=f"Job listing endpoint /api/jobs",
            )
            async def list_jobs(request: Request):
                self.logger.info(f"Received request at {datetime.now()}")
                params = request.path_params
                state = params.get("state")

                if state:
                    jobs = await self.job_scheduler.list_jobs(state=state)
                else:
                    jobs = await self.job_scheduler.list_jobs()

                return {"status": "OK", "result": jobs}

            @app.api_route(
                path="/api/jobsXX/{job_id}",
                methods=["GET"],
                summary="Stop a job /api/jobs/{job_id}",
            )
            async def get_job_info(request: Request):
                self.logger.info(f"Received request at {datetime.now()}")
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
                self.logger.info(f"Received request at {datetime.now()}")
                return {"status": "OK", "result": "Job stopped"}

            @app.api_route(
                path="/api/jobs/{job_id}",
                methods=["DELETE"],
                summary="Delete a job /api/jobs/{job_id}/stop",
            )
            async def delete_job(request: Request):
                self.logger.info(f"Received request at {datetime.now()}")
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
                self.logger.info(f"Received request at {datetime.now()}")
                self.logger.info(f"Token : {token}")

                payload = await request.json()
                header = payload.get("header", {})
                message = payload.get("parameters", {})

                if "api_key" not in message or message["api_key"] is None:
                    message["api_key"] = token

                req = DataRequest()
                req.parameters = message

                async def caller(req: DataRequest):
                    print(f"Received request: {req}")
                    decoded = await self.decode_request(req)

                    print("Decoded request: ", decoded)
                    if isinstance(decoded, AsyncIterator):
                        async for response in decoded:
                            yield response
                    else:
                        yield decoded

                event_generator = caller(req)
                response = await event_generator.__anext__()
                return {"header": {}, "parameters": response.parameters, "data": None}

                # event_generator = _gen_dict_documents(caller(req))
                # return EventSourceResponse(event_generator)

                # # ['header', 'parameters', 'routes', 'data'
                # return {"header": {}, "parameters": {}, "data": None}

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
        self.logger.info(f"intercepting stream")
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
        self.logger.info(f"Processing request: {request}")
        message = request.parameters

        # print message details
        self.logger.info(f"Message details : {message}")

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
        self.logger.info(f"Handling job action : {action}")

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
        self.logger.info(f"Handling job submit command : {message}")
        silence_exceptions = strtobool(
            os.environ.get("MARIE_SILENCE_EXCEPTIONS", False)
        )

        submission_model = JobSubmissionModel(**message)
        self.logger.info(f"Submission model : {submission_model}")

        metadata = submission_model.metadata
        project_id = metadata.get("project_id", None)
        ref_type = metadata.get("ref_type", None)
        ref_id = metadata.get("ref_id", None)
        submission_policy = metadata.get("policy", None)

        # ensure that project_id, ref_type, ref_id are the metadata of the submission model
        # we need this as this what we will use for Toast events
        if not ref_type or not ref_id:
            return self.error_response(
                "Project ID , Reference Type and Reference ID are required in the metadata",
                None,
            )

        retry = DEFAULT_RETRY_POLICY

        work_info = WorkInfo(
            name=submission_model.name,
            priority=0,
            data=message,
            state=WorkState.CREATED,
            retry_limit=retry.retry_limit,
            retry_delay=retry.retry_delay,
            retry_backoff=retry.retry_backoff,
            start_after=datetime.now(),
            expire_in_seconds=0,
            keep_until=datetime.now(),
            policy=submission_policy,
        )

        try:
            job_id = await self.job_scheduler.submit_job(work_info)

            response = Response()
            response.parameters = {
                "status": "ok",
                "msg": f"job submitted with id {job_id}",
                "job_id": job_id,
            }
            await mark_as_scheduled(
                api_key=project_id,
                job_id=job_id,
                event_name=work_info.name,
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
                detail = "Internal Server Error"
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

        setup_toast_events(self.args.get("toast", {}))
        setup_storage(self.args.get("storage", {}))
        setup_auth(self.args.get("auth", {}))

        await self.job_scheduler.start()
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

        # task for processing events
        run_server_tasks.append(asyncio.create_task(self.process_events(max_errors=5)))
        await asyncio.gather(*run_server_tasks)

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
                    etcd_host,
                    etcd_port,
                    namespace="marie",
                    start_listener=False,
                    listen_timeout=watchdog_interval,
                )

                self.resolver.watch_service(
                    service_name, self.handle_discovery_event, notify_on_start=True
                )
                self.resolver.watch_service(
                    DEPLOYMENT_STATUS_PREFIX,
                    self.handle_discovery_event,
                    notify_on_start=True,
                )
            except Exception as e:
                self.logger.error(
                    f"Failed to initialize etcd client on {etcd_host}:{etcd_port}"
                )
                if isinstance(e, RuntimeFailToStart):
                    raise e
                raise RuntimeFailToStart(
                    f"Initialize etcd client failed on {etcd_host}:{etcd_port}, ensure the etcd server is running.",
                    details=str(e),
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

    async def process_events(self, max_errors=5) -> None:
        """
        Handle a discovery event.
        :param max_errors: The maximum number of errors to allow before stopping the event processing.
        :return: None
        """

        error_counter = 0
        gateway_changed = False

        while True:
            service, event = await self.event_queue.get()
            try:
                self.logger.info(
                    f"Queue size : {self.event_queue.qsize()} service = {service}, event = {event}"
                )

                ev_key = event.key
                ev_type = event.event
                ev_value = event.value

                if ev_key.startswith(DEPLOYMENT_STATUS_PREFIX):
                    await self.deployment_changed(ev_key, ev_type, ev_value)
                else:
                    gateway_changed = True
                    if ev_type == "put":
                        await self.gateway_server_online(service, ev_value)
                    elif ev_type == "delete":
                        self.logger.info(f"Service {service} is unavailable")
                        await self.gateway_server_offline(service, ev_value)
                    else:
                        raise TypeError(f"Not recognized event type : {ev_type}")
                    error_counter = 0  # reset error counter on successful processing

                # if there are no more events, update the gateway streamer to reflect the changes
                if self.event_queue.qsize() == 0 and gateway_changed:
                    ClusterState.deployment_nodes = self.deployment_nodes
                    await self.update_gateway_streamer()
                    gateway_changed = False

            except Exception as ex:
                raise ex
                self.logger.error(f"Error processing event: {ex}")
                error_counter += 1
                if error_counter >= max_errors:
                    self.logger.error(f"Reached maximum error limit: {max_errors}")
                    break
                await asyncio.sleep(1)
            finally:
                self.event_queue.task_done()

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

    async def gateway_server_online(self, service, event_value):
        """
        Handle the event when a gateway server comes online.

        :param service: The name of the service that is available.
        :param event_value: The value of the event that triggered the method.
        :return: None

        This method is used to handle the event when a gateway server comes online. It checks if the gateway server is ready and then discovers all executors from the gateway. It updates the gateway streamer with the discovered nodes.

        """
        self.logger.info(f"Service {service} is available @ {event_value}")

        # convert event_value to JsonAddress
        json_address = JsonAddress.from_value(event_value)
        ctrl_address = json_address._addr
        metadata = json.loads(json_address._metadata)

        self.logger.info(f"JsonAddress : {ctrl_address}, {metadata}")

        max_tries = 10
        tries = 0
        is_ready = False

        while tries < max_tries:
            self.logger.info(f"checking is ready at {ctrl_address}")
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
                            e.code() != grpc.StatusCode.UNAVAILABLE
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
                    self.logger.info(
                        f"Discovered endpoint: {executor} : {deployment_details}"
                    )

        for executor, nodes in self.deployment_nodes.items():
            self.logger.info(
                f"Discovered nodes for executor : {executor}, {len(nodes)}"
            )
            for node in nodes:
                self.logger.info(f"\tNode : {node}")

        # await self.update_gateway_streamer()

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

        print(f"graph_description: {graph_description}")

        # FIXME: testing with only one executor
        deployments_addresses = {}
        graph_descriptionXXXX = {
            "start-gateway": ["executor0"],
            "executor0": ["end-gateway"],
        }

        deployments_metadata = {"deployment0": {"key": "value"}}
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

        load_balancer = RoundRobinLoadBalancer(
            "deployment-gateway",
            self.logger,
            tracing_interceptors=[create_balancer_interceptor()],
        )

        streamer = GatewayStreamer(
            graph_representation=graph_description,
            executor_addresses=deployments_addresses,
            deployments_metadata=deployments_metadata,
            load_balancer_type=LoadBalancerType.ROUND_ROBIN.name,
            load_balancer=load_balancer,
            # aio_tracing_client_interceptors=[create_trace_interceptor()],
        )

        self.streamer = streamer
        self.distributor.streamer = streamer
        self.distributor.deployment_nodes = self.deployment_nodes

        ClusterState.deployments = self.deployments

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
            return  # ignore the root key, need to handle this differently

        suffix = ev_key[len(DEPLOYMENT_STATUS_PREFIX) :]
        if suffix.startswith("/"):
            suffix = suffix[1:]
        address = suffix.split("/", 1)[0]

        if ev_type == 'delete':
            if address in self.deployments:
                del self.deployments[address]
        else:
            deployment = self.parse_deployment_details(address, ev_value)
            self.deployments[address] = deployment

        ClusterState.deployments = self.deployments


class GatewayLoadBalancerInterceptor(LoadBalancerInterceptor):
    def __init__(self, notifier: Optional[Callable] = None):
        super().__init__()
        self.active_connection = None
        self.notifier = notifier

    def notify(self, event: str, connection: _ConnectionStubs):
        """
        :param event: The event that triggered the notification.
        :param connection: The connection that initiated the event.
        :return: None

        """
        if self.notifier:
            self.notifier(event, connection)

    def on_connection_released(self, connection):
        # print(f"on_connection_released: {connection}")
        self.active_connection = None
        self.notify("released", connection)

    def on_connection_failed(self, connection: _ConnectionStubs, exception):
        # print(f"on_connection_failed: {connection}, {exception}")
        self.active_connection = None
        self.notify("failed", connection)

    def on_connection_acquired(self, connection: _ConnectionStubs):
        # print(f"on_connection_acquired: {connection}")
        self.active_connection = connection
        self.notify("acquired", connection)

    def on_connections_updated(self, connections: list[_ConnectionStubs]):
        # print(f"on_connections_updated: {connections}")
        self.notify("updated", connections)

    def get_active_connection(self):
        """
        Get the active connection.
        :return:
        """
        return self.active_connection
