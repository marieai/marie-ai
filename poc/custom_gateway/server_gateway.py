import asyncio
import json
import time
from datetime import datetime
from typing import Any, AsyncGenerator, AsyncIterator, Callable, Dict, Optional
from urllib.parse import urlparse

import grpc
from docarray import DocList
from docarray.documents import TextDoc

import marie
import marie.helper
from marie import Gateway as BaseGateway
from marie.excepts import BadConfigSource, RuntimeFailToStart
from marie.helper import get_or_reuse_loop
from marie.job.common import JobInfo, JobStatus
from marie.job.gateway_job_distributor import GatewayJobDistributor
from marie.job.job_manager import JobManager
from marie.logging.logger import MarieLogger
from marie.proto import jina_pb2, jina_pb2_grpc
from marie.serve.discovery import JsonAddress
from marie.serve.discovery.resolver import EtcdServiceResolver
from marie.serve.networking.balancer.interceptor import LoadBalancerInterceptor
from marie.serve.networking.balancer.load_balancer import LoadBalancerType
from marie.serve.networking.balancer.round_robin_balancer import RoundRobinLoadBalancer
from marie.serve.networking.connection_stub import _ConnectionStubs
from marie.serve.networking.sse import EventSourceResponse
from marie.serve.networking.utils import get_grpc_channel
from marie.serve.runtimes.gateway.request_handling import GatewayRequestHandler
from marie.serve.runtimes.gateway.streamer import GatewayStreamer
from marie.serve.runtimes.servers.composite import CompositeServer
from marie.serve.runtimes.servers.grpc import GRPCServer
from marie.serve.runtimes.worker.http_fastapi_app import _gen_dict_documents
from marie.storage.kv.psql import PostgreSQLKV
from marie.types.request import Request
from marie.types.request.data import DataRequest, Response
from marie.types.request.status import StatusMessage
from marie_server.scheduler import PostgreSQLJobScheduler
from marie_server.scheduler.models import (
    DEFAULT_RETRY_POLICY,
    JobSubmissionModel,
    WorkInfo,
)
from marie_server.scheduler.state import WorkState


def create_balancer_interceptor() -> LoadBalancerInterceptor:
    def notify(event, connection):
        # print(f"notify: {event}, {connection}")
        pass

    return GatewayLoadBalancerInterceptor(notifier=notify)


class MarieServerGateway(BaseGateway, CompositeServer):
    """A custom Gateway for Marie server. Effectively we are providing a custom implementation of the Gateway class
    that providers communication between individual executors and the server.

    This utilizes service discovery(ETCD) to find deployed Executors from discovered gateways that could have spawned them(Flow/Deployment).

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.logger = MarieLogger(self.__class__.__name__)
        self.logger.info(f"Setting up MarieServerGateway")
        self._loop = get_or_reuse_loop()
        self.deployment_nodes = {}
        self.event_queue = asyncio.Queue()

        if "kv_store_kwargs" not in kwargs:
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

        if "job_scheduler_kwargs" not in kwargs:
            raise BadConfigSource("Missing job_scheduler_kwargs in config")

        job_scheduler_kwargs = kwargs["job_scheduler_kwargs"]
        if not all(key in job_scheduler_kwargs for key in expected_keys):
            raise ValueError(
                f"job_scheduler_kwargs must contain the following keys: {expected_keys}"
            )

        self.distributor = GatewayJobDistributor(
            gateway_streamer=None, logger=self.logger
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
                    on_complete=False,
                )

                result = await self.job_scheduler.submit_job(work_info)
                return {"result": result}

            @app.get("/endpoint")
            async def get(text: str):
                self.logger.info(f"Received request at {datetime.now()}")
                docs = DocList[TextDoc]([TextDoc(text=text)])
                doc = TextDoc(text=text)

                if False:
                    result = await self.distributor.submit_job(
                        JobInfo(status=JobStatus.PENDING, entrypoint="_jina_dry_run_"),
                        doc=doc,
                    )

                    return {"result": result}

                if True:
                    result = None
                    async for docs in self.streamer.stream_docs(
                        docs=DocList[TextDoc]([TextDoc(text=text)]),
                        # doc=TextDoc(text=text),
                        # exec_endpoint="/extract",  # _jina_dry_run_
                        exec_endpoint="_jina_dry_run_",  # _jina_dry_run_
                        # exec_endpoint="/endpoint",
                        # target_executor="executor0",
                        return_results=False,
                    ):
                        result = docs[0].text
                        # result = docs
                        print(f"result: {result}")
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
            )
            async def invoke_command(request: Request):
                self.logger.info(f"Received request at {datetime.now()}")
                payload = await request.json()
                header = payload.get("header", {})
                message = payload.get("parameters", {})

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
            print(f"Decoded request GRPC: {decoded}")
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
        print(f"Processing request: {request}")
        print(request.parameters)
        print(request.data)
        message = request.parameters
        if "invoke_action" not in message:
            response = Response()
            response.parameters = {"error": "Invalid request, missing invoke_action"}
            return response

        invoke_action = message["invoke_action"]
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
        submission_model = JobSubmissionModel(**message)
        self.logger.info(f"Submission model : {submission_model}")
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
            on_complete=False,
        )
        # TODO : convert to using Errors as Values instead of Exceptions
        try:
            job_id = await self.job_scheduler.submit_job(work_info)

            response = Response()
            response.parameters = {
                "status": "ok",
                "msg": f"job submitted with id {job_id}",
                "job_id": job_id,
            }

            return response
        except ValueError as ex:
            return self.error_response(f"Failed to submit job. {ex}", ex)

    def error_response(self, msg: str, exception: Optional[Exception]) -> Response:
        """
        Set the response parameters to indicate a failure.
        :param msg: A string representing the error message.
        :param exception: An optional exception that triggered the error.
        :return: The response object with the error parameters set.
        """
        response = Response()

        exc_msg = ""
        if exception:
            exc_msg = str(exception)

        response.parameters = {"status": "error", "msg": msg, "exception": exc_msg}
        return response

    async def custom_dry_run(self, empty, context) -> jina_pb2.StatusProto:
        print("Running custom dry run logic")

        status_message = StatusMessage()
        status_message.set_code(jina_pb2.StatusProto.SUCCESS)
        return status_message.proto

    async def setup_server(self):
        """
        setup servers inside CompositeServer
        """
        self.logger.debug(f"Setting up MarieGateway server")
        await super().setup_server()
        await self.job_scheduler.start()
        await self.setup_service_discovery(
            etcd_host=self.runtime_args.discovery_host,
            etcd_port=self.runtime_args.discovery_port,
            service_name=self.runtime_args.discovery_service_name,
        )

    async def run_server(self):
        """Run servers inside CompositeServer forever"""
        run_server_tasks = []
        for server in self.servers:
            run_server_tasks.append(asyncio.create_task(server.run_server()))

        # # task for processing events
        run_server_tasks.append(asyncio.create_task(self.process_events(max_errors=5)))
        await asyncio.gather(*run_server_tasks)

    async def setup_service_discovery(
        self,
        etcd_host: str,
        etcd_port: int,
        service_name: str,
        watchdog_interval: int = 2,
    ):
        """
         Setup service discovery for the gateway.

        :param etcd_host: str - The host address of the ETCD service. Default is "0.0.0.0".
        :param etcd_port: int - The port of the ETCD service. Default is 2379.
        :param watchdog_interval: int - The interval in seconds between each service address check. Default is 2.
        :return: None

        """
        self.logger.info(f"Setting up service discovery : {service_name}")
        self.logger.info(f"ETCD host : {etcd_host}:{etcd_port}")

        async def _start_watcher():
            resolver = EtcdServiceResolver(
                etcd_host,
                etcd_port,
                namespace="marie",
                start_listener=False,
                listen_timeout=5,
            )

            self.logger.debug(f"etcd checking : {resolver.resolve(service_name)}")
            resolver.watch_service(service_name, self.handle_discovery_event)

        task = asyncio.create_task(_start_watcher())
        try:
            await task  # This raises an exception if the task had an exception
        except Exception as e:
            self.logger.error(
                f"Initialize etcd client failed failed on {etcd_host}:{etcd_port}"
            )
            if isinstance(e, RuntimeFailToStart):
                raise e
            raise RuntimeFailToStart(
                f"Initialize etcd client failed failed on {etcd_host}:{etcd_port}, ensure the etcd server is running."
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
        while True:
            service, event = await self.event_queue.get()
            try:
                self.logger.info(
                    f"Queue size : {self.event_queue.qsize()} event =  {service}, {event}"
                )
                ev_type = event.event
                ev_value = event.value
                if ev_type == "put":
                    await self.gateway_server_online(service, ev_value)
                elif ev_type == "delete":
                    self.logger.info(f"Service {service} is unavailable")
                    await self.gateway_server_offline(service, ev_value)
                else:
                    raise TypeError(f"Not recognized event type : {ev_type}")
                error_counter = 0  # reset error counter on successful processing
            except Exception as ex:
                self.logger.error(f"Error processing event: {ex}")
                error_counter += 1
                if error_counter >= max_errors:
                    self.logger.error(f"Reached maximum error limit: {max_errors}")
                    break
                await asyncio.sleep(1)
            finally:
                self.event_queue.task_done()

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

        await self.update_gateway_streamer()

    async def update_gateway_streamer(self):
        """Update the gateway streamer with the discovered executors."""
        self.logger.info("Updating gateway streamer")

        # FIXME: testing with only one executor
        deployments_addresses = {}
        graph_description = {
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
        JobManager.SLOTS_AVAILABLE = load_balancer.connection_count()

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
        await self.update_gateway_streamer()


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


# clear;for i in {0..10};do curl localhost:51000/endpoint?text=x_${i} ;done;
