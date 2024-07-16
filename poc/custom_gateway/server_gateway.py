import asyncio
import json
import time
from datetime import datetime
from typing import Callable, Optional
from urllib.parse import urlparse

import grpc
from docarray import DocList
from docarray.documents import TextDoc
from grpc.aio import ClientInterceptor

import marie
import marie.helper
from marie import Gateway as BaseGateway
from marie.excepts import RuntimeFailToStart
from marie.helper import get_or_reuse_loop
from marie.logging.logger import MarieLogger
from marie.proto import jina_pb2, jina_pb2_grpc
from marie.serve.discovery import JsonAddress
from marie.serve.discovery.resolver import EtcdServiceResolver
from marie.serve.networking.balancer.interceptor import LoadBalancerInterceptor
from marie.serve.networking.balancer.load_balancer import LoadBalancerType
from marie.serve.networking.balancer.round_robin_balancer import RoundRobinLoadBalancer
from marie.serve.networking.connection_stub import _ConnectionStubs
from marie.serve.networking.utils import get_grpc_channel
from marie.serve.runtimes.gateway.streamer import GatewayStreamer
from marie.serve.runtimes.servers.composite import CompositeServer
from marie.serve.runtimes.servers.grpc import GRPCServer
from marie_server.job.common import JobInfo, JobStatus
from marie_server.job.gateway_job_distributor import GatewayJobDistributor


def create_trace_interceptor() -> ClientInterceptor:
    return CustomClientInterceptor()


def create_balancer_interceptor() -> LoadBalancerInterceptor:
    def notify(event, connection):
        print(f"notify: {event}, {connection}")

    return GatewayLoadBalancerInterceptor(notifier=notify)


class MarieServerGateway(BaseGateway, CompositeServer):
    """A custom Gateway for Marie server.
    Effectively we are providing a custom implementation of the Gateway class that providers communication between individual executors and the server.

    This utilizes service discovery to find deployed Executors from discovered gateways that could have spawned them(Flow/Deployment).

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.logger = MarieLogger(self.__class__.__name__)
        self.logger.info(f"Setting up MarieServerGateway")
        self._loop = get_or_reuse_loop()
        self.deployment_nodes = {}
        self.event_queue = asyncio.Queue()
        self.distributor = GatewayJobDistributor(
            gateway_streamer=self.streamer, logger=self.logger
        )

        def _extend_rest_function(app):
            @app.on_event("shutdown")
            async def _shutdown():
                await self.distributor.close()

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

            return app

        marie.helper.extend_rest_interface = _extend_rest_function

    async def setup_server(self):
        """
        setup servers inside CompositeServer
        """
        self.logger.debug(f"Setting up MarieGateway server")
        await super().setup_server()
        await self.setup_service_discovery()

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
        etcd_host: Optional[str] = "0.0.0.0",
        etcd_port: Optional[int] = 2379,
        watchdog_interval: Optional[int] = 2,
    ):
        """
         Setup service discovery for the gateway.

        :param etcd_host: Optional[str] - The host address of the ETCD service. Default is "0.0.0.0".
        :param etcd_port: Optional[int] - The port of the ETCD service. Default is 2379.
        :param watchdog_interval: Optional[int] - The interval in seconds between each service address check. Default is 2.
        :return: None

        """
        self.logger.info("Setting up service discovery ")
        # FIXME : This is a temporary solution to test the service discovery
        service_name = "gateway/service_test"

        async def _start_watcher():
            resolver = EtcdServiceResolver(
                etcd_host,
                etcd_port,
                namespace="marie",
                start_listener=False,
                listen_timeout=5,
            )

            self.logger.info(f"checking : {resolver.resolve(service_name)}")
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
        :return:
        """

        self._loop.call_soon_threadsafe(
            lambda: asyncio.ensure_future(self.event_queue.put((service, event)))
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
                ev_key = event.key
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
            is_ready = GRPCServer.is_ready(ctrl_address)
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
            self.logger.info(f"Discovered nodes for executor : {executor}")
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
            aio_tracing_client_interceptors=[create_trace_interceptor()],
        )
        self.streamer = streamer
        self.distributor.streamer = streamer
        request_models_map = streamer._endpoints_models_map
        self.logger.info(f"request_models_map: {request_models_map}")

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
        print(f"on_connection_released: {connection}")
        self.active_connection = None
        self.notify("released", connection)

    def on_connection_failed(self, connection: _ConnectionStubs, exception):
        print(f"on_connection_failed: {connection}, {exception}")
        self.active_connection = None
        self.notify("failed", connection)

    def on_connection_acquired(self, connection: _ConnectionStubs):
        print(f"on_connection_acquired: {connection}")
        self.active_connection = connection
        self.notify("acquired", connection)

    def on_connections_updated(self, connections: list[_ConnectionStubs]):
        print(f"on_connections_updated: {connections}")
        self.notify("updated", connections)

    def get_active_connection(self):
        """
        Get the active connection.
        :return:
        """
        return self.active_connection


class CustomClientInterceptor(
    grpc.aio.UnaryUnaryClientInterceptor,
    grpc.aio.UnaryStreamClientInterceptor,
    grpc.aio.StreamUnaryClientInterceptor,
    grpc.aio.StreamStreamClientInterceptor,
):
    async def intercept_unary_unary(self, continuation, client_call_details, request):
        print(f"intercept_unary_unary: {client_call_details}, {request}")
        return await continuation(client_call_details, request)

    async def intercept_unary_stream(self, continuation, client_call_details, request):
        print(f"intercept_unary_stream: {client_call_details}, {request}")
        return await continuation(client_call_details, request)

    async def intercept_stream_unary(
        self, continuation, client_call_details, request_iterator
    ):
        print(f"intercept_stream_unary: {client_call_details}, {request_iterator}")
        return await continuation(client_call_details, request_iterator)

    async def intercept_stream_stream(
        self, continuation, client_call_details, request_iterator
    ):
        print(f"intercept_stream_stream: {client_call_details}, {request_iterator}")
        return await continuation(client_call_details, request_iterator)


# clear;for i in {0..10};do curl localhost:51000/endpoint?text=x_${i} ;done;
