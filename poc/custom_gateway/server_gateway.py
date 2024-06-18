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

        print("MarieServerGateway.__init__")
        self._loop = get_or_reuse_loop()
        self.deployment_nodes = {}
        self.setup_service_discovery()

        def _extend_rest_function(app):
            @app.get("/endpoint")
            async def get(text: str):
                self.logger.info(f"Received request at {datetime.now()}")
                result = None
                async for docs in self.streamer.stream_docs(
                    docs=DocList[TextDoc]([TextDoc(text=text)]),
                    # exec_endpoint="/extract",  # _jina_dry_run_
                    exec_endpoint="_jina_dry_run_",  # _jina_dry_run_
                    # exec_endpoint="/endpoint",
                    # target_executor="executor0",
                    return_results=False,
                ):
                    result = docs[0].text
                return {"result": result}

            @app.get("/check")
            async def get_health(text: str):
                self.logger.info(f"Received request at {datetime.now()}")
                return {"result": "ok"}

            return app

        marie.helper.extend_rest_interface = _extend_rest_function

    def setup_service_discovery(
        self,
        etcd_host: Optional[str] = "0.0.0.0",
        etcd_port: Optional[int] = 2379,
        watchdog_interval: Optional[int] = 2,
    ):
        """Setup service discovery for the gateway."""
        self.logger.info("Setting up service discovery ")
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
            # validate the service address
            if False:
                while True:
                    self.logger.info("Checking service address...")
                    await asyncio.sleep(watchdog_interval)

        asyncio.create_task(_start_watcher())

    def handle_discovery_event(self, service, event):
        """
        Handle a discovery event.

        :param service: The service that triggered the event.
        :param event: The event object containing information about the event.
        :return: None
        """
        self.logger.info(f"Event from service : {service}, {event}")
        ev_type = event.event
        ev_key = event.key
        ev_value = event.value

        if ev_type == "put":
            self.gateway_server_online(service, ev_value)
        elif ev_type == "delete":
            self.logger.info(f"Service {service} is unavailable")
            print("event: ", event)
            print("ev_type: ", ev_type)
            print("ev_value: ", ev_value)
            self.gateway_server_offline(service, ev_value)
        else:
            raise TypeError(f"Not recognized event type : {ev_type}")

    def gateway_server_online(self, service, event_value):
        """
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
            self.logger.info(
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
            print(f"deployment_addresses: {deployment_addresses}")
            for deployment_address in deployment_addresses:
                print(f"deployment_address: {deployment_address}")
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

        self.update_gateway_streamer()

    def update_gateway_streamer(self):
        """Update the gateway streamer with the discovered executors."""
        self.logger.info("Updating gateway streamer")

        async def _streamer_setup():
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

        self._loop.create_task(_streamer_setup())

    def gateway_server_offline(self, service: str, ev_value):
        """
        :param service: The name of the service.
        :param ev_value: The value representing the offline gateway.
        :return: None
        """
        # loop over the deployments_addresses and remove the offline gateway from the list
        ctrl_address = service.split("/")[-1]
        self.logger.info(
            f"Service {service} is offline @ {ctrl_address}, removing nodes"
        )
        for executor, nodes in self.deployment_nodes.items():
            for node in nodes:
                if node["gateway"] == ctrl_address:
                    self.logger.info(f"Removing node: {node}")
                    self.deployment_nodes[executor].remove(node)
        self.update_gateway_streamer()


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
