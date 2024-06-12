import asyncio
import time
import traceback
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Union
from urllib.parse import urlparse

import grpc
from docarray import DocList
from docarray.documents import TextDoc

import marie
import marie.helper
from marie import Gateway as BaseGateway
from marie.logging.logger import MarieLogger
from marie.proto import jina_pb2, jina_pb2_grpc
from marie.serve.discovery.resolver import EtcdServiceResolver
from marie.serve.networking import create_async_channel_stub
from marie.serve.networking.utils import get_grpc_channel
from marie.serve.runtimes.servers.composite import CompositeServer
from marie.serve.runtimes.servers.grpc import GRPCServer


class MarieServerGateway(BaseGateway, CompositeServer):
    """A custom Gateway for Marie server.
    Effectively we are providing a custom implementation of the Gateway class that providers communication between individual executors and the server.

    This utilizes service discovery to find deployed Executors from discovered gateways that could have spawned them(Flow/Deployment).

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.logger = MarieLogger(self.__class__.__name__)
        self.logger.info(f"Setting up MarieServerGateway")
        self.setup_service_discovery()

        def _extend_rest_function(app):
            @app.get("/endpoint")
            async def get(text: str):
                print(f"Received request at {datetime.now()}")

                result = None
                async for docs in self.streamer.stream_docs(
                    docs=DocList[TextDoc]([TextDoc(text=text)]),
                    exec_endpoint="/",
                    # exec_endpoint="/endpoint",
                    # target_executor="executor0",
                    return_results=False,
                ):
                    result = docs[0].text
                return {"result": result}

            return app

        marie.helper.extend_rest_interface = _extend_rest_function

    def setup_service_discovery(
        self,
        etcd_host: Optional[str] = '0.0.0.0',
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
        self.logger.info(f"Event from service : {service}, {event}")
        ev_type = event.event
        ev_key = event.key
        ev_value = event.value

        if ev_type == "put":
            self.gateway_server_online(service, ev_value)
        elif ev_type == "delete":
            self.logger.info(f"Service {service} is unavailable")
        else:
            raise TypeError(f'Not recognized event type : {ev_type}')

    def gateway_server_online(self, service, event_value):
        self.logger.info(f"Service {service} is available @ {event_value}")
        ctrl_address = f"{event_value}"

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
        deployment_name = "gateway-xyz"
        TLS_PROTOCOL_SCHEMES = ['grpcs']

        parsed_address = urlparse(ctrl_address)
        address = parsed_address.netloc if parsed_address.netloc else ctrl_address
        use_tls = parsed_address.scheme in TLS_PROTOCOL_SCHEMES
        channel_options = None
        timeout = 2

        print(f"address: {address}")
        print(f"use_tls: {use_tls}")

        for i in range(2):
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
                    self.logger.info(f"response: {response}")
                    break
            except grpc.RpcError as e:
                if e.code() != grpc.StatusCode.UNAVAILABLE or i == 1:
                    raise

        # validate the service address
        # if False:
        #     while True:
        #         self.logger.info("Checking service address...")
        #         await asyncio.sleep(watchdog_interval)
        #
        # asyncio.create_task(_start_watcher())
