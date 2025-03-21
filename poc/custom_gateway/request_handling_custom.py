import itertools
from typing import TYPE_CHECKING, AsyncIterator, Dict

from marie.enums import ProtocolType
from marie.helper import get_full_version
from marie.proto import jina_pb2
from marie.types_core.request.data import DataRequest, SingleDocumentRequest
from marie.types_core.request.status import StatusMessage

if TYPE_CHECKING:  # pragma: no cover
    from types import SimpleNamespace

    import grpc

    from marie.logging_core.logger import MarieLogger
    from marie.serve.runtimes.gateway.streamer import GatewayStreamer
    from marie.types_core.request import Request


class GatewayRequestHandler:
    """Object to encapsulate the code related to handle the data requests in the Gateway"""

    def __init__(
        self,
        args: "SimpleNamespace",
        logger: "MarieLogger",
        metrics_registry=None,
        meter=None,
        aio_tracing_client_interceptors=None,
        tracing_client_interceptor=None,
        **kwargs,
    ):
        print("CUSTOM GATEWAY REQUEST HANDLER INITIALIZED")
        import json

        from marie.serve.runtimes.gateway.streamer import (
            GatewayStreamer,
            _ExecutorStreamer,
        )

        self.runtime_args = args
        self.logger = logger
        graph_description = json.loads(self.runtime_args.graph_description)
        graph_conditions = json.loads(self.runtime_args.graph_conditions)
        deployments_addresses = json.loads(self.runtime_args.deployments_addresses)
        deployments_metadata = json.loads(self.runtime_args.deployments_metadata)
        deployments_no_reduce = json.loads(self.runtime_args.deployments_no_reduce)

        deployment_grpc_addresses = {}
        for deployment_name, addresses in deployments_addresses.items():
            if isinstance(addresses, Dict):
                deployment_grpc_addresses[deployment_name] = addresses.get(
                    ProtocolType.GRPC.to_string(), []
                )
            else:
                deployment_grpc_addresses[deployment_name] = addresses

        self.streamer = GatewayStreamer(
            graph_representation=graph_description,
            executor_addresses=deployment_grpc_addresses,
            graph_conditions=graph_conditions,
            deployments_metadata=deployments_metadata,
            deployments_no_reduce=deployments_no_reduce,
            timeout_send=self.runtime_args.timeout_send,
            retries=self.runtime_args.retries,
            compression=self.runtime_args.compression,
            runtime_name=self.runtime_args.runtime_name,
            prefetch=self.runtime_args.prefetch,
            logger=self.logger,
            metrics_registry=metrics_registry,
            meter=meter,
            aio_tracing_client_interceptors=aio_tracing_client_interceptors,
            tracing_client_interceptor=tracing_client_interceptor,
            grpc_channel_options=(
                self.runtime_args.grpc_channel_options
                if hasattr(self.runtime_args, "grpc_channel_options")
                else None
            ),
        )

        GatewayStreamer._set_env_streamer_args(
            graph_representation=graph_description,
            executor_addresses=deployment_grpc_addresses,
            graph_conditions=graph_conditions,
            deployments_metadata=deployments_metadata,
            deployments_no_reduce=deployments_no_reduce,
            timeout_send=self.runtime_args.timeout_send,
            retries=self.runtime_args.retries,
            compression=self.runtime_args.compression,
            runtime_name=self.runtime_args.runtime_name,
            prefetch=self.runtime_args.prefetch,
        )

        self.executor = {
            executor_name: _ExecutorStreamer(
                self.streamer._connection_pool, executor_name=executor_name
            )
            for executor_name in deployment_grpc_addresses.keys()
        }
        servers = []
        for addresses in deployments_addresses.values():
            if isinstance(addresses, Dict):
                servers.extend(addresses.get(ProtocolType.HTTP.to_string(), []))
        self.load_balancer_servers = itertools.cycle(servers)

    async def close(self):
        """
        Gratefully closes the object making sure all the floating requests are taken care and the connections are closed gracefully
        """
        self.logger.debug(f"Closing Request Handler")
        await self.streamer.close()
        self.logger.debug(f"Request Handler closed")

    def _http_fastapi_default_app(
        self,
        title,
        description,
        no_debug_endpoints,
        no_crud_endpoints,
        expose_endpoints,
        expose_graphql_endpoint,
        cors,
        tracing,
        tracer_provider,
        **kwargs,
    ):

        from marie._docarray import docarray_v2
        from marie.helper import extend_rest_interface

        if not docarray_v2:
            from marie.serve.runtimes.gateway.http_fastapi_app import get_fastapi_app
        else:
            from marie.serve.runtimes.gateway.http_fastapi_app_docarrayv2 import (
                get_fastapi_app,
            )

        return extend_rest_interface(
            get_fastapi_app(
                streamer=self.streamer,
                title=title,
                description=description,
                no_debug_endpoints=no_debug_endpoints,
                no_crud_endpoints=no_crud_endpoints,
                expose_endpoints=expose_endpoints,
                expose_graphql_endpoint=expose_graphql_endpoint,
                cors=cors,
                logger=self.logger,
                tracing=tracing,
                tracer_provider=tracer_provider,
            )
        )

    async def _load_balance(self, request):
        import aiohttp
        from aiohttp import web

        target_server = next(self.load_balancer_servers)
        target_url = f"{target_server}{request.path_qs}"

        try:
            async with aiohttp.ClientSession() as session:

                request_kwargs = {}
                try:
                    payload = await request.json()
                    if payload:
                        request_kwargs["json"] = payload
                except Exception:
                    self.logger.debug("No JSON payload found in request")

                async with session.request(
                    request.method,
                    url=target_url,
                    auto_decompress=False,
                    **request_kwargs,
                ) as response:
                    # Create a StreamResponse with the same headers and status as the target response
                    stream_response = web.StreamResponse(
                        status=response.status,
                        headers=response.headers,
                    )

                    # Prepare the response to send headers
                    await stream_response.prepare(request)

                    # Stream the response from the target server to the client
                    async for chunk in response.content.iter_any():
                        await stream_response.write(chunk)

                    # Close the stream response once all chunks are sent
                    await stream_response.write_eof()
                    return stream_response

        except aiohttp.ClientError as e:
            return web.Response(text=f"Error: {str(e)}", status=500)

    def _websocket_fastapi_default_app(self, tracing, tracer_provider):
        from marie.helper import extend_rest_interface
        from marie.serve.runtimes.gateway.websocket_fastapi_app import get_fastapi_app

        return extend_rest_interface(
            get_fastapi_app(
                streamer=self.streamer,
                logger=self.logger,
                tracing=tracing,
                tracer_provider=tracer_provider,
            )
        )

    async def dry_run(self, empty, context) -> jina_pb2.StatusProto:
        """
        Process the call requested by having a dry run call to every Executor in the graph

        :param empty: The service expects an empty protobuf message
        :param context: grpc context
        :returns: the response request
        """
        self.logger.debug("recv a dry_run request")
        from marie._docarray import Document, DocumentArray
        from marie.serve.executors import __dry_run_endpoint__

        da = DocumentArray([Document()])
        try:
            async for _ in self.streamer.stream_docs(
                docs=da, exec_endpoint=__dry_run_endpoint__, request_size=1
            ):
                pass
            status_message = StatusMessage()
            status_message.set_code(jina_pb2.StatusProto.SUCCESS)
            return status_message.proto
        except Exception as ex:
            status_message = StatusMessage()
            status_message.set_exception(ex)
            return status_message.proto

    async def _status(self, empty, context) -> jina_pb2.JinaInfoProto:
        """
        Process the the call requested and return the JinaInfo of the Runtime

        :param empty: The service expects an empty protobuf message
        :param context: grpc context
        :returns: the response request
        """
        self.logger.debug("recv a _status request")
        info_proto = jina_pb2.JinaInfoProto()
        version, env_info = get_full_version()
        for k, v in version.items():
            info_proto.jina[k] = str(v)
        for k, v in env_info.items():
            info_proto.envs[k] = str(v)
        return info_proto

    async def stream(
        self, request_iterator, context=None, *args, **kwargs
    ) -> AsyncIterator["Request"]:
        """
        stream requests from client iterator and stream responses back.

        :param request_iterator: iterator of requests
        :param context: context of the grpc call
        :param args: positional arguments
        :param kwargs: keyword arguments
        :yield: responses to the request after streaming to Executors in Flow
        """
        self.logger.debug("recv a stream request GATEWAY")

        async for resp in self.streamer.rpc_stream(
            request_iterator=request_iterator, context=context, *args, **kwargs
        ):
            yield resp

    async def stream_doc(
        self, request: SingleDocumentRequest, context: "grpc.aio.ServicerContext"
    ) -> SingleDocumentRequest:
        """
        Process the received requests and return the result as a new request

        :param request: the data request to process
        :param context: grpc context
        :yields: the response request
        """
        self.logger.debug("recv a stream_doc request")
        async for result in self.streamer.rpc_stream_doc(
            request=request,
        ):
            yield result

    async def process_single_data(
        self, request: DataRequest, context=None
    ) -> DataRequest:
        """Implements request and response handling of a single DataRequest
        :param request: DataRequest from Client
        :param context: grpc context
        :return: response DataRequest
        """
        self.logger.debug(f"recv a process_single_data request")
        return await self.streamer.process_single_data(request, context)

    async def endpoint_discovery(self, empty, context) -> jina_pb2.EndpointsProto:
        """
        Uses the connection pool to send a discover endpoint call to the Executors

        :param empty: The service expects an empty protobuf message
        :param context: grpc context
        :returns: the response request
        """
        from google.protobuf import json_format

        self.logger.debug("got an endpoint discovery request")
        response = jina_pb2.EndpointsProto()
        await self.streamer._get_endpoints_input_output_models(is_cancel=None)
        request_models_map = self.streamer._endpoints_models_map
        if request_models_map is not None and len(request_models_map) > 0:
            schema_maps = {}
            for k, v in request_models_map.items():
                schema_maps[k] = {}
                schema_maps[k]["input"] = v["input"].schema()
                schema_maps[k]["output"] = v["output"].schema()
                schema_maps[k]["is_generator"] = v["is_generator"]
                schema_maps[k]["is_singleton_doc"] = v["is_singleton_doc"]
            response.endpoints.extend(schema_maps.keys())
            json_format.ParseDict(schema_maps, response.schemas)
        else:
            endpoints = await self.streamer.topology_graph._get_all_endpoints(
                self.streamer._connection_pool, retry_forever=True, is_cancel=None
            )
            response.endpoints.extend(list(endpoints))
        return response

    Call = stream
