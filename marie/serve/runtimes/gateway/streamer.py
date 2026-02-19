import asyncio
import json
import os
from typing import (
    TYPE_CHECKING,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

from docarray import DocList

from marie._docarray import Document, DocumentArray
from marie.excepts import ExecutorError
from marie.logging_core.logger import MarieLogger
from marie.proto import jina_pb2
from marie.serve.networking import GrpcConnectionPool
from marie.serve.runtimes.gateway.async_request_response_handling import (
    AsyncRequestResponseHandler,
)
from marie.serve.runtimes.gateway.graph.topology_graph import TopologyGraph
from marie.serve.stream import RequestStreamer
from marie.types_core.request import Request
from marie.types_core.request.data import DataRequest, SingleDocumentRequest

__all__ = ["GatewayStreamer"]

if TYPE_CHECKING:  # pragma: no cover
    from grpc.aio._interceptor import ClientInterceptor
    from opentelemetry.instrumentation.grpc._client import (
        OpenTelemetryClientInterceptor,
    )
    from opentelemetry.metrics import Meter
    from prometheus_client import CollectorRegistry


class GatewayStreamer:
    """
    Wrapper object to be used in a Custom Gateway. Naming to be defined
    """

    def __init__(
        self,
        graph_representation: Dict,
        executor_addresses: Dict[str, Union[str, List[str]]],
        graph_conditions: Dict = {},
        deployments_metadata: Dict[str, Dict[str, str]] = {},
        deployments_no_reduce: List[str] = [],
        timeout_send: Optional[float] = None,
        retries: int = 0,
        compression: Optional[str] = None,
        runtime_name: str = "custom gateway",
        prefetch: int = 0,
        logger: Optional["MarieLogger"] = None,
        metrics_registry: Optional["CollectorRegistry"] = None,
        meter: Optional["Meter"] = None,
        aio_tracing_client_interceptors: Optional[Sequence["ClientInterceptor"]] = None,
        tracing_client_interceptor: Optional["OpenTelemetryClientInterceptor"] = None,
        grpc_channel_options: Optional[list] = None,
        load_balancer_type: Optional[str] = "round_robin",
    ):
        """
        :param graph_representation: A dictionary describing the topology of the Deployments. 2 special nodes are expected, the name `start-gateway` and `end-gateway` to
            determine the nodes that receive the very first request and the ones whose response needs to be sent back to the client. All the nodes with no outgoing nodes
            will be considered to be floating, and they will be "flagged" so that the user can ignore their tasks and not await them.
        :param executor_addresses: dictionary JSON with the input addresses of each Deployment. Each Executor can have one single address or a list of addrresses for each Executor
        :param graph_conditions: Dictionary stating which filtering conditions each Executor in the graph requires to receive Documents.
        :param deployments_metadata: Dictionary with the metadata of each Deployment. Each executor deployment can have a list of key-value pairs to
            provide information associated with the request to the deployment.
        :param deployments_no_reduce: list of Executor disabling the built-in merging mechanism.
        :param timeout_send: Timeout to be considered when sending requests to Executors
        :param retries: Number of retries to try to make successfull sendings to Executors
        :param compression: The compression mechanism used when sending requests from the Head to the WorkerRuntimes. For more details, check https://grpc.github.io/grpc/python/grpc.html#compression.
        :param runtime_name: Name to be used for monitoring.
        :param prefetch: How many Requests are processed from the Client at the same time.
        :param logger: Optional logger that can be used for logging
        :param metrics_registry: optional metrics registry for prometheus used if we need to expose metrics
        :param meter: optional OpenTelemetry meter that can provide instruments for collecting metrics
        :param aio_tracing_client_interceptors: Optional list of aio grpc tracing server interceptors.
        :param tracing_client_interceptor: Optional gprc tracing server interceptor.
        :param grpc_channel_options: Optional gprc channel options.
        :param load_balancer_type: Optional load balancer type. Default is round-robin.
        """
        self.logger = logger or MarieLogger(self.__class__.__name__)
        self.topology_graph = TopologyGraph(
            graph_representation=graph_representation,
            graph_conditions=graph_conditions,
            deployments_metadata=deployments_metadata,
            deployments_no_reduce=deployments_no_reduce,
            timeout_send=timeout_send,
            retries=retries,
            logger=logger,
        )

        self.runtime_name = runtime_name
        self.aio_tracing_client_interceptors = aio_tracing_client_interceptors
        self.tracing_client_interceptor = tracing_client_interceptor
        self._executor_addresses = executor_addresses

        self._connection_pool = self._create_connection_pool(
            executor_addresses,
            compression,
            metrics_registry,
            meter,
            logger,
            aio_tracing_client_interceptors,
            tracing_client_interceptor,
            grpc_channel_options,
            load_balancer_type,
        )
        request_handler = AsyncRequestResponseHandler(
            metrics_registry, meter, runtime_name, logger
        )
        self._single_doc_request_handler = (
            request_handler.handle_single_document_request(
                graph=self.topology_graph, connection_pool=self._connection_pool
            )
        )
        self._streamer = RequestStreamer(
            request_handler=request_handler.handle_request(
                graph=self.topology_graph, connection_pool=self._connection_pool
            ),
            result_handler=request_handler.handle_result(),
            prefetch=prefetch,
            logger=logger,
        )
        self._endpoints_models_map = None
        self._streamer.Call = self._streamer.stream

    def _create_connection_pool(
        self,
        deployments_addresses,
        compression,
        metrics_registry,
        meter,
        logger,
        aio_tracing_client_interceptors,
        tracing_client_interceptor,
        grpc_channel_options=None,
        load_balancer_type=None,
    ):
        # add the connections needed
        connection_pool = GrpcConnectionPool(
            runtime_name=self.runtime_name,
            logger=logger,
            compression=compression,
            metrics_registry=metrics_registry,
            meter=meter,
            aio_tracing_client_interceptors=aio_tracing_client_interceptors,
            tracing_client_interceptor=tracing_client_interceptor,
            channel_options=grpc_channel_options,
            load_balancer_type=load_balancer_type,
        )
        for deployment_name, addresses in deployments_addresses.items():
            for address in addresses:
                connection_pool.add_connection(
                    deployment=deployment_name, address=address, head=True
                )

        return connection_pool

    def rpc_stream(self, *args, **kwargs):
        """
        stream requests from client iterator and stream responses back.

        :param args: positional arguments to be passed to inner RequestStreamer
        :param kwargs: keyword arguments to be passed to inner RequestStreamer
        :return: An iterator over the responses from the Executors
        """
        return self._streamer.stream(*args, **kwargs)

    def rpc_stream_doc(self, *args, **kwargs):
        """
        stream requests from client iterator and stream responses back.

        :param args: positional arguments to be passed to inner RequestStreamer
        :param kwargs: keyword arguments to be passed to inner RequestStreamer
        :return: An iterator over the responses from the Executors
        """
        return self._single_doc_request_handler(*args, **kwargs)

    async def _get_endpoints_input_output_models(self, is_cancel):
        """
        Return a Dictionary with endpoints as keys and values as a dictionary of input and output schemas and names
        taken from the endpoints proto endpoint of Executors.
        :param is_cancel: event signal to show that you should stop trying
        """
        # The logic should be to get the response of all the endpoints protos schemas from all the nodes. Then do a
        # logic that for every endpoint fom every Executor computes what is the input and output schema seen by the
        # Flow.
        self._endpoints_models_map = (
            await self._streamer._get_endpoints_input_output_models(
                self.topology_graph, self._connection_pool, is_cancel
            )
        )

    def _validate_flow_docarray_compatibility(self):
        """
        This method aims to validate that the input-output docarray models of Executors are good
        """
        self.topology_graph._validate_flow_docarray_compatibility()

    async def stream(
        self,
        docs: DocumentArray,
        request_size: int = 100,
        return_results: bool = False,
        exec_endpoint: Optional[str] = None,
        target_executor: Optional[str] = None,
        parameters: Optional[Dict] = None,
        results_in_order: bool = False,
        return_type: Type[DocumentArray] = DocumentArray,
    ) -> AsyncIterator[Tuple[Union[DocumentArray, "Request"], "ExecutorError"]]:
        """
        stream Documents and yield Documents or Responses and unpacked Executor error if any.

        :param docs: The Documents to be sent to all the Executors
        :param request_size: The amount of Documents to be put inside a single request.
        :param return_results: If set to True, the generator will yield Responses and not `DocumentArrays`
        :param exec_endpoint: The Executor endpoint to which to send the Documents
        :param target_executor: A regex expression indicating the Executors that should receive the Request
        :param parameters: Parameters to be attached to the Requests
        :param results_in_order: return the results in the same order as the request_iterator
        :param return_type: the DocumentArray type to be returned. By default, it is `DocumentArray`.
        :yield: tuple of Documents or Responses and unpacked error from Executors if any
        """
        async for result in self.stream_docs(
            docs=docs,
            request_size=request_size,
            return_results=True,  # force return Responses
            exec_endpoint=exec_endpoint,
            target_executor=target_executor,
            parameters=parameters,
            results_in_order=results_in_order,
            return_type=return_type,
        ):
            error = None
            if jina_pb2.StatusProto.ERROR == result.status.code:
                exception = result.status.exception
                error = ExecutorError(
                    name=exception.name,
                    args=exception.args,
                    stacks=exception.stacks,
                    executor=exception.executor,
                )
            if return_results:
                yield result, error
            else:
                yield result.data.docs, error

    async def stream_doc(
        self,
        doc: "Document",
        return_results: bool = False,
        exec_endpoint: Optional[str] = None,
        target_executor: Optional[str] = None,
        parameters: Optional[Dict] = None,
        request_id: Optional[str] = None,
        return_type: Type[DocumentArray] = DocumentArray,
        send_callback: Optional[Callable] = None,
    ) -> AsyncIterator[Tuple[Union[DocumentArray, "Request"], "ExecutorError"]]:
        """
        stream Documents and yield Documents or Responses and unpacked Executor error if any.

        :param doc: The Documents to be sent to all the Executors
        :param return_results: If set to True, the generator will yield Responses and not `DocumentArrays`
        :param exec_endpoint: The Executor endpoint to which to send the Documents
        :param target_executor: A regex expression indicating the Executors that should receive the Request
        :param parameters: Parameters to be attached to the Requests
        :param request_id: Request ID to add to the request streamed to Executor. Only applicable if request_size is equal or less to the length of the docs
        :param return_type: the DocumentArray type to be returned. By default, it is `DocumentArray`.
        :param send_callback: callback function to notify the client
        :yield: tuple of Documents or Responses and unpacked error from Executors if any
        """
        req = SingleDocumentRequest()
        req.document_cls = doc.__class__
        req.data.doc = doc
        if request_id:
            req.header.request_id = request_id
        if exec_endpoint:
            req.header.exec_endpoint = exec_endpoint
        if target_executor:
            req.header.target_executor = target_executor
        if parameters:
            req.parameters = parameters

        async for result in self.rpc_stream_doc(
            request=req, return_type=return_type, send_callback=send_callback
        ):
            error = None
            if jina_pb2.StatusProto.ERROR == result.status.code:
                exception = result.status.exception
                error = ExecutorError(
                    name=exception.name,
                    args=exception.args,
                    stacks=exception.stacks,
                    executor=exception.executor,
                )
            if return_results:
                yield result, error
            else:
                yield result.data.doc, error

    async def stream_docs(
        self,
        docs: DocumentArray,
        request_size: int = 100,
        return_results: bool = False,
        exec_endpoint: Optional[str] = None,
        target_executor: Optional[str] = None,
        parameters: Optional[Dict] = None,
        results_in_order: bool = False,
        request_id: Optional[str] = None,
        return_type: Type[DocumentArray] = DocumentArray,
        send_callback: Optional[Callable] = None,
    ):
        """
        stream documents and stream responses back.

        :param docs: The Documents to be sent to all the Executors
        :param request_size: The amount of Documents to be put inside a single request.
        :param return_results: If set to True, the generator will yield Responses and not `DocumentArrays`
        :param exec_endpoint: The Executor endpoint to which to send the Documents
        :param target_executor: A regex expression indicating the Executors that should receive the Request
        :param parameters: Parameters to be attached to the Requests
        :param results_in_order: return the results in the same order as the request_iterator
        :param request_id: Request ID to add to the request streamed to Executor. Only applicable if request_size is equal or less to the length of the docs
        :param return_type: the DocumentArray type to be returned. By default, it is `DocumentArray`.
        :param send_callback: callback function to notify the client
        :yield: Yields DocumentArrays or Responses from the Executors
        """
        request_id = request_id if len(docs) <= request_size else None

        def _req_generator():
            from docarray import BaseDoc

            def batch(iterable, n=1):
                l = len(iterable)
                for ndx in range(0, l, n):
                    yield iterable[ndx : min(ndx + n, l)]

            if len(docs) > 0:
                for docs_batch in batch(docs, n=request_size):
                    req = DataRequest()
                    req.document_array_cls = DocList[docs_batch.doc_type]
                    req.data.docs = docs_batch
                    if request_id:
                        req.header.request_id = request_id
                    if exec_endpoint:
                        req.header.exec_endpoint = exec_endpoint
                    if target_executor:
                        req.header.target_executor = target_executor
                    if parameters:
                        req.parameters = parameters
                    yield req
            else:
                req = DataRequest()
                req.document_array_cls = DocList[BaseDoc]
                req.data.docs = DocList[BaseDoc]()
                if request_id:
                    req.header.request_id = request_id
                if exec_endpoint:
                    req.header.exec_endpoint = exec_endpoint
                if target_executor:
                    req.header.target_executor = target_executor
                if parameters:
                    req.parameters = parameters
                yield req

        async for resp in self.rpc_stream(
            request_iterator=_req_generator(),
            results_in_order=results_in_order,
            return_type=return_type,
            send_callback=send_callback,
        ):
            if return_results:
                yield resp
            else:
                yield resp.docs

    async def close(self):
        """
        Gratefully closes the object making sure all the floating requests are taken care and the connections are closed gracefully
        """
        await self._streamer.wait_floating_requests_end()
        await self._connection_pool.close()

    Call = rpc_stream

    async def process_single_data(
        self,
        request: DataRequest,
        context=None,
        send_callback: Optional[Callable] = None,
    ) -> DataRequest:
        """Implements request and response handling of a single DataRequest
        :param request: DataRequest from Client
        :param context: grpc context
        :param send_callback: callback function to notify the client
        :return: response DataRequest
        """
        return await self._streamer.process_single_data(request, context, send_callback)

    @staticmethod
    def get_streamer():
        """
        Return a streamer object based on the current environment context.
        The streamer object is contructed using runtime arguments stored in the `JINA_STREAMER_ARGS` environment variable.
        If this method is used outside a Jina context (process not controlled/orchestrated by jina), this method will
        raise an error.
        The streamer object does not have tracing/instrumentation capabilities.

        :return: Returns an instance of `GatewayStreamer`
        """
        if "JINA_STREAMER_ARGS" in os.environ:
            args_dict = json.loads(os.environ["JINA_STREAMER_ARGS"])
            return GatewayStreamer(**args_dict)
        else:
            raise OSError("JINA_STREAMER_ARGS environment variable is not set")

    @staticmethod
    def _set_env_streamer_args(**kwargs):
        os.environ["JINA_STREAMER_ARGS"] = json.dumps(kwargs)

    # Incremental update methods for node registration

    def add_connection(self, deployment: str, address: str) -> None:
        """
        Add a single connection incrementally without recreating the streamer.

        :param deployment: Name of the deployment/executor
        :param address: Address to add (host:port format)
        """
        self._connection_pool.add_connection(
            deployment=deployment, address=address, head=True
        )
        if deployment not in self._executor_addresses:
            self._executor_addresses[deployment] = []
        if address not in self._executor_addresses[deployment]:
            self._executor_addresses[deployment].append(address)
        self.logger.debug(f"Added connection for {deployment}: {address}")

    async def remove_connection(self, deployment: str, address: str) -> None:
        """
        Remove a single connection incrementally without recreating the streamer.

        :param deployment: Name of the deployment/executor
        :param address: Address to remove (host:port format)
        """
        await self._connection_pool.remove_connection(
            deployment=deployment, address=address, head=True
        )
        if deployment in self._executor_addresses:
            self._executor_addresses[deployment] = [
                a for a in self._executor_addresses[deployment] if a != address
            ]
        self.logger.debug(f"Removed connection for {deployment}: {address}")

    async def update_executor_addresses(
        self, deployment: str, new_addresses: List[str]
    ) -> None:
        """
        Update addresses for a deployment incrementally.
        Adds new addresses and removes stale ones.

        :param deployment: Name of the deployment/executor
        :param new_addresses: New list of addresses for the deployment
        """
        current = set(self._executor_addresses.get(deployment, []))
        new = set(new_addresses)

        # Add new addresses
        for addr in new - current:
            self.add_connection(deployment, addr)

        # Remove old addresses
        for addr in current - new:
            await self.remove_connection(deployment, addr)

        self.logger.info(
            f"Updated {deployment}: added {len(new - current)}, removed {len(current - new)}"
        )

    def get_executor_addresses(self, deployment: str) -> List[str]:
        """
        Get current addresses for a deployment.

        :param deployment: Name of the deployment/executor
        :return: List of addresses for the deployment
        """
        return list(self._executor_addresses.get(deployment, []))

    def get_all_executor_addresses(self) -> Dict[str, List[str]]:
        """
        Get all executor addresses.

        :return: Dictionary mapping deployment names to address lists
        """
        return {k: list(v) for k, v in self._executor_addresses.items()}


class _ExecutorStreamer:
    def __init__(self, connection_pool: GrpcConnectionPool, executor_name: str) -> None:
        self._connection_pool: GrpcConnectionPool = connection_pool
        self.executor_name = executor_name

    async def post(
        self,
        inputs: DocumentArray,
        request_size: int = 100,
        on: Optional[str] = None,
        parameters: Optional[Dict] = None,
        return_type: Type[DocumentArray] = DocumentArray,
        **kwargs,
    ):
        if not parameters:
            parameters = {}

        from docarray import BaseDoc

        def batch(iterable, n=1):
            l = len(iterable)
            for ndx in range(0, l, n):
                yield iterable[ndx : min(ndx + n, l)]

        reqs = []

        if len(inputs) > 0:
            for docs_batch in batch(inputs, n=request_size):
                req = DataRequest()
                req.document_array_cls = DocList[docs_batch.doc_type]
                req.data.docs = docs_batch
                req.header.exec_endpoint = on
                req.header.target_executor = self.executor_name
                req.parameters = parameters
                reqs.append(req)
        else:
            req = DataRequest()
            req.document_array_cls = DocList[BaseDoc]
            req.data.docs = DocList[BaseDoc]()
            req.header.exec_endpoint = on
            req.header.target_executor = self.executor_name
            req.parameters = parameters
            reqs.append(req)

        tasks = [
            self._connection_pool.send_requests_once(
                requests=[req], deployment=self.executor_name, head=True, endpoint=on
            )
            for req in reqs
        ]

        results = await asyncio.gather(*tasks)

        docs = DocList[return_type.doc_type]()
        for resp, _ in results:
            resp.document_array_cls = return_type
            docs.extend(resp.docs)
        return docs

    async def stream_doc(
        self,
        inputs: "Document",
        on: Optional[str] = None,
        parameters: Optional[Dict] = None,
        **kwargs,
    ):
        req: SingleDocumentRequest = SingleDocumentRequest(inputs.to_protobuf())
        req.header.exec_endpoint = on
        req.header.target_executor = self.executor_name
        req.parameters = parameters
        async_generator = self._connection_pool.send_single_document_request(
            request=req, deployment=self.executor_name, head=True, endpoint=on
        )

        async for resp, _ in async_generator:
            yield resp
