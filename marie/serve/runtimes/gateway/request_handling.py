import asyncio
import copy
import time
from typing import TYPE_CHECKING, Callable, List, Optional

from marie.excepts import InternalNetworkError
from marie.importer import ImportExtensions
from marie.serve.networking import GrpcConnectionPool
from marie.serve.runtimes.gateway.graph.topology_graph import TopologyGraph

if TYPE_CHECKING:
    from prometheus_client import CollectorRegistry

    from marie.types.request import Request


class RequestHandler:
    """
    Class that handles the requests arriving to the gateway and the result extracted from the requests future.

    :param metrics_registry: optional metrics registry for prometheus used if we need to expose metrics from the executor or from the data request handler
    :param runtime_name: optional runtime_name that will be registered during monitoring
    """

    def __init__(
        self,
        metrics_registry: Optional["CollectorRegistry"] = None,
        runtime_name: Optional[str] = None,
    ):
        self.request_init_time = {} if metrics_registry else None
        self._executor_endpoint_mapping = None

        if metrics_registry:
            with ImportExtensions(
                required=True,
                help_text="You need to install the `prometheus_client` to use the montitoring functionality of Marie",
            ):
                from prometheus_client import Summary

            self._summary = Summary(
                "receiving_request_seconds",
                "Time spent processing request",
                registry=metrics_registry,
                namespace="marie",
                labelnames=("runtime_name",),
            ).labels(runtime_name)

        else:
            self._summary = None

    def handle_request(
        self, graph: "TopologyGraph", connection_pool: "GrpcConnectionPool"
    ) -> Callable[["Request"], "asyncio.Future"]:
        """
        Function that handles the requests arriving to the gateway. This will be passed to the streamer.

        :param graph: The TopologyGraph of the Flow.
        :param connection_pool: The connection pool to be used to send messages to specific nodes of the graph
        :return: Return a Function that given a Request will return a Future from where to extract the response
        """

        def _handle_request(request: "Request") -> "asyncio.Future":
            if self._summary:
                self.request_init_time[request.request_id] = time.time()

            print("request : " + request)

        return _handle_request

    def handle_result(self) -> Callable[["Request"], "asyncio.Future"]:
        """
        Function that handles the result when extracted from the request future

        :return: Return a Function that returns a request to be returned to the client
        """

        def _handle_result(result: "Request"):
            """
            Function that handles the result when extracted from the request future

            :param result: The result returned to the gateway. It extracts the request to be returned to the client
            :return: Returns a request to be returned to the client
            """

            for route in result.routes:
                if route.executor == "gateway":
                    route.end_time.GetCurrentTime()

            if self._summary:
                self._summary.observe(time.time() - self.request_init_time[result.request_id])

            return result

        return _handle_result
