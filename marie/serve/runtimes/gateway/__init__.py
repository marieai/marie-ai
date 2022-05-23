import argparse
from abc import ABC
from typing import TYPE_CHECKING, Optional, Union

# from jina.serve.networking import GrpcConnectionPool
from marie.serve.runtimes.asyncio import AsyncNewLoopRuntime
# from jina.serve.runtimes.gateway.graph.topology_graph import TopologyGraph

if TYPE_CHECKING:
    import asyncio
    import multiprocessing
    import threading


class GatewayRuntime(AsyncNewLoopRuntime, ABC):
    """
    The Runtime from which the GatewayRuntimes need to inherit
    """

    def __init__(
        self,
        args: argparse.Namespace,
        cancel_event: Optional[
            Union['asyncio.Event', 'multiprocessing.Event', 'threading.Event']
        ] = None,
        **kwargs,
    ):
        # this order is intentional: The timeout is needed in _set_topology_graph(), called by super
        self.timeout_send = args.timeout_send
        if self.timeout_send:
            self.timeout_send /= 1e3  # convert ms to seconds
        super().__init__(args, cancel_event, **kwargs)

    def _set_topology_graph(self):
        pass

    def _set_connection_pool(self):
        pass
