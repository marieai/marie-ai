import asyncio
from collections import defaultdict

from marie.excepts import EstablishGrpcConnectionError
from marie.logging_core.logger import MarieLogger
from marie.serve.networking.balancer.load_balancer import LoadBalancer


class LeastConnectionsLoadBalancer(LoadBalancer):
    """
    Least connections load balancer.

    Find the server/s with the lowest active connections.
    If there are multiple servers with the same number of active connections, use round robin to select one of them.

    """

    def __init__(self, deployment_name: str, logger: MarieLogger):
        super().__init__(deployment_name, logger)
        self._rr_counter = 0
        self.selection_counter = defaultdict(
            int
        )  # Tracks how many times each connection is selected

    async def _get_next_connection(self, num_retries=3):
        with self._lock:  # lock to ensure thread safety
            connections = list(self._connections)
            if not connections:
                raise EstablishGrpcConnectionError(
                    f"No connections available for {self._deployment_name}"
                )

            # Find min active count
            min_active = min(self.active_counter.get(c.address, 0) for c in connections)
            min_use_connections = [
                c
                for c in connections
                if self.active_counter.get(c.address, 0) == min_active
            ]

            if not min_use_connections:
                raise EstablishGrpcConnectionError(
                    f"No usable connections for {self._deployment_name}"
                )

            if len(min_use_connections) == 1:
                self._rr_counter = 0

            try:
                index = self._rr_counter % len(min_use_connections)
                connection = min_use_connections[index]

                if connection is None:
                    if num_retries <= 0:
                        raise EstablishGrpcConnectionError(
                            f"All connections unavailable for {self._deployment_name}"
                        )
                    self._logger.info(
                        f"Retrying connection selection for {self._deployment_name}"
                    )
                    await asyncio.sleep(0)  # yield to loop
                    return await self._get_next_connection(num_retries - 1)

            except IndexError:
                self._rr_counter = 0
                connection = min_use_connections[0]

            # advance counter safely
            self._rr_counter = (self._rr_counter + 1) % len(min_use_connections)

            if connection:
                self.selection_counter[connection.address] += 1
            self.print_selection_stats()
            return connection

    def print_selection_stats(self):
        self._logger.debug(f"Connection selection stats for {self._deployment_name}:")
        for address, count in self.selection_counter.items():
            self._logger.debug(f"  {address}: selected {count} times")

    def close(self):
        pass
