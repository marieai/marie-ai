import asyncio
from collections import defaultdict
from typing import TYPE_CHECKING, Optional

from marie.excepts import EstablishGrpcConnectionError
from marie.logging_core.logger import MarieLogger
from marie.serve.networking.balancer.load_balancer import LoadBalancer

if TYPE_CHECKING:
    from marie.serve.networking.balancer.circuit_breaker import CircuitBreakerConfig


class LeastConnectionsLoadBalancer(LoadBalancer):
    """
    Least connections load balancer.

    Find the server/s with the lowest active connections.
    If there are multiple servers with the same number of active connections, use round robin to select one of them.

    Supports optional circuit breaker to exclude unhealthy connections.
    """

    def __init__(
        self,
        deployment_name: str,
        logger: MarieLogger,
        circuit_breaker_config: Optional["CircuitBreakerConfig"] = None,
    ):
        super().__init__(
            deployment_name, logger, circuit_breaker_config=circuit_breaker_config
        )
        self._rr_counter = 0
        self.selection_counter = defaultdict(
            int
        )  # Tracks how many times each connection is selected
        # Async lock for async methods (fix: avoid blocking event loop with threading.Lock)
        self._async_lock = asyncio.Lock()

    async def _get_next_connection(self, num_retries=3):
        """
        Select the next connection using least-connections algorithm.
        Uses asyncio.Lock to avoid blocking the event loop.
        """
        connection = None
        should_retry = False

        # Use asyncio.Lock for async method to avoid blocking event loop
        async with self._async_lock:
            connections = list(self._connections)
            if not connections:
                raise EstablishGrpcConnectionError(
                    f"No connections available for {self._deployment_name}"
                )

            # Filter by circuit breaker state if enabled
            if self._circuit_breaker:
                available_connections = [
                    c
                    for c in connections
                    if self._circuit_breaker.is_available(c.address)
                ]
                # Fallback: if all circuits are open, use all connections
                # This prevents complete service unavailability
                if not available_connections:
                    self._logger.warning(
                        f"All circuits open for {self._deployment_name}, using fallback"
                    )
                    available_connections = connections
            else:
                available_connections = connections

            # Find min active count among available connections
            min_active = min(
                self.active_counter.get(c.address, 0) for c in available_connections
            )
            min_use_connections = [
                c
                for c in available_connections
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
            except IndexError:
                self._rr_counter = 0
                connection = min_use_connections[0]

            if connection is None:
                if num_retries <= 0:
                    raise EstablishGrpcConnectionError(
                        f"All connections unavailable for {self._deployment_name}"
                    )
                self._logger.info(
                    f"Retrying connection selection for {self._deployment_name}"
                )
                should_retry = True
            else:
                # advance counter safely
                self._rr_counter = (self._rr_counter + 1) % len(min_use_connections)
                self.selection_counter[connection.address] += 1
                self.print_selection_stats()

        # Handle retry outside the lock to avoid blocking during await
        if should_retry:
            await asyncio.sleep(0)  # yield to event loop
            return await self._get_next_connection(num_retries - 1)

        return connection

    def update_connections(self, connections: list):
        """
        Rebalance the connections and clean up stale selection stats.
        Fix B6: Prevents unbounded growth of selection_counter.
        """
        # Clean up selection_counter for removed addresses
        new_addresses = {c.address for c in connections if c is not None}
        addresses_to_remove = [
            addr for addr in self.selection_counter if addr not in new_addresses
        ]
        for addr in addresses_to_remove:
            del self.selection_counter[addr]
            if self.debug_loging_enabled:
                self._logger.debug(
                    f"Cleaned up selection_counter for removed address: {addr}"
                )

        # Call parent implementation
        super().update_connections(connections)

        # Reset round-robin counter if needed
        conn_count = len(self._connections)
        if conn_count > 0:
            self._rr_counter = self._rr_counter % conn_count
        else:
            self._rr_counter = 0

    def print_selection_stats(self):
        self._logger.debug(f"Connection selection stats for {self._deployment_name}:")
        for address, count in self.selection_counter.items():
            self._logger.debug(f"  {address}: selected {count} times")

    def close(self):
        """Close the load balancer and clean up resources."""
        self._rr_counter = 0
        self.selection_counter.clear()
