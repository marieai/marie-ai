import asyncio
from typing import TYPE_CHECKING, Optional, Sequence

from marie.excepts import EstablishGrpcConnectionError
from marie.logging_core.logger import MarieLogger
from marie.serve.networking.balancer.interceptor import LoadBalancerInterceptor
from marie.serve.networking.balancer.load_balancer import LoadBalancer

if TYPE_CHECKING:
    from marie.serve.networking.balancer.circuit_breaker import CircuitBreakerConfig


class RoundRobinLoadBalancer(LoadBalancer):
    """
    Round-robin load balancer.

    Supports optional circuit breaker to exclude unhealthy connections.
    """

    def __init__(
        self,
        deployment_name: str,
        logger: Optional[MarieLogger] = None,
        tracing_interceptors: Optional[Sequence[LoadBalancerInterceptor]] = None,
        circuit_breaker_config: Optional["CircuitBreakerConfig"] = None,
    ):
        super().__init__(
            deployment_name,
            logger,
            tracing_interceptors=tracing_interceptors,
            circuit_breaker_config=circuit_breaker_config,
        )
        self._rr_counter = 0  # round-robin counter
        # Async lock for async methods (fix B8: ensure counter atomicity)
        self._async_lock = asyncio.Lock()

    async def _get_next_connection(self, num_retries=3):
        """
        Select the next connection using round-robin algorithm.
        Uses asyncio.Lock to ensure counter updates are atomic.

        :param num_retries: how many retries should be performed when all connections are currently unavailable
        :returns: A connection from the pool
        """
        connection = None
        should_retry = False

        # Use asyncio.Lock to ensure atomic counter updates (fix B8)
        async with self._async_lock:
            if self.debug_loging_enabled:
                self._logger.debug(
                    f"round_robin_balancer.py: self._connections: {self._connections} , {num_retries}"
                )

            # Filter by circuit breaker state if enabled
            if self._circuit_breaker:
                available_connections = [
                    c
                    for c in self._connections
                    if c is not None and self._circuit_breaker.is_available(c.address)
                ]
                # Fallback: if all circuits are open, use all connections
                if not available_connections:
                    self._logger.warning(
                        f"All circuits open for {self._deployment_name}, using fallback"
                    )
                    available_connections = [
                        c for c in self._connections if c is not None
                    ]
            else:
                available_connections = [c for c in self._connections if c is not None]

            if not available_connections:
                if num_retries <= 0:
                    raise EstablishGrpcConnectionError(
                        f"No connections available for {self._deployment_name}"
                    )
                # Mark for retry outside lock
                if self.debug_loging_enabled:
                    self._logger.debug(
                        f"No valid connection found for {self._deployment_name}, retrying"
                    )
                should_retry = True
            else:
                try:
                    index = self._rr_counter % len(available_connections)
                    if self.debug_loging_enabled:
                        self._logger.debug(
                            f"round_robin_balancer.py: index: {index}, available: {len(available_connections)}"
                        )
                    connection = available_connections[index]
                except IndexError:
                    # This can happen as a race condition while _removing_ connections
                    self._rr_counter = 0
                    connection = (
                        available_connections[0] if available_connections else None
                    )

                if connection is None:
                    if num_retries <= 0:
                        raise EstablishGrpcConnectionError(
                            f"Error while resetting connections for {self._deployment_name}. Connections cannot be used."
                        )
                    should_retry = True
                else:
                    # Advance counter atomically within the lock
                    self._rr_counter = (self._rr_counter + 1) % len(
                        available_connections
                    )

        # Handle retry outside lock to avoid blocking during await
        if should_retry:
            await asyncio.sleep(0)  # yield to event loop
            return await self._get_next_connection(num_retries=num_retries - 1)

        return connection

    def update_connections(self, connections: list):
        """Rebalance the connections."""
        super().update_connections(connections)
        # Fix: use len() not len()-1, and handle empty case properly
        conn_count = len(self._connections)
        if conn_count > 0:
            self._rr_counter = self._rr_counter % conn_count
        else:
            self._rr_counter = 0

    def close(self):
        self._rr_counter = 0
