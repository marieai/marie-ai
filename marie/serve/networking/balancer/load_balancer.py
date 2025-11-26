import abc
import threading
from enum import Enum
from typing import TYPE_CHECKING, Optional, Sequence, Union

from marie.excepts import EstablishGrpcConnectionError
from marie.logging_core.logger import MarieLogger
from marie.serve.networking.balancer.interceptor import LoadBalancerInterceptor

if TYPE_CHECKING:
    from marie.serve.networking.balancer.circuit_breaker import (
        CircuitBreaker,
        CircuitBreakerConfig,
    )


class LoadBalancerType(Enum):
    """
    Enum for the type of load balancer to be used.
    """

    ROUND_ROBIN = "ROUND_ROBIN"
    LEAST_CONNECTION = "LEAST_CONNECTION"  # SHORTEST QUEUE
    CONSISTENT_HASHING = "CONSISTENT_HASHING"  # TODO: Implement this
    RANDOM = "RANDOM"  # TODO: Implement this

    @staticmethod
    def from_value(value: str) -> "LoadBalancerType":
        """
        Get the load balancer type from the value.
        :param value: Value of the load balancer type.
        :return: LoadBalancerType
        """
        if value is None or value == "":
            return LoadBalancerType.ROUND_ROBIN
        for data in LoadBalancerType:
            if data.value.lower() == value.lower():
                return data

        raise ValueError(
            f"Invalid load balancer type: {value}. Supported types are: {[item.value for item in LoadBalancerType]}."
        )


class LoadBalancer(abc.ABC):
    """Base class for load balancers."""

    def __init__(
        self,
        deployment_name: str,
        logger: Optional[MarieLogger] = None,
        tracing_interceptors: Optional[Sequence[LoadBalancerInterceptor]] = None,
        circuit_breaker_config: Optional["CircuitBreakerConfig"] = None,
    ):
        self._connections = []
        self._deployment_name = deployment_name
        self._logger = logger or MarieLogger(self.__class__.__name__)
        self.active_counter = {}
        self.debug_loging_enabled = False
        self.tracing_interceptors = tracing_interceptors or []
        self._lock = threading.Lock()  # sync lock

        # Initialize circuit breaker if config provided (opt-in)
        self._circuit_breaker: Optional["CircuitBreaker"] = None
        if circuit_breaker_config is not None:
            from marie.serve.networking.balancer.circuit_breaker import CircuitBreaker

            self._circuit_breaker = CircuitBreaker(circuit_breaker_config, self._logger)
            self._logger.info(
                f"LoadBalancer: Circuit breaker enabled for {self._deployment_name}"
            )

        self._logger.info(f"LoadBalancer: for {self._deployment_name} initialized.")

    async def get_next_connection(self, num_retries=3):
        """
        Returns the next connection to be used based on the load balancing algorithm.
        :param num_retries:  Number of times to retry if the connection is not available.
        """
        connection = await self._get_next_connection(num_retries=num_retries)

        if connection is None:
            raise EstablishGrpcConnectionError(
                f"Error while acquiring connection {self._deployment_name}. Connection cannot be used."
            )

        for interceptor in self.tracing_interceptors:
            interceptor.on_connection_acquired(connection)

        return connection

    @abc.abstractmethod
    async def _get_next_connection(self, num_retries=3):
        """
        Implementation that returns the next connection to be used based on the load balancing algorithm.
        :param num_retries:  Number of times to retry if the connection is not available.
        """
        raise NotImplementedError

    def update_connections(self, connections: list):
        """
        Rebalance the connections.
        :param connections: List of connections to be used for load balancing.
        """
        removed_addresses = set()
        with self._lock:
            # Calculate addresses to remove (memory leak fix)
            new_addresses = {c.address for c in connections}
            old_addresses = set(self.active_counter.keys())
            removed_addresses = old_addresses - new_addresses

            # Clean up removed addresses from active_counter
            for addr in removed_addresses:
                del self.active_counter[addr]
                if self.debug_loging_enabled:
                    self._logger.debug(
                        f"Cleaned up active_counter for removed address: {addr}"
                    )

            # Update connections list
            self._connections = list(connections)

            # Initialize counters for new connections
            for connection in self._connections:
                if connection.address not in self.active_counter:
                    self.active_counter[connection.address] = 0

        # Clean up circuit breaker state for removed addresses (outside lock)
        if self._circuit_breaker:
            for addr in removed_addresses:
                self._circuit_breaker.remove_address(addr)

        if self.debug_loging_enabled:
            self._logger.debug(
                f"update_connections: self._connections: {self._connections}"
            )

        for interceptor in self.tracing_interceptors:
            interceptor.on_connections_updated(self._connections)

    @staticmethod
    def create_load_balancer(
        load_balancer_type: Union[LoadBalancerType, str],
        deployment_name: str,
        logger: MarieLogger,
        circuit_breaker_config: Optional["CircuitBreakerConfig"] = None,
    ) -> "LoadBalancer":
        """
        Get the load balancer based on the type.
        :param logger: Logger to be used.
        :param load_balancer_type: Type of load balancer.
        :param deployment_name: Name of the deployment.
        :param circuit_breaker_config: Optional circuit breaker configuration.
        :return:
        """
        from marie.serve.networking.balancer.least_connection_balancer import (
            LeastConnectionsLoadBalancer,
        )
        from marie.serve.networking.balancer.round_robin_balancer import (
            RoundRobinLoadBalancer,
        )

        if isinstance(load_balancer_type, str):
            load_balancer_type = LoadBalancerType.from_value(load_balancer_type)

        if load_balancer_type == LoadBalancerType.ROUND_ROBIN:
            return RoundRobinLoadBalancer(
                deployment_name, logger, circuit_breaker_config=circuit_breaker_config
            )
        elif load_balancer_type == LoadBalancerType.LEAST_CONNECTION:
            return LeastConnectionsLoadBalancer(
                deployment_name, logger, circuit_breaker_config=circuit_breaker_config
            )
        elif load_balancer_type == LoadBalancerType.RANDOM:
            raise NotImplementedError("Random load balancer not implemented yet.")
        else:
            return RoundRobinLoadBalancer(
                deployment_name, logger, circuit_breaker_config=circuit_breaker_config
            )

    @abc.abstractmethod
    def close(self):
        """Close the load balancer."""
        ...

    def incr_usage(self, address: str) -> int:
        """
        Increment connection with address as in use
        :param address: Address of the connection
        """
        with self._lock:
            self._logger.debug(f"Incrementing usage for address : {address}")
            self.active_counter[address] = self.active_counter.get(address, 0) + 1
            self._logger.debug(f"incr_usage: {self.active_counter}")
            return self.active_counter[address]

    def decr_usage(self, address: str) -> int:
        """
        Decrement connection with address as not in use
        :param address: Address of the connection
        """
        with self._lock:
            self._logger.debug(f"Decrementing usage for address: {address}")
            self.active_counter[address] = max(
                0, self.active_counter.get(address, 0) - 1
            )
            self._logger.debug(f"decr_usage: {self.active_counter}")
            return self.active_counter[address]

    def get_active_count(self, address: str) -> int:
        """Get the number of active requests for a given address"""
        with self._lock:
            return self.active_counter.get(address, 0)

    def get_active_counter(self) -> dict:
        """
        Get the active counter for all the connections
        :return:
        """
        with self._lock:
            return dict(self.active_counter)

    def connection_count(self) -> int:
        """
        Get the number of connections
        :return:
        """
        return len(self._connections)

    # Circuit breaker methods

    def record_failure(self, address: str) -> None:
        """
        Record a failure for the given address.
        If circuit breaker is enabled, this may cause the circuit to open.

        :param address: The address that experienced a failure.
        """
        if self._circuit_breaker:
            self._circuit_breaker.record_failure(address)

    def record_success(self, address: str) -> None:
        """
        Record a success for the given address.
        If circuit breaker is enabled, this may help close an open circuit.

        :param address: The address that experienced a success.
        """
        if self._circuit_breaker:
            self._circuit_breaker.record_success(address)

    def is_connection_available(self, address: str) -> bool:
        """
        Check if a connection is available (not circuit-broken).

        :param address: The address to check.
        :return: True if the connection can be used, False if circuit is open.
        """
        if self._circuit_breaker is None:
            return True
        return self._circuit_breaker.is_available(address)

    def get_available_connections(self) -> list:
        """
        Get list of connections that are available (not circuit-broken).

        :return: List of available connections.
        """
        if self._circuit_breaker is None:
            return list(self._connections)

        return [
            conn
            for conn in self._connections
            if self._circuit_breaker.is_available(conn.address)
        ]

    def has_circuit_breaker(self) -> bool:
        """
        Check if circuit breaker is enabled.

        :return: True if circuit breaker is enabled.
        """
        return self._circuit_breaker is not None

    def get_circuit_breaker_stats(self) -> Optional[dict]:
        """
        Get circuit breaker statistics for all addresses.

        :return: Dictionary of address to stats, or None if circuit breaker is disabled.
        """
        if self._circuit_breaker is None:
            return None
        return {
            addr: {
                "state": stats.state.value,
                "consecutive_failures": stats.consecutive_failures,
                "total_failures": stats.total_failures,
                "total_successes": stats.total_successes,
            }
            for addr, stats in self._circuit_breaker.get_all_stats().items()
        }
