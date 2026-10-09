import abc
import threading
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Collection, Optional, Sequence, Union

from marie.excepts import EstablishGrpcConnectionError
from marie.logging_core.logger import MarieLogger
from marie.serve.networking.balancer.interceptor import LoadBalancerInterceptor

if TYPE_CHECKING:
    from marie.engine.circuit_breaker import (
        CircuitBreaker,
        CircuitBreakerConfig,
        CircuitPermit,
    )

    from marie.serve.networking.connection_stub import _ConnectionStubs


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


@dataclass(eq=False)
class ConnectionLease:
    """One outstanding request and its circuit admission permit."""

    connection: '_ConnectionStubs'
    permit: Optional['CircuitPermit'] = None


class LoadBalancer(abc.ABC):
    """Base class for load balancers."""

    def __init__(
        self,
        deployment_name: str,
        logger: Optional[MarieLogger] = None,
        tracing_interceptors: Optional[Sequence[LoadBalancerInterceptor]] = None,
        circuit_breaker_config: Optional["CircuitBreakerConfig"] = None,
    ) -> None:
        self._connections: list['_ConnectionStubs'] = []
        self._deployment_name = deployment_name
        self._logger = logger or MarieLogger(self.__class__.__name__)
        self.active_counter = {}
        self.debug_loging_enabled = False
        self.tracing_interceptors = tracing_interceptors or []
        self._lock = threading.Lock()  # sync lock
        self._closed = False
        self._usage_by_address: dict[str, int] = {}
        self._leases: set[ConnectionLease] = set()
        self._fail_open = (
            circuit_breaker_config.fail_open if circuit_breaker_config else False
        )

        # Initialize circuit breaker if config provided (opt-in)
        self._circuit_breaker: Optional["CircuitBreaker"] = None
        if circuit_breaker_config is not None:
            from marie.engine.circuit_breaker import CircuitBreaker

            self._circuit_breaker = CircuitBreaker(circuit_breaker_config, self._logger)
            self._logger.info(
                f"LoadBalancer: Circuit breaker enabled for {self._deployment_name}"
            )

        self._logger.info(f"LoadBalancer: for {self._deployment_name} initialized.")

    async def get_next_connection(
        self, num_retries: int = 3, exclude_addresses: Optional[Collection[str]] = None
    ) -> '_ConnectionStubs':
        """Select without reserving usage; num_retries is kept for compatibility."""
        with self._lock:
            connection, _ = self._select_available_connection(exclude_addresses)
        for interceptor in self.tracing_interceptors:
            interceptor.on_connection_acquired(connection)
        return connection

    async def acquire_connection(
        self, exclude_addresses: Optional[Collection[str]] = None
    ) -> ConnectionLease:
        """Atomically select a connection, admit its circuit, and count the request."""
        with self._lock:
            connection, permit = self._select_available_connection(
                exclude_addresses, reserve=True
            )
            lease = ConnectionLease(connection, permit)
            self._leases.add(lease)
            self._increment_usage(connection.address)
        try:
            for interceptor in self.tracing_interceptors:
                interceptor.on_connection_acquired(connection)
        except Exception:
            self.release_connection(lease)
            raise
        return lease

    def release_connection(self, lease: ConnectionLease) -> None:
        """Release an acquisition once, including after membership changes."""
        with self._lock:
            if lease not in self._leases:
                return
            self._leases.remove(lease)
            self._decrement_usage(lease.connection.address)
            if self._circuit_breaker and lease.permit:
                self._circuit_breaker.release(lease.permit)
        for interceptor in self.tracing_interceptors:
            try:
                interceptor.on_connection_released(lease.connection)
            except Exception:
                self._logger.warning(
                    'Connection release interceptor failed for %s',
                    self._deployment_name,
                    exc_info=True,
                )

    def _select_available_connection(
        self, exclude_addresses: Optional[Collection[str]], reserve: bool = False
    ) -> tuple['_ConnectionStubs', Optional['CircuitPermit']]:
        if self._closed:
            raise EstablishGrpcConnectionError(
                f'Load balancer closed for {self._deployment_name}'
            )
        excluded = exclude_addresses or ()
        candidates = [c for c in self._connections if c.address not in excluded]
        if self._circuit_breaker:
            healthy = [
                c for c in candidates if self._circuit_breaker.is_available(c.address)
            ]
            if healthy or not self._fail_open:
                candidates = healthy
            else:
                from marie.engine.circuit_breaker import CircuitState

                candidates = [
                    c
                    for c in candidates
                    if self._circuit_breaker.get_state(c.address) == CircuitState.OPEN
                ]
        while candidates:
            connection = self._select_connection(candidates)
            permit = None
            if reserve and self._circuit_breaker:
                permit = self._circuit_breaker.try_acquire(
                    connection.address, allow_open=self._fail_open
                )
                if permit is None:
                    candidates = [c for c in candidates if c is not connection]
                    continue
            return connection, permit
        raise EstablishGrpcConnectionError(
            f'No available connections for {self._deployment_name}'
        )

    @abc.abstractmethod
    def _select_connection(
        self, connections: list['_ConnectionStubs']
    ) -> '_ConnectionStubs':
        raise NotImplementedError

    def update_connections(self, connections: list['_ConnectionStubs']) -> None:
        """
        Rebalance the connections.
        :param connections: List of connections to be used for load balancing.
        """
        with self._lock:
            if self._closed:
                raise EstablishGrpcConnectionError(
                    f'Load balancer closed for {self._deployment_name}'
                )
            new_addresses = {c.address for c in connections}
            old_addresses = set(self.active_counter.keys())
            removed_addresses = old_addresses - new_addresses

            for addr in removed_addresses:
                del self.active_counter[addr]
                if self.debug_loging_enabled:
                    self._logger.debug(
                        f"Cleaned up active_counter for removed address: {addr}"
                    )

            self._connections = list(connections)

            for connection in self._connections:
                self.active_counter[connection.address] = self._usage_by_address.get(
                    connection.address, 0
                )
            if self._circuit_breaker:
                for addr in removed_addresses:
                    self._circuit_breaker.remove_address(addr)
            self._on_connections_updated()

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
        elif load_balancer_type is None:
            return RoundRobinLoadBalancer(
                deployment_name, logger, circuit_breaker_config=circuit_breaker_config
            )
        raise NotImplementedError(
            f'Load balancer policy {load_balancer_type} is not implemented.'
        )

    def close(self) -> None:
        """Close the load balancer."""
        with self._lock:
            self._closed = True
            self._connections.clear()
            if self._circuit_breaker:
                for lease in self._leases:
                    if lease.permit:
                        self._circuit_breaker.release(lease.permit)
                for address in self.active_counter:
                    self._circuit_breaker.remove_address(address)
            self._leases.clear()
            self._usage_by_address.clear()
            self.active_counter.clear()
            self._on_closed()

    def _on_connections_updated(self) -> None:
        pass

    def _on_closed(self) -> None:
        pass

    def incr_usage(self, address: str) -> int:
        """
        Increment connection with address as in use
        :param address: Address of the connection
        """
        with self._lock:
            if self._closed or address not in self.active_counter:
                raise EstablishGrpcConnectionError(
                    f'Unknown connection {address} for {self._deployment_name}'
                )
            return self._increment_usage(address)

    def _increment_usage(self, address: str) -> int:
        count = self._usage_by_address.get(address, 0) + 1
        self._usage_by_address[address] = count
        self.active_counter[address] = count
        return count

    def decr_usage(self, address: str) -> int:
        """
        Decrement connection with address as not in use
        :param address: Address of the connection
        """
        with self._lock:
            return self._decrement_usage(address)

    def _decrement_usage(self, address: str) -> int:
        count = max(0, self._usage_by_address.get(address, 0) - 1)
        if count:
            self._usage_by_address[address] = count
        else:
            self._usage_by_address.pop(address, None)
        if address in self.active_counter:
            self.active_counter[address] = count
        return count

    def get_active_count(self, address: str) -> int:
        """Get the number of active requests for a given address"""
        with self._lock:
            return self.active_counter.get(address, 0)

    def get_active_counter(self) -> dict[str, int]:
        """
        Get the active counter for all the connections
        :return:
        """
        with self._lock:
            return dict(self.active_counter)

    def get_selection_counts(self) -> dict[str, int]:
        """Return cumulative selections by address when tracked."""
        return {}

    def connection_count(self) -> int:
        """
        Get the number of connections
        :return:
        """
        with self._lock:
            return len(self._connections)

    # Circuit breaker methods

    def record_failure(
        self, address: str, lease: Optional[ConnectionLease] = None
    ) -> None:
        """
        Record a failure for the given address.
        If circuit breaker is enabled, this may cause the circuit to open.

        :param address: The address that experienced a failure.
        """
        if self._circuit_breaker:
            self._circuit_breaker.record_failure(
                address, permit=lease.permit if lease else None
            )

    def record_success(
        self, address: str, lease: Optional[ConnectionLease] = None
    ) -> None:
        """
        Record a success for the given address.
        If circuit breaker is enabled, this may help close an open circuit.

        :param address: The address that experienced a success.
        """
        if self._circuit_breaker:
            self._circuit_breaker.record_success(
                address, permit=lease.permit if lease else None
            )

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
