import abc
from enum import Enum
from typing import Optional, Sequence, Union

from marie.excepts import EstablishGrpcConnectionError
from marie.logging.logger import MarieLogger
from marie.serve.networking.balancer.interceptor import LoadBalancerInterceptor


class LoadBalancerType(Enum):
    """
    Enum for the type of load balancer to be used.
    """

    ROUND_ROBIN = "ROUND_ROBIN"
    LEAST_CONNECTION = "LEAST_CONNECTION"
    CONSISTENT_HASHING = "CONSISTENT_HASHING"  # TODO: Implement this
    RANDOM = "RANDOM"  # TODO: Implement this

    @staticmethod
    def from_value(value: str):
        """
        Get the load balancer type from the value.
        :param value:
        :return:
        """
        if value is None or value == "":
            return LoadBalancerType.ROUND_ROBIN
        for data in LoadBalancerType:
            if data.value.lower() == value.lower():
                return data
        return LoadBalancerType.ROUND_ROBIN


class LoadBalancer(abc.ABC):
    """Base class for load balancers."""

    def __init__(
        self,
        deployment_name: str,
        logger: Optional[MarieLogger] = None,
        tracing_interceptors: Optional[Sequence[LoadBalancerInterceptor]] = None,
    ):
        self._connections = []
        self._deployment_name = deployment_name
        self._logger = logger or MarieLogger(self.__class__.__name__)
        self.active_counter = {}
        self.debug_loging_enabled = False
        self.tracing_interceptors = tracing_interceptors or []

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
        self._connections = connections
        for connection in self._connections:
            if connection.address not in self.active_counter:
                self.active_counter[connection.address] = 0

        if self.debug_loging_enabled:
            self._logger.debug(
                f"update_connections: self._connections: {self._connections}"
            )

        for interceptor in self.tracing_interceptors:
            interceptor.on_connections_updated(self._connections)

    @staticmethod
    def get_load_balancer(
        load_balancer_type: Union[LoadBalancerType, str],
        deployment_name: str,
        logger: MarieLogger,
    ) -> "LoadBalancer":
        """
        Get the load balancer based on the type.
        :param logger: Logger to be used.
        :param load_balancer_type: Type of load balancer.
        :param deployment_name: Name of the deployment.
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
            return RoundRobinLoadBalancer(deployment_name, logger)
        elif load_balancer_type == LoadBalancerType.LEAST_CONNECTION:
            return LeastConnectionsLoadBalancer(deployment_name, logger)
        elif load_balancer_type == LoadBalancerType.RANDOM:
            raise NotImplementedError("Random load balancer not implemented yet.")
        else:
            return RoundRobinLoadBalancer(deployment_name, logger)

    @abc.abstractmethod
    def close(self):
        """Close the load balancer."""
        ...

    def incr_usage(self, address: str) -> int:
        """
        Increment connection with address as in use
        :param address: Address of the connection
        """
        self._logger.debug(f"Incrementing usage for address : {address}")
        self.active_counter[address] = self.active_counter.get(address, 0) + 1

        self._logger.debug(f"incr_usage: self.active_counter: {self.active_counter}")

        return self.active_counter[address]

    def decr_usage(self, address: str) -> int:
        """
        Decrement connection with address as not in use
        :param address: Address of the connection
        """
        self._logger.debug(f"Decrementing usage for address: {address}")
        self.active_counter[address] = max(0, self.active_counter.get(address, 0) - 1)

        self._logger.debug(f"decr_usage: self.active_counter: {self.active_counter}")
        return self.active_counter[address]

    def get_active_count(self, address: str) -> int:
        """Get the number of active requests for a given address"""
        return self.active_counter.get(address, 0)

    def get_active_counter(self) -> dict:
        """
        Get the active counter for all the connections
        :return:
        """
        return self.active_counter

    def connection_count(self) -> int:
        """
        Get the number of connections
        :return:
        """
        return len(self._connections)
