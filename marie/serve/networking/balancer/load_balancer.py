import abc
from enum import Enum
from typing import Union

from marie.logging.logger import MarieLogger


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


class LoadBalancer(metaclass=abc.ABCMeta):
    """Base class for load balancers."""

    def __init__(self, deployment_name: str, logger):
        self._connections = []
        self._deployment_name = deployment_name
        self._logger = logger
        self.active_counter = {}

    @abc.abstractmethod
    async def get_next_connection(self, num_retries=3):
        """
        Returns the next connection to be used based on the load balancing algorithm.
        :param num_retries:  Number of times to retry if the connection is not available.
        """
        ...

    def update_connections(self, connections):
        """
        Rebalance the connections.
        :param connections: List of connections to be used for load balancing.
        """
        self._connections = connections
        for connection in self._connections:
            if connection.address not in self.active_counter:
                self.active_counter[connection.address] = 0

    @staticmethod
    def get_load_balancer(
        load_balancer_type: Union[LoadBalancerType, str],
        deployment_name: str,
        logger: MarieLogger,
    ) -> "LoadBalancer":
        """
        Get the load balancer based on the type.
        :param logger:
        :param load_balancer_type: Type of load balancer.
        :param deployment_name: Name of the deployment.
        :return:
        """
        from marie.serve.networking.balancer.round_robin_balancer import (
            RoundRobinLoadBalancer,
        )
        from marie.serve.networking.balancer.least_connection_balancer import (
            LeastConnectionsLoadBalancer,
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
        self._logger.debug("Incrementing usage for address", address)
        self.active_counter[address] = self.active_counter.get(address, 0) + 1

        self._logger.debug(f'incr_usage: self.active_counter: {self.active_counter}')

        return self.active_counter[address]

    def decr_usage(self, address: str) -> int:
        """
        Decrement connection with address as not in use
        :param address: Address of the connection
        """
        self._logger.debug("Decrementing usage for address", address)
        self.active_counter[address] = max(0, self.active_counter.get(address, 0) - 1)

        self._logger.debug(f'decr_usage: self.active_counter: {self.active_counter}')
        return self.active_counter[address]

    def get_active_count(self, address: str) -> int:
        """Get the number of active requests for a given address"""
        return self.active_counter.get(address, 0)

    def get_active_counter(self):
        return self.active_counter
