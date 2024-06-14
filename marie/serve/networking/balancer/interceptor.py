import abc
from typing import Callable


class LoadBalancerInterceptor(abc.ABC):
    """
    Base class for load balancer interceptors, that can be used to intercept the connection acquisition process
    and provide callbacks.

    """

    @abc.abstractmethod
    async def on_connection_acquired(self, connection, callback: Callable = None):
        """
        Called when a connection is acquired from the load balancer.
        """
        pass

    @abc.abstractmethod
    async def on_connection_released(self, connection, callback: Callable = None):
        """
        Called when a connection is released back to the load balancer.
        """
        pass

    @abc.abstractmethod
    async def on_connection_failed(
        self, connection, exception, callback: Callable = None
    ):
        """
        Called when a connection attempt fails.
        """
        pass
