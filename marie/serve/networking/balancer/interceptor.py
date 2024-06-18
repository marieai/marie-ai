import abc


class LoadBalancerInterceptor(abc.ABC):
    """
    Base class for load balancer interceptors, that can be used to intercept the connection acquisition process
    and provide callbacks.

    """

    @abc.abstractmethod
    def on_connection_acquired(self, connection):
        """
        Called when a connection is acquired from the load balancer.
        """
        pass

    @abc.abstractmethod
    def on_connection_released(self, connection):
        """
        Called when a connection is released back to the load balancer.
        """
        pass

    @abc.abstractmethod
    def on_connection_failed(self, connection, exception):
        """
        Called when a connection attempt fails.
        """
        pass

    @abc.abstractmethod
    def on_connections_updated(self, connections):
        """
        Called when a connections have been updated.
        This can be used to update the internal state of the interceptor.
        """
        pass
