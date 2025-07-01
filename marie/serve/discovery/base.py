import abc
import time
from enum import Enum


class ConnectionState(Enum):
    """Enum representing etcd connection states"""

    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"


class ConnectionEvent:
    """Event object for connection state changes"""

    def __init__(
        self,
        old_state: ConnectionState,
        new_state: ConnectionState,
        error: Exception = None,
    ):
        self.old_state = old_state
        self.new_state = new_state
        self.error = error
        self.timestamp = time.time()


class ServiceRegistry(abc.ABC):
    """A service registry."""

    @abc.abstractmethod
    def register(self, service_names, service_addr, service_ttl):
        """Register services with the specific address."""
        raise NotImplementedError

    @abc.abstractmethod
    def heartbeat(self, service_addr=None):
        """Service registry heartbeat."""
        raise NotImplementedError

    @abc.abstractmethod
    def unregister(self, service_names, service_addr):
        """Unregister services with the same address."""
        raise NotImplementedError


class ServiceResolver(abc.ABC):
    """
    This class provides a blueprint for implementing service resolution,
    updates, and event listening. Subclasses inheriting this class must
    provide concrete implementations of the abstract methods.
    """

    @abc.abstractmethod
    def resolve(self, name: str):
        raise NotImplementedError

    @abc.abstractmethod
    def update(self, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def listen(self):
        raise NotImplementedError
