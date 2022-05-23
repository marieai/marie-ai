class BaseMarieException(BaseException):
    """A base class for all exceptions raised by Marie"""


class RuntimeException(BaseMarieException):
    """Default runtime exception."""


class RuntimeTerminated(KeyboardInterrupt, BaseMarieException):
    """The event loop of BasePod ends."""


class BadConfigSource(FileNotFoundError, BaseMarieException):
    """The yaml config file is bad, not loadable or not exist."""


class PortAlreadyUsed(RuntimeError, BaseMarieException):
    """Raised when to use a port which is already used"""


class InternalNetworkError(BaseMarieException):
    """Internal network exception."""
