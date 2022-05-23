class BaseMarieException(BaseException):
    """A base class for all exceptions raised by Marie"""


class RuntimeException(BaseMarieException):
    """Default runtime exception."""


class RuntimeTerminated(KeyboardInterrupt, BaseMarieException):
    """The event loop of BasePod ends."""


class InternalNetworkError(BaseMarieException):
    """Internal network exception."""

