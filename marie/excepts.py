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


class BadRequestType(TypeError, BaseMarieException):
    """Exception when can not construct a request object from given data."""


class BadYAMLVersion(Exception, BaseMarieException):
    """Exception when YAML config specifies a wrong version number."""


class NotSupportedError(Exception, BaseMarieException):
    """Exception when user accidentally using a retired argument."""


class RuntimeRunForeverEarlyError(Exception, BaseMarieException):
    """Raised when an error occurs when starting the run_forever of Runtime"""


class DockerVersionError(SystemError, BaseMarieException):
    """Raised when the docker version is incompatible"""


class NoContainerizedError(Exception, BaseMarieException):
    """Raised when trying to use non-containerized Executor in K8s or Docker Compose"""


class PortAlreadyUsed(RuntimeError, BaseMarieException):
    """Raised when to use a port which is already used"""


class FlowMissingDeploymentError(Exception, BaseMarieException):
    """Flow exception when a deployment can not be found in the flow."""

class FlowTopologyError(Exception, BaseMarieException):
    """Flow exception when the topology is ambiguous."""

class RuntimeFailToStart(SystemError, BaseMarieException):
    """When pod/deployment is failed to started."""


class BadClient(Exception, BaseMarieException):
    """A wrongly defined client, can not communicate with jina server correctly."""

class BadClientCallback(BadClient, BaseMarieException):
    """Error in the callback function on the client side."""


class BadClientInput(BadClient, BaseMarieException):
    """Error in the request generator function on the client side."""
