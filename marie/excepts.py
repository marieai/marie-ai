"""This modules defines all kinds of exceptions raised in Jina."""
from typing import Set, Union, Any, Optional

import grpc.aio

from marie.serve.helper import extract_trailing_metadata


class BaseMarieException(BaseException):
    """A base class for all exceptions raised by Jina"""


class RuntimeFailToStart(SystemError, BaseMarieException):
    """When pod/deployment is failed to started."""


class RuntimeTerminated(KeyboardInterrupt, BaseMarieException):
    """The event loop of BasePod ends."""


class FlowTopologyError(Exception, BaseMarieException):
    """Flow exception when the topology is ambiguous."""


class FlowMissingDeploymentError(Exception, BaseMarieException):
    """Flow exception when a deployment can not be found in the flow."""


class FlowBuildLevelError(Exception, BaseMarieException):
    """Flow exception when required build level is higher than the current build level."""


class BadConfigSource(FileNotFoundError, BaseMarieException):
    """The yaml config file is bad, not loadable or not exist."""


class BadServerFlow(Exception, BaseMarieException):
    """A wrongly defined Flow on the server side"""


class BadClient(Exception, BaseMarieException):
    """A wrongly defined client, can not communicate with jina server correctly."""


class BadServer(Exception, BaseMarieException):
    """Error happens on the server side."""


class BadClientCallback(BadClient, BaseMarieException):
    """Error in the callback function on the client side."""


class BadClientInput(BadClient, BaseMarieException):
    """Error in the request generator function on the client side."""


class BadRequestType(TypeError, BaseMarieException):
    """Exception when can not construct a request object from given data."""


class BadImageNameError(Exception, BaseMarieException):
    """Exception when an image name can not be found either local & remote"""


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
    """Raised when trying to use a port which is already used"""


class EstablishGrpcConnectionError(Exception, BaseMarieException):
    """Raised when Exception occurs when establishing or resetting gRPC connection"""


class InternalNetworkError(grpc.aio.AioRpcError, BaseMarieException):
    """
    Raised when communication between microservices fails.
    Needed to propagate information about the root cause event, such as request_id and dest_addr.
    """

    def __init__(
        self,
        og_exception: grpc.aio.AioRpcError,
        request_id: str = "",
        dest_addr: Union[str, Set[str]] = {""},
        details: str = "",
    ):
        """
        :param og_exception: the original exception that caused the network error
        :param request_id: id of the request that caused the error
        :param dest_addr: destination (microservice) address(es) of the problematic network call(s)
        :param details: details of the error
        """
        self.og_exception = og_exception
        self.request_id = request_id
        self.dest_addr = dest_addr
        self._details = details
        super().__init__(
            og_exception.code(),
            og_exception.initial_metadata(),
            og_exception.trailing_metadata(),
            self.details(),
            og_exception.debug_error_string(),
        )

    def __str__(self):
        return self.details()

    def __repr__(self):
        return self.__str__()

    def code(self):
        """
        :return: error code of this exception
        """
        return self.og_exception.code()

    def details(self):
        """
        :return: details of this exception
        """
        if self._details:
            trailing_metadata = extract_trailing_metadata(self.og_exception)
            if trailing_metadata:
                return f"{self._details}\n{trailing_metadata}"
            else:
                return self._details

        return self.og_exception.details()


def raise_exception(
    e,
    suppress_errors: bool,
    logger: "MarieLogger",  # noqa: F821
    msg: str,
    exc_info: Optional[bool] = False,
) -> Optional[bool]:
    if suppress_errors:
        logger.warning(msg, exc_info=exc_info)
        return False
    else:
        raise BadConfigSource(msg) from e
