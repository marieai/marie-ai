from marie.serve.runtimes.gateway.gateway import BaseGateway
from marie.serve.runtimes.servers.grpc import GRPCServer

__all__ = ['GRPCGateway']


class GRPCGateway(GRPCServer, BaseGateway):
    """
    :class:`GRPCGateway` is a GRPCServer that can be loaded from YAML as any other Gateway
    """

    pass
