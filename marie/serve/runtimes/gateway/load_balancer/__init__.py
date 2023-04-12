from marie.serve.runtimes.gateway.gateway import BaseGateway
from marie.serve.runtimes.servers.load_balancer import LoadBalancingServer

__all__ = ['LoadBalancerGateway']


class LoadBalancerGateway(LoadBalancingServer, BaseGateway):
    """
    :class:`LoadBalancerGateway`
    """

    pass
