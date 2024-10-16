from marie.serve.runtimes.gateway.gateway import BaseGateway
from marie.serve.runtimes.servers.marie_gateway import MarieServerGateway

__all__ = ["MarieGateway"]


class MarieGateway(MarieServerGateway, BaseGateway):
    """
    :class:`MarieGateway` is a MarieServerGateway that can be loaded from YAML as any other Gateway
    """

    pass
