from marie.serve.runtimes.gateway.gateway import BaseGateway
from marie.serve.runtimes.servers.websocket import WebSocketServer

__all__ = ['WebSocketGateway']


class WebSocketGateway(WebSocketServer, BaseGateway):
    """
    :class:`WebSocketGateway` is a WebSocketServer that can be loaded from YAML as any other Gateway
    """

    pass
