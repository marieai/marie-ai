from marie.serve.runtimes.gateway.gateway import BaseGateway
from marie.serve.runtimes.gateway.http.fastapi import (  # keep import here for backwards compatibility
    FastAPIBaseGateway,
)
from marie.serve.runtimes.servers.http import HTTPServer

__all__ = ['HTTPGateway']


class HTTPGateway(HTTPServer, BaseGateway):
    """
    :class:`HTTPGateway` is a FastAPIBaseGateway that uses the default FastAPI app
    """

    pass
