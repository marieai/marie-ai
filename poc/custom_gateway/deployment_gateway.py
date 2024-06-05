from marie import Gateway as BaseGateway
from marie.serve.runtimes.servers.composite import CompositeServer


class MariePodGateway(BaseGateway, CompositeServer):
    """A custom Gateway for Marie deployment pods (Worker nodes) ."""

    def __init__(self, **kwargs):
        """Initialize a new Gateway."""
        super().__init__(**kwargs)
