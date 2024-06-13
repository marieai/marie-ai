import asyncio

from marie import Gateway as BaseGateway
from marie.serve.runtimes.servers.composite import CompositeServer
from marie.serve.runtimes.servers.grpc import GRPCServer


class MariePodGateway(BaseGateway, CompositeServer):
    """A custom Gateway for Marie deployment pods (Worker nodes) ."""

    def __init__(self, **kwargs):
        """Initialize a new Gateway."""
        print("MariePodGateway init called")
        super().__init__(**kwargs)

    async def setup_server(self):
        """
        setup servers inside MariePodGateway
        """
        self.logger.debug(f"Setting up MariePodGateway server")
        await super().setup_server()
        self.logger.debug(f"MariePodGateway server setup successful")

        for server in self.servers:
            if isinstance(server, GRPCServer):
                print(f"Registering GRPC server {server}")
                host = server.host
                port = server.port
                ctrl_address = f"{host}:{port}"
                self._register_gateway(ctrl_address)

    def _register_gateway(self, ctrl_address: str):
        """Check if the gateway is ready."""
        print("Registering gateway with controller at : ", ctrl_address)

        async def _async_wait_all(ctrl_address: str):
            """Wait for all servers to be ready."""

            print("waiting for all servers to be ready at : ", ctrl_address)
            while True:
                print(f"checking is ready at {ctrl_address}")
                res = GRPCServer.is_ready(ctrl_address)
                print(f"res: {res}")
                if res:
                    print(f"Gateway is ready at {ctrl_address}")
                    break
                await asyncio.sleep(1)

        asyncio.create_task(_async_wait_all(ctrl_address))
        print("Done waiting for all servers to be ready")
