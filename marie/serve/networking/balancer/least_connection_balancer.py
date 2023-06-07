from marie.excepts import EstablishGrpcConnectionError
from marie.logging.logger import MarieLogger
from marie.serve.networking.balancer.load_balancer import LoadBalancer


class LeastConnectionsLoadBalancer(LoadBalancer):
    def __init__(self, deployment_name: str, logger: MarieLogger):
        super().__init__(deployment_name, logger)
        self._rr_counter = 0  # round robin counter

    async def get_next_connection(self, num_retries=3):
        return await self._get_next_connection(num_retries=num_retries)

    async def _get_next_connection(self, num_retries=3):
        self._logger.info("LeastConnectionsLoadBalancer._get_next_connection")
        org_connections = self._connections.copy()
        self._connections.sort(key=lambda x: self.active_counter[x.address])

        # if the order is different, reset the round-robin counter
        for i in range(len(self._connections)):
            if org_connections[i].address != self._connections[i].address:
                self._rr_counter = 0
                break

        self._logger.debug(f"connections B: {org_connections}")
        self._logger.debug(f"connections A: {self._connections}")
        try:
            connection = None
            for i in range(len(self._connections)):
                internal_rr_counter = (self._rr_counter + i) % len(self._connections)
                connection = self._connections[internal_rr_counter]
                # connection is None if it is currently being reset. In that case, try different connection
                if connection is not None:
                    break
            all_connections_unavailable = connection is None and num_retries <= 0
            if all_connections_unavailable:
                if num_retries <= 0:
                    raise EstablishGrpcConnectionError(
                        f'Error while resetting connections {self._connections} for {self._deployment_name}. Connections cannot be used.'
                    )
            elif connection is None:
                # give control back to async event loop so connection resetting can be completed; then retry
                self._logger.debug(
                    f' No valid connection found for {self._deployment_name}, give chance for potential resetting of connection'
                )
                return await self._get_next_connection(num_retries=num_retries - 1)
        except IndexError:
            # This can happen as a race condition while _removing_ connections
            self._rr_counter = 0
            connection = self._connections[self._rr_counter]
        self._rr_counter = (self._rr_counter + 1) % len(self._connections)
        return connection

    def update_connections(self, connections: list):
        """Rebalance the connections."""
        self._connections = connections
        for connection in self._connections:
            if connection.address not in self.active_counter:
                self.active_counter[connection.address] = 0

    def close(self):
        pass
