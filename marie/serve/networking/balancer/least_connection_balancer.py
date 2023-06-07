from marie.excepts import EstablishGrpcConnectionError
from marie.logging.logger import MarieLogger
from marie.serve.networking.balancer.load_balancer import LoadBalancer


class LeastConnectionsLoadBalancer(LoadBalancer):
    """
    Least connections load balancer.

    Find the server/s with the lowest active connections.
    If there are multiple servers with the same number of active connections, use round robin to select one of them.

    """

    def __init__(self, deployment_name: str, logger: MarieLogger):
        super().__init__(deployment_name, logger)
        self._rr_counter = 0  # round robin counter

    async def get_next_connection(self, num_retries=3):
        return await self._get_next_connection(num_retries=num_retries)

    async def _get_next_connection(self, num_retries=3):
        # Find the connection with the least active connections
        min_active_connections = int(1e9)
        for connection in self._connections:
            if self.active_counter[connection.address] < min_active_connections:
                min_active_connections = self.active_counter[connection.address]

        # get all connections with the least active connections
        min_use_connections = []
        for connection in self._connections:
            if self.active_counter[connection.address] == min_active_connections:
                min_use_connections.append(connection)

        if len(min_use_connections) == 1:
            self._rr_counter = 0

        self._logger.debug(
            f'least_connection_balancer.py: min_use_connections: {min_use_connections}'
            f' min_active_connections: {min_active_connections}'
            f' self._rr_counter: {self._rr_counter}'
        )

        # Round robin between the connections with the least active connections
        try:
            connection = None
            for i in range(len(min_use_connections)):
                internal_rr_counter = (self._rr_counter + i) % len(min_use_connections)
                connection = min_use_connections[internal_rr_counter]
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
            connection = min_use_connections[self._rr_counter]
        self._rr_counter = (self._rr_counter + 1) % len(min_use_connections)
        return connection

    def close(self):
        pass
