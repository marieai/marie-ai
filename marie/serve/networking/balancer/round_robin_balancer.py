from marie.excepts import EstablishGrpcConnectionError
from marie.logging.logger import MarieLogger
from marie.serve.networking.balancer.load_balancer import LoadBalancer


class RoundRobinLoadBalancer(LoadBalancer):
    """
    Round robin load balancer.
    """

    def __init__(self, deployment_name: str, logger: MarieLogger):
        super().__init__(deployment_name, logger)
        self._rr_counter = 0  # round robin counter

    async def get_next_connection(self, num_retries=3):
        return await self._get_next_connection(num_retries=num_retries)

    async def _get_next_connection(self, num_retries=3):
        """
        :param num_retries: how many retries should be performed when all connections are currently unavailable
        :returns: A connection from the pool
        """
        self._logger.debug(
            f'round_robin_balancer.py: self._connections: {self._connections} , {num_retries}'
        )
        try:
            connection = None
            for i in range(len(self._connections)):
                internal_rr_counter = (self._rr_counter + i) % len(self._connections)
                self._logger.debug(
                    f'round_robin_balancer.py: internal_rr_counter: {internal_rr_counter}'
                )
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
        super().update_connections(connections)
        self._rr_counter = (
            self._rr_counter % (len(self._connections) - 1)
            if (len(self._connections) - 1)
            else 0
        )

    def close(self):
        self._rr_counter = 0
