from typing import TYPE_CHECKING, Optional, Sequence

from marie.logging_core.logger import MarieLogger
from marie.serve.networking.balancer.interceptor import LoadBalancerInterceptor
from marie.serve.networking.balancer.load_balancer import LoadBalancer

if TYPE_CHECKING:
    from marie.engine.circuit_breaker import CircuitBreakerConfig

    from marie.serve.networking.connection_stub import _ConnectionStubs


class RoundRobinLoadBalancer(LoadBalancer):
    """Select current eligible replicas in round-robin order."""

    def __init__(
        self,
        deployment_name: str,
        logger: Optional[MarieLogger] = None,
        tracing_interceptors: Optional[Sequence[LoadBalancerInterceptor]] = None,
        circuit_breaker_config: Optional['CircuitBreakerConfig'] = None,
    ) -> None:
        super().__init__(
            deployment_name, logger, tracing_interceptors, circuit_breaker_config
        )
        self._rr_counter = 0

    def _select_connection(
        self, connections: list['_ConnectionStubs']
    ) -> '_ConnectionStubs':
        connection = connections[self._rr_counter % len(connections)]
        self._rr_counter = (self._rr_counter + 1) % len(connections)
        return connection

    def _on_connections_updated(self) -> None:
        self._rr_counter %= max(1, len(self._connections))

    def _on_closed(self) -> None:
        self._rr_counter = 0
