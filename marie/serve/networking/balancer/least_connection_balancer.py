from collections import defaultdict
from typing import TYPE_CHECKING, Optional

from marie.logging_core.logger import MarieLogger
from marie.serve.networking.balancer.load_balancer import LoadBalancer

if TYPE_CHECKING:
    from marie.engine.circuit_breaker import CircuitBreakerConfig

    from marie.serve.networking.connection_stub import _ConnectionStubs


class LeastConnectionsLoadBalancer(LoadBalancer):
    """Select the least-used eligible replica, rotating ties."""

    def __init__(
        self,
        deployment_name: str,
        logger: Optional[MarieLogger] = None,
        circuit_breaker_config: Optional['CircuitBreakerConfig'] = None,
    ) -> None:
        super().__init__(
            deployment_name, logger, circuit_breaker_config=circuit_breaker_config
        )
        self._rr_counter = 0
        self.selection_counter = defaultdict(int)

    def _select_connection(
        self, connections: list['_ConnectionStubs']
    ) -> '_ConnectionStubs':
        min_active = min(self.active_counter[c.address] for c in connections)
        candidates = [
            c for c in connections if self.active_counter[c.address] == min_active
        ]
        if len(candidates) == 1:
            self._rr_counter = 0
        connection = candidates[self._rr_counter % len(candidates)]
        self._rr_counter = (self._rr_counter + 1) % len(candidates)
        self.selection_counter[connection.address] += 1
        return connection

    def _on_connections_updated(self) -> None:
        for address in list(self.selection_counter):
            if address not in self.active_counter:
                del self.selection_counter[address]
        self._rr_counter %= max(1, len(self._connections))

    def _on_closed(self) -> None:
        self._rr_counter = 0
        self.selection_counter.clear()

    def print_selection_stats(self) -> None:
        if self.debug_loging_enabled and self._logger.debug_enabled:
            self._logger.debug(
                "Connection selection stats for %s: %s",
                self._deployment_name,
                self.get_selection_counts(),
            )

    def get_selection_counts(self) -> dict[str, int]:
        """Return cumulative selections of current replicas."""
        with self._lock:
            return dict(self.selection_counter)
