import asyncio
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class _ClusterState:
    """Shared deployment routing state."""

    _deployment_nodes: Optional[Dict[str, Any]] = None

    deployment_update_event = asyncio.Event()  # Event to signal deployment updates

    @property
    def deployment_nodes(self) -> Dict[str, Any]:
        """Get or initialize a dictionary of nodes for each deployment."""
        if self._deployment_nodes is None:
            self._deployment_nodes = {}
        return self._deployment_nodes

    @deployment_nodes.setter
    def deployment_nodes(self, value: Dict[str, Any]) -> None:
        self._deployment_nodes = value

    def notify_deployment_update(self) -> None:
        self.deployment_update_event.set()
        self.deployment_update_event.clear()


ClusterState = _ClusterState()
