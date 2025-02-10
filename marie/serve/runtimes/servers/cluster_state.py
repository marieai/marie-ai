import asyncio
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class _ClusterState:
    """_ClusterState include fields for deployments and nodes."""

    _deployments: Optional[Dict[str, Any]] = None
    _deployment_nodes: Optional[Dict[str, Any]] = None
    _deployments_last_updated: Optional[float] = -1

    scheduled_event = asyncio.Event()  # Notification event for job scheduling

    @property
    def deployments(self) -> Dict[str, Any]:
        """Get or initialize a dictionary of deployments (e.g., their metadata)."""
        if self._deployments is None:
            self._deployments = {}
        return self._deployments

    @deployments.setter
    def deployments(self, value: Dict[str, Any]) -> None:
        self._deployments = value
        self._deployments_last_updated = time.time()

    @property
    def deployment_nodes(self) -> Dict[str, Any]:
        """Get or initialize a dictionary of nodes for each deployment."""
        if self._deployment_nodes is None:
            self._deployment_nodes = {}
        return self._deployment_nodes

    @property
    def deployments_last_updated(self) -> float:
        return self._deployments_last_updated

    @deployment_nodes.setter
    def deployment_nodes(self, value: Dict[str, Any]) -> None:
        self._deployment_nodes = value


ClusterState = _ClusterState()
