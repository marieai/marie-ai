from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class _ClusterState:
    """_ClusterState include fields for deployments and nodes."""

    _deployments: Optional[Dict[str, Any]] = None
    _deployment_nodes: Optional[Dict[str, Any]] = None

    @property
    def deployments(self) -> Dict[str, Any]:
        """Get or initialize a dictionary of deployments (e.g., their metadata)."""
        if self._deployments is None:
            self._deployments = {}
        return self._deployments

    @deployments.setter
    def deployments(self, value: Dict[str, Any]) -> None:
        self._deployments = value

    @property
    def deployment_nodes(self) -> Dict[str, Any]:
        """Get or initialize a dictionary of nodes for each deployment."""
        if self._deployment_nodes is None:
            self._deployment_nodes = {}
        return self._deployment_nodes

    @deployment_nodes.setter
    def deployment_nodes(self, value: Dict[str, Any]) -> None:
        self._deployment_nodes = value


ClusterState = _ClusterState()
