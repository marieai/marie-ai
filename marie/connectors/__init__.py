"""Connector registration and discovery for Marie-AI."""

from marie.connectors.model import ConnectorManifest, ConnectorsConf
from marie.connectors.registry import ConnectorRegistry

__all__ = ["ConnectorManifest", "ConnectorsConf", "ConnectorRegistry"]
