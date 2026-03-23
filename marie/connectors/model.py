from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class ConnectorOperationMeta(BaseModel):
    name: str
    display_name: str


class ConnectorResourceMeta(BaseModel):
    name: str
    display_name: str
    operations: List[ConnectorOperationMeta] = []


class ConnectorCredentialMeta(BaseModel):
    module: str
    name: str
    auth_type: str = "api_key"


class ConnectorRunnerMeta(BaseModel):
    op: str
    timeout_sec: int = 30
    max_retries: int = 2


class ConnectorBackendMeta(BaseModel):
    query_definition_module: str
    executor_module: str
    query_type: str
    endpoint: str
    credentials: List[ConnectorCredentialMeta] = []


class ConnectorManifest(BaseModel):
    """Parsed connector.yaml manifest."""

    marie_connector: str = "1.0"
    id: str
    version: str = "1.0.0"
    display_name: str
    description: str = ""
    icon: str = "Plug"
    category: str = "Connectors"
    tags: List[str] = []
    backend: ConnectorBackendMeta
    resources: List[ConnectorResourceMeta] = []
    runner: ConnectorRunnerMeta

    # Set by registry during registration, not from YAML
    source_type: str = "builtin"
    package_name: Optional[str] = None

    @classmethod
    def from_yaml_file(
        cls, path: str, package_name: Optional[str] = None
    ) -> "ConnectorManifest":
        import yaml

        with open(path, "r") as f:
            data = yaml.safe_load(f)
        manifest = cls(**data)
        if package_name:
            manifest.package_name = package_name
        return manifest

    @classmethod
    def from_yaml_str(
        cls, content: str, package_name: Optional[str] = None
    ) -> "ConnectorManifest":
        import yaml

        data = yaml.safe_load(content)
        manifest = cls(**data)
        if package_name:
            manifest.package_name = package_name
        return manifest


@dataclass
class ConnectorConf:
    """Explicit connector entry (name + py_module). Mirrors PlannerConf."""

    name: str
    py_module: str

    def __post_init__(self):
        if not self.name:
            raise ValueError("Connector name cannot be empty")
        if not self.py_module:
            raise ValueError("Connector py_module cannot be empty")


@dataclass
class DiscoverPackageConf:
    """Package discovery config. Mirrors query_planner DiscoverPackageConf."""

    package: str
    pattern: str = "*"

    def __post_init__(self):
        if not self.package:
            raise ValueError("Package name cannot be empty")


@dataclass
class ConnectorsConf:
    """Top-level connector configuration from marie.yml."""

    connectors: List[ConnectorConf] = None
    discover_packages: List[DiscoverPackageConf] = None
    discover_installed: bool = True

    def __post_init__(self):
        if self.connectors is None:
            self.connectors = []
        if self.discover_packages is None:
            self.discover_packages = []

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConnectorsConf":
        connectors = []
        for c in data.get("connectors", []) or []:
            connectors.append(ConnectorConf(name=c["name"], py_module=c["py_module"]))

        discover_packages = []
        for dp in data.get("discover_packages", []) or []:
            discover_packages.append(
                DiscoverPackageConf(
                    package=dp["package"], pattern=dp.get("pattern", "*")
                )
            )

        return cls(
            connectors=connectors,
            discover_packages=discover_packages,
            discover_installed=data.get("discover_installed", True),
        )
