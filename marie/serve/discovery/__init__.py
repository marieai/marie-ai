from typing import TYPE_CHECKING, Dict, Optional

from marie.enums import ProtocolType
from marie.serve.discovery.address import JsonAddress
from marie.serve.discovery.registry import EtcdServiceRegistry

if TYPE_CHECKING:  # pragma: no cover
    pass


class DiscoveryServiceMixin:
    """Instrumentation mixin for Service Discovery handling"""

    def _setup_service_discovery(
        self,
        protocol: ProtocolType,
        name: str,
        host: str,
        port: int,
        scheme: Optional[str] = "http",
        discovery: Optional[bool] = False,
        discovery_host: Optional[str] = "0.0.0.0",
        discovery_port: Optional[int] = 8500,
        discovery_scheme: Optional[str] = "http",
        discovery_watchdog_interval: Optional[int] = 60,
        discovery_service_name: str = "gateway/marie",
        runtime_args: Optional[Dict] = None,
    ) -> None:
        if self.logger is None:
            raise Exception("Expected logger to be configured")

        self._setup_service_discovery_etcd(
            name=name,
            host=host,
            port=port,
            scheme=scheme,
            discovery=discovery,
            discovery_host=discovery_host,
            discovery_port=discovery_port,
            discovery_scheme=discovery_scheme,
            discovery_watchdog_interval=discovery_watchdog_interval,
            discovery_service_name=discovery_service_name,
            runtime_args=runtime_args,
        )

    def _setup_service_discovery_etcd(
        self,
        name: str,
        host: str,
        port: int,
        scheme: Optional[str] = "http",
        discovery: Optional[bool] = False,
        discovery_host: Optional[str] = "0.0.0.0",
        discovery_port: Optional[int] = 8500,
        discovery_scheme: Optional[str] = "http",
        discovery_watchdog_interval: Optional[int] = 60,
        discovery_service_name: str = "gateway/marie",
        runtime_args: Optional[Dict] = None,
    ) -> None:
        if self.logger is None:
            raise Exception("Expected logger to be configured")
        if runtime_args is None:
            raise Exception("Expected runtime_args to be configured")

        self.logger.info("Setting up service discovery ETCD ...")
        self.discovery_host = discovery_host
        self.discovery_port = discovery_port
        self.discovery_scheme = discovery_scheme
        deployments_addresses = runtime_args.deployments_addresses
        scheme = "grpc"
        ctrl_address = f"{scheme}://{host}:{port}"
        ctrl_address = f"{host}:{port}"
        self.logger.info(f"Deployments addresses: {deployments_addresses}")

        # TODO - this should be configurable
        service_ttl = 6
        heartbeat_time = 2

        etcd_registry = EtcdServiceRegistry(
            self.discovery_host,
            self.discovery_port,
            heartbeat_time=heartbeat_time,
        )
        lease = etcd_registry.register(
            [discovery_service_name],
            ctrl_address,
            service_ttl=service_ttl,
            addr_cls=JsonAddress,
            metadata=deployments_addresses,
        )

        self.sd_state = "started"
        self.logger.info(f"Lease ID: {lease.id}")

    def _teardown_service_discovery(
        self,
    ) -> None:
        """Teardown service discovery, by unregistering existing service from the catalog"""
        if self.sd_state != "ready":
            return
        self.sd_state = "stopping"
        try:
            # TODO - Implement service discovery teardown
            pass
        except Exception:
            pass
