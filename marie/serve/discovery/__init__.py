import json
from typing import TYPE_CHECKING, Dict, Optional

from marie.enums import ProtocolType
from marie.helper import get_internal_ip
from marie.serve.discovery.address import JsonAddress
from marie.serve.discovery.registry import EtcdServiceRegistry

if TYPE_CHECKING:  # pragma: no cover
    pass

DEFAULT_DISCOVERY_LEASE_SEC = 16
DEFAULT_DISCOVERY_HEARTBEAT_SEC = 3.0


def _discovery_lease_params(runtime_args) -> tuple[int, float]:
    """Resolve discovery lease TTL and heartbeat interval from runtime args."""
    ttl = getattr(runtime_args, "discovery_lease_sec", None)
    beat = getattr(runtime_args, "discovery_heartbeat_sec", None)
    return (
        int(ttl) if ttl else DEFAULT_DISCOVERY_LEASE_SEC,
        float(beat) if beat else DEFAULT_DISCOVERY_HEARTBEAT_SEC,
    )


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
        deployments_addresses = json.loads(runtime_args.deployments_addresses)
        scheme = "grpc"
        ctrl_address = f"{scheme}://{host}:{port}"
        ctrl_address = f"{host}:{port}"
        self.logger.info(f"Deployments addresses: {deployments_addresses}")
        self.logger.info(f"Deployments ctrl_address: {ctrl_address}")
        self.logger.info(f"Deployments runtime_args: {runtime_args}")

        service_ttl, heartbeat_time = _discovery_lease_params(runtime_args)

        self.sd_state = "starting"
        self._etcd_registry = EtcdServiceRegistry(
            self.discovery_host,
            self.discovery_port,
            heartbeat_time=heartbeat_time,
        )
        etcd_registry = self._etcd_registry

        # we are unrolling the deployments_addresses to register each deployment separately
        # this is to allow for FLOW deployments to be registered separately without the gateway

        self.logger.info(f"Registering service : {name}")
        for deployment_name, deployment_addresses in deployments_addresses.items():
            for deployment_address in deployment_addresses:
                # TODO: we need to handle both internal and public IPs or have a way to distinguish between them
                # TODO: Long term solution is to have a way to register both internal and public IPs at this same time
                # When registering the deployment with ETCD, we need to ensure that IP can access from another machine
                # get_internal_ip()  or   get_public_ip()

                if "://" in deployment_address:
                    scheme, address = deployment_address.split("://")
                    ip, port = address.split(":")
                    private_ip = get_internal_ip()
                    deployment_address = f"{scheme}://{private_ip}:{port}"
                else:
                    ip, port = deployment_address.split(":")
                    private_ip = get_internal_ip()
                    deployment_address = f"{private_ip}:{port}"

                single_deployments_addresses = {
                    deployment_name: [deployment_address]
                }  # we keeping the original format
                single_ctrl_address = deployment_address
                if "://" in deployment_address:
                    single_ctrl_address = deployment_address.split("://")[1]

                # FIXME - this is a workaround for the fact that we are not able to register the same service with different addresses
                # This needs to be reworked in the future to be able to handle Deployment without the gateway
                # single_ctrl_address = ctrl_address

                self.logger.info(
                    f"Registering deployment {deployment_name} with address {single_ctrl_address}"
                )
                lease = etcd_registry.register(
                    [discovery_service_name],
                    single_ctrl_address,
                    service_ttl=service_ttl,
                    addr_cls=JsonAddress,
                    metadata=json.dumps(single_deployments_addresses),
                )
                self.logger.info(f"Lease ID: {lease.id}")

        self.sd_state = "ready"

    def _teardown_service_discovery(
        self,
    ) -> None:
        """Stop service discovery before deployments begin shutting down."""
        registry = self._etcd_registry
        if registry is None:
            return
        self.sd_state = "stopping"
        try:
            registry.shutdown()
        except Exception as exc:
            self.logger.warning(f"Failed to stop service discovery: {exc}")
        finally:
            self._etcd_registry = None
            self.sd_state = "stopped"
