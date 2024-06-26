import threading
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union

import requests

from marie._annotations import deprecated
from marie.enums import ProtocolType
from marie.helper import get_internal_ip
from marie.importer import ImportExtensions
from marie.serve.discovery.address import JsonAddress
from marie.serve.discovery.registry import EtcdServiceRegistry
from marie.utils.timer import RepeatedTimer

if TYPE_CHECKING:  # pragma: no cover
    import consul


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
        runtime_args: Optional[Dict] = None,
    ) -> None:
        if self.logger is None:
            raise Exception("Expected logger to be configured")

        if protocol == ProtocolType.GRPC:
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
                runtime_args=runtime_args,
            )
        elif protocol == ProtocolType.HTTP:  # DEPRECATED : HTTP is deprecated
            self._setup_service_discovery_consul(
                name=name,
                host=host,
                port=port,
                scheme=scheme,
                discovery=discovery,
                discovery_host=discovery_host,
                discovery_port=discovery_port,
                discovery_scheme=discovery_scheme,
                discovery_watchdog_interval=discovery_watchdog_interval,
            )
        else:
            raise NotImplementedError(f"Protocol {protocol} is not supported")

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
        runtime_args: Optional[Dict] = None,
    ) -> None:
        if self.logger is None:
            raise Exception("Expected logger to be configured")
        if runtime_args is None:
            raise Exception("Expected runtime_args to be configured")

        self.logger.info("Setting up service discovery ETCD ...")
        self.sd_state = "started"
        self.discovery_host = discovery_host
        self.discovery_port = discovery_port
        self.discovery_scheme = discovery_scheme
        deployments_addresses = runtime_args.deployments_addresses
        scheme = "grpc"
        ctrl_address = f"{scheme}://{host}:{port}"
        ctrl_address = f"{host}:{port}"
        service_name = "gateway/service_test"

        self.logger.info(f"Deployments addresses: {deployments_addresses}")

        etcd_registry = EtcdServiceRegistry(
            "0.0.0.0",
            2379,
            heartbeat_time=5,
        )
        lease = etcd_registry.register(
            [service_name],
            ctrl_address,
            6,
            addr_cls=JsonAddress,
            metadata=deployments_addresses,
        )

        self.logger.info(f"Lease ID: {lease.id}")

    @deprecated
    def _setup_service_discovery_consul(
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
    ) -> None:

        # testing
        if True:
            return

        if discovery:
            with ImportExtensions(
                required=True,
                help_text="You need to install the `python-consul` to use the service discovery functionality of marie",
            ):
                import consul

                # Ban advertising 0.0.0.0 or setting it as a service address #2961
                if host == "0.0.0.0":
                    host = get_internal_ip()

                def _watchdog_target():
                    return self._start_discovery_watchdog(
                        name=name,
                        service_host=host,
                        service_port=port,
                        service_scheme=scheme,
                        discovery_host=discovery_host,
                        discovery_port=discovery_port,
                        discovery_scheme=discovery_scheme,
                        discovery_watchdog_interval=discovery_watchdog_interval,
                    )

                t = threading.Thread(target=_watchdog_target, daemon=True)
                t.start()

    def _is_discovery_online(self, client: Union["consul.Consul", None]) -> bool:
        """Check if service discovery is online"""
        if client is None:
            return False
        try:
            client.agent.self()
            return True
        except (requests.exceptions.ConnectionError, ConnectionError) as e:
            pass
        except Exception as e:
            self.logger.warning("Unable to verify connection : {msg}".format(msg=e))
        return False

    def _teardown_service_discovery(
        self,
    ) -> None:
        """Teardown service discovery, by unregistering existing service from the catalog"""
        if self.sd_state != "ready":
            return
        self.sd_state = "stopping"
        try:
            self.discovery_client.agent.service.deregister(self.service_id)
        except Exception:
            pass

    def _start_discovery_watchdog(
        self,
        name,
        service_host,
        service_port,
        service_scheme,
        discovery_host,
        discovery_port,
        discovery_scheme,
        discovery_watchdog_interval,
    ):
        # TODO : Service ID generation needs to be configurable
        # Create new service id, otherwise we will re-register same id
        self.service_id = f"{name}@{service_host}:{service_port}"
        self.service_name = "traefik-system-ingress"
        self.sd_state = "ready"
        self.discovery_client, online = self._create_discovery_client(True)

        def __register(_service_host, _service_port, _service_scheme):

            # Calling /dry_run on the flow will check if the Flow is initialized fully
            # Expecting to get InternalNetworkError: failed to connect to all addresses

            if self._is_discovery_online(client=self.discovery_client):
                service_node = self._get_service_node(
                    self.service_name, self.service_id
                )
                if service_node is None:
                    self.service_id = self._register_with_catalog(
                        service_host=_service_host,
                        service_port=_service_port,
                        service_scheme=_service_scheme,
                        service_id=self.service_id,
                    )
                    self.logger.debug("Re-registered service: %s", self.service_id)

        __register(service_host, service_port, service_scheme)

        rt = RepeatedTimer(
            discovery_watchdog_interval,
            __register,
            service_host,
            service_port,
            service_scheme,
        )

    def _verify_discovery_connection(
        self,
        discovery_host: str,
        discovery_port: int = 8500,
        discovery_scheme: Optional[str] = "http",
    ) -> bool:
        """Verify consul connection
        Exceptions throw such as ConnectionError will be captured
        """
        import consul

        self.logger.debug(
            "Verifying Consul connection to %s://%s:%s",
            discovery_scheme,
            discovery_host,
            discovery_port,
        )

        try:
            client = consul.Consul(
                host=discovery_host, port=discovery_port, scheme=discovery_scheme
            )
            client.agent.self()
            return True
        except (requests.exceptions.ConnectionError, ConnectionError) as e:
            pass
        except Exception as e:
            self.logger.warning("Unable to verify connection : {msg}".format(msg=e))

        return False

    def _create_discovery_client(
        self,
        verify: bool = True,
    ) -> Tuple[Union["consul.Consul", None], bool]:
        """Create new consul client"""
        import consul

        try:
            self.logger.debug(
                "Consul Host: %s Port: %s ", self.discovery_host, self.discovery_port
            )
            client = consul.Consul(host=self.discovery_host, port=self.discovery_port)
            online = False
            if verify:
                try:
                    client.agent.self()
                    online = True
                except Exception:
                    pass
                self.logger.debug("Consul online status : %s", online)
            return client, online
        except Exception as ex:
            raise ex
            # pass

    def _get_service_node(self, service_name, service_id):
        try:
            index, nodes = self.discovery_client.catalog.service(service_name)
            if nodes is None:
                return None
            for node in nodes:
                if node["ServiceID"] == service_id:
                    return node
        except Exception as e:
            raise e

    def _register_with_catalog(
        self, service_host, service_port, service_scheme, service_id
    ) -> Union[None, str]:
        """
        Register new service in consul
        """
        from consul.base import Check

        self.logger.debug(
            "Registering ServiceHost: %s Port: %s ", service_host, service_port
        )

        if service_id is None:
            raise Exception("service_id was none")

        service_name = "traefik-system-ingress"
        # dry_run will block until the service is ready as well as it will block until the request is processed Executor
        # which causes the service to be de-registered in the consul catalog
        # service_url = f"http://{service_host}:{service_port}/health/status"

        service_url = f"{service_scheme}://{service_host}:{service_port}/health/status"

        if not self._is_discovery_online(self.discovery_client):
            self.logger.debug("Consul service is offline")
            return service_id

        try:
            self.discovery_client.agent.service.register(
                name=service_name,
                service_id=service_id,
                port=service_port,
                address=service_host,
                check=Check.http(service_url, "10s"),
                tags=[
                    "traefik.enable=true",
                    "traefik.consulcatalog.connect=false",
                    "traefik.http.routers.traefik-system-ingress.entrypoints=marie",
                    "traefik.http.routers.traefik-system-ingress.service=traefik-system-ingress",
                    "traefik.http.routers.traefik-system-ingress.rule=HostRegexp(`{host:.+}`)",
                    "traefik.http.services.traefik-system-ingress.loadbalancer.server.scheme=http",
                ],
            )
        except Exception as e:
            raise e
        return service_id
