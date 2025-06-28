import abc
import logging
import threading
import time
from typing import Union

import etcd3

from marie.helper import get_or_reuse_loop
from marie.serve.discovery.address import JsonAddress, PlainAddress
from marie.serve.discovery.container import EtcdConfig
from marie.serve.discovery.etcd_client import EtcdClient
from marie.serve.discovery.etcd_manager import convert_to_etcd_args, get_etcd_client
from marie.serve.discovery.util import form_service_key
from marie.utils.timer import RepeatedTimer

__all__ = ["EtcdServiceRegistry"]

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)


class ServiceRegistry(abc.ABC):
    """A service registry."""

    @abc.abstractmethod
    def register(self, service_names, service_addr, service_ttl):
        """Register services with the same address."""
        raise NotImplementedError

    @abc.abstractmethod
    def heartbeat(self, service_addr=None):
        """Service registry heartbeat."""
        raise NotImplementedError

    @abc.abstractmethod
    def unregister(self, service_names, service_addr):
        """Unregister services with the same address."""
        raise NotImplementedError


class EtcdServiceRegistry(ServiceRegistry):
    """service registry based on etcd."""

    def __init__(
        self,
        etcd_host: str,
        etcd_port: int,
        etcd_client: EtcdClient = None,
        heartbeat_time=-1,
    ):
        """Initialize etcd service registry.

        :param heartbeat_time:
        :param etcd_host: (optional) etcd node host for :class:`client.EtcdClient`.
        :param etcd_port: (optional) etcd node port for :class:`client.EtcdClient`.
        :param etcd_client: (optional) A :class:`client.EtcdClient` object.
        :param heartbeat_time: (optional) service registry heartbeat time interval, default -1. If -1, no heartbeat.

        """
        if etcd_host is None and etcd_client is None:
            raise ValueError("etcd_host or etcd_client must be provided.")

        if etcd_client and (etcd_host or etcd_port):
            log.warning(
                "Both etcd_client and etcd_host/etcd_port are provided. Using etcd_client."
            )
        if etcd_client:
            if not isinstance(etcd_client, EtcdClient):
                raise TypeError("etcd_client must be an instance of EtcdClient.")
            self._client = etcd_client
        else:
            args_dict = {
                "discovery_host": etcd_host,
                "discovery_port": etcd_port,
            }
            etcd_args = convert_to_etcd_args(args_dict)
            etcd_config = EtcdConfig.from_dict(etcd_args)
            self._client = get_etcd_client(etcd_args)

        self._leases = {}
        self._services = {}
        self._loop = get_or_reuse_loop()
        self._heartbeat_time = heartbeat_time
        self.setup_heartbeat_async()
        # self.setup_heartbeat()

    def get_lease(self, service_addr, service_ttl):
        """Get a service lease from etcd.

        :param service_addr: service address.
        :param service_ttl: service lease ttl(seconds).
        :rtype `etcd3.lease.Lease`

        """
        lease = self._leases.get(service_addr)
        if lease and lease.remaining_ttl > 0:
            return lease

        lease_id = hash(service_addr)
        lease = self._client.lease(service_ttl, lease_id)
        self._leases[service_addr] = lease
        return lease

    def register(
        self,
        service_names: Union[str, list[str]],
        service_addr: str,
        service_ttl: int,
        addr_cls=None,
        metadata=None,
    ) -> int:
        """Register services with the same address.

        :param service_names: A collection of service name.
        :param service_addr: server address.
        :param service_ttl: service ttl(seconds).
        :param addr_cls: format class of service address.
        :param metadata: extra meta data for JsonAddress.
        rtype `etcd3.lease.Lease`
        """
        if isinstance(service_names, str):
            service_names = [service_names]
        lease = self.get_lease(service_addr, service_ttl)
        addr_cls = addr_cls or PlainAddress

        for service_name in service_names:
            key = form_service_key(service_name, service_addr)
            resolved = self._client.get(key)

            if resolved:
                log.warning(
                    f"Service already registered : {service_name}@{service_addr}"
                )
                continue

            if addr_cls == JsonAddress:
                addr_obj = addr_cls(service_addr, metadata=metadata)
            else:
                addr_obj = addr_cls(service_addr)

            addr_val = addr_obj.add_value()
            put_key, _ = self._client.put(key, addr_val, lease=lease)
            log.warning(
                f"Registering service : {service_name}@{service_addr} : {put_key}"
            )
            try:
                self._services[service_addr].add(service_name)
            except KeyError:
                self._services[service_addr] = {service_name}
        return lease

    def heartbeat(self, service_addr=None, service_ttl=5):
        """service heartbeat."""
        log.info(f"Heartbeat service_addr : {service_addr}")
        if service_addr:
            lease = self.get_lease(service_addr, service_ttl)
            leases = ((service_addr, lease),)
        else:
            leases = tuple(self._leases.items())

        for service_addr, lease in leases:
            registered = self._services.get(service_addr, None)
            if not registered:
                continue
            try:
                log.debug(
                    f"Refreshing lease for: {service_addr}, {lease.remaining_ttl}"
                )
                ret = lease.refresh()[0]
                if ret.TTL == 0:
                    self.register(
                        self._services.get(service_addr, []),
                        service_addr,
                        lease.ttl,
                    )
            except (ValueError, etcd3.exceptions.ConnectionFailedError) as e:
                if (
                    isinstance(e, etcd3.exceptions.ConnectionFailedError)
                    or str(e) == "Trying to use a failed node"
                ):
                    log.warning(
                        f"Trying to use a failed node, attempting to reconnect."
                    )
                    if self._client.reconnect():
                        log.info("Reconnected to etcd")
                        lease.etcd_client = self._client.client
            except Exception as e:
                raise e

    def unregister(self, service_names, service_addr, addr_cls=None):
        """Unregister services with the same address.

        :param service_names: A collection of service name.
        :param service_addr: server address.
        :param addr_cls: format class of service address.
        """

        addr_cls = addr_cls or PlainAddress
        etcd_delete = True
        if addr_cls != PlainAddress:
            etcd_delete = False

        registered_services = self._services.get(service_addr, {})
        for service_name in service_names:
            log.info(f"Unregistering service : {service_name}@{service_addr}")
            key = form_service_key(service_name, service_addr)
            if etcd_delete:
                self._client.delete(key)
            else:
                self._client.put(addr_cls(service_addr).delete_value())
            registered_services.discard(service_name)

    def setup_heartbeat_async(self):
        """
        Set up an asynchronous heartbeat process.

        :return: None
        """

        def _heartbeat_setup():
            log.info(
                f"Setting up heartbeat for etcd service registry  : {self._heartbeat_time}"
            )
            time.sleep(self._heartbeat_time)
            while True:
                try:
                    self.heartbeat()
                except Exception as e:
                    log.error(f"Error in heartbeat : {str(e)}")
                time.sleep(self._heartbeat_time)

        polling_status_thread = threading.Thread(
            target=_heartbeat_setup,
            daemon=True,
        )
        polling_status_thread.start()

    def setup_heartbeat(self):
        """
        This method is used to set up a heartbeat for the etcd service registry.

        :return: None
        """
        log.info(
            f"Setting up heartbeat for etcd service registry  : {self._heartbeat_time}",
        )
        if self._heartbeat_time > 0:
            rt = RepeatedTimer(
                self._heartbeat_time,
                self.heartbeat,
            )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="service discovery etcd cluster")
    parser.add_argument(
        "--host",
        help="the etcd host, default = 127.0.0.1",
        required=False,
        default="127.0.0.1XXX",
    )
    parser.add_argument(
        "--port",
        help="the etcd port, default = 2379",
        required=False,
        default=2379,
        type=int,
    )
    parser.add_argument("--ca-cert", help="the etcd ca-cert", required=False)
    parser.add_argument("--cert-key", help="the etcd cert key", required=False)
    parser.add_argument("--cert-cert", help="the etcd cert", required=False)
    parser.add_argument("--service-key", help="the service key", required=True)
    parser.add_argument(
        "--service-addr", help="the service address host:port ", required=True
    )
    parser.add_argument(
        "--lease-ttl",
        help="the lease ttl in seconds, default is 10",
        required=False,
        default=10,
        type=int,
    )
    parser.add_argument("--my-id", help="my identifier", required=True)
    parser.add_argument(
        "--timeout",
        help="the etcd operation timeout in seconds, default is 2",
        required=False,
        type=int,
        default=2,
    )
    args = parser.parse_args()

    params = {"host": args.host, "port": args.port, "timeout": args.timeout}
    if args.ca_cert:
        params["ca_cert"] = args.ca_cert
    if args.cert_key:
        params["cert_key"] = args.cert_key
    if args.cert_cert:
        params["cert_cert"] = args.cert_cert

    log.info(f"args : {args}")

    etcd_registry = EtcdServiceRegistry(args.host, args.port, heartbeat_time=5)
    etcd_registry.register([args.service_key], args.service_addr, args.lease_ttl)

    try:
        while True:
            time.sleep(2)  # Keep the program running.
    except KeyboardInterrupt:
        etcd_registry.unregister([args.service_key], args.service_addr)
        print("Service unregistered.")


if __name__ == "__main__":
    main()
