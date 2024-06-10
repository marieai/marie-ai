"""gRPC service registry module."""

import abc
from typing import Union

from marie.serve.discovery import RepeatedTimer
from marie.serve.discovery.address import JsonAddress, PlainAddress
from marie.serve.discovery.etcd_client import EtcdClient

__all__ = ['EtcdServiceRegistry']


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
    """gRPC service registry based on etcd."""

    def __init__(
        self,
        etcd_host: str,
        etcd_port: int,
        etcd_client: object = None,
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
        self._client = etcd_client if etcd_client else EtcdClient(etcd_host, etcd_port)
        self._leases = {}
        self._services = {}
        self._heartbeat_time = heartbeat_time
        self.setup_heartbeat()

    def get_lease(self, service_addr, service_ttl):
        """Get a gRPC service lease from etcd.

        :param service_addr: gRPC service address.
        :param service_ttl: gRPC service lease ttl(seconds).
        :rtype `etcd3.lease.Lease`

        """
        lease = self._leases.get(service_addr)
        if lease and lease.remaining_ttl > 0:
            return lease

        lease_id = hash(service_addr)
        lease = self._client.lease(service_ttl, lease_id)
        self._leases[service_addr] = lease
        return lease

    def _form_service_key(self, service_name, service_addr):
        """Return service's key in etcd."""
        return '/'.join((service_name, service_addr))

    def register(
        self,
        service_names: Union[str, list[str]],
        service_addr: str,
        service_ttl: int,
        addr_cls=None,
        metadata=None,
    ) -> int:
        """Register gRPC services with the same address.

        :param service_names: A collection of service name.
        :param service_addr: server address.
        :param service_ttl: service ttl(seconds).
        :param addr_cls: format class of gRPC service address.
        :param metadata: extra meta data for JsonAddress.
        rtype `etcd3.lease.Lease`
        """
        if isinstance(service_names, str):
            service_names = [service_names]
        lease = self.get_lease(service_addr, service_ttl)
        addr_cls = addr_cls or PlainAddress
        for service_name in service_names:
            key = self._form_service_key(service_name, service_addr)
            if addr_cls == JsonAddress:
                addr_obj = addr_cls(service_addr, metadata=metadata)
            else:
                addr_obj = addr_cls(service_addr)

            addr_val = addr_obj.add_value()
            put_key, _ = self._client.put(key, addr_val, lease=lease)
            print("Registering service", service_name, service_addr, put_key)
            try:
                self._services[service_addr].add(service_name)
            except KeyError:
                self._services[service_addr] = {service_name}
        return lease

    def heartbeat(self, service_addr=None, service_ttl=5):
        """service heartbeat."""
        if service_addr:
            lease = self.get_lease(service_addr, service_ttl)
            leases = ((service_addr, lease),)
        else:
            leases = tuple(self._leases.items())

        for service_addr, lease in leases:
            print("Refreshing lease for", service_addr, lease.remaining_ttl)
            print(self._services[service_addr])
            ret = lease.refresh()[0]
            if ret.TTL == 0:
                self.register(self._services[service_addr], service_addr, lease.ttl)

    def unregister(self, service_names, service_addr, addr_cls=None):
        """Unregister gRPC services with the same address.

        :param service_names: A collection of gRPC service name.
        :param service_addr: gRPC server address.

        """
        addr_cls = addr_cls or PlainAddress
        etcd_delete = True
        if addr_cls != PlainAddress:
            etcd_delete = False

        for service_name in service_names:
            key = self._form_service_key(service_name, service_addr)
            if etcd_delete:
                self._client.delete(key)
            else:
                self._client.put(addr_cls(service_addr).delete_value())

            self._services.get(service_addr, {}).discard(service_name)

    def setup_heartbeat(self):
        print(
            "Setting up heartbeat for etcd service registry  : ", self._heartbeat_time
        )
        if self._heartbeat_time > 0:
            rt = RepeatedTimer(
                self._heartbeat_time,
                self.heartbeat,
            )


if __name__ == '__main__':
    etcd_registry = EtcdServiceRegistry('127.0.0.1', 2379, heartbeat_time=5)
    etcd_registry.register(['gateway/service_test'], '127.0.0.1:50011', 12)

    print(etcd_registry._services)
    print(etcd_registry._leases)
