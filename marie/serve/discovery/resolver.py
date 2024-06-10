import abc
import threading
import time

from marie.serve.discovery.address import PlainAddress
from marie.serve.discovery.etcd_client import EtcdClient

__all__ = ['EtcdServiceResolver']


class ServiceResolver(abc.ABC):
    """gRPC service Resolver class."""

    @abc.abstractmethod
    def resolve(self, name):
        raise NotADirectoryError

    @abc.abstractmethod
    def update(self, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def listen(self):
        raise NotImplementedError


class EtcdServiceResolver(ServiceResolver):
    """gRPC service resolver based on Etcd."""

    def __init__(
        self,
        etcd_host=None,
        etcd_port=None,
        etcd_client=None,
        start_listener=True,
        listen_timeout=5,
        addr_cls=None,
        namespace='marie',
    ):
        """Initialize etcd service resolver.

        :param etcd_host: (optional) etcd node host for :class:`client.EtcdClient`.
        :param etcd_port: (optional) etcd node port for :class:`client.EtcdClient`.
        :param etcd_client: (optional) A :class:`client.EtcdClient` object.
        :param start_listener: (optional) Indicate whether starting the resolver listen thread.
        :param listen_timeout: (optional) Resolver thread listen timeout.
        :param addr_cls: (optional) address format class.
        :param namespace: (optional) Etcd namespace.
        """
        super().__init__()

        if etcd_host is None and etcd_client is None:
            raise ValueError("etcd_host or etcd_client must be provided.")
        self._listening = False
        self._stopped = False
        self._listen_thread = None
        self._listen_timeout = listen_timeout
        self._lock = threading.Lock()
        self._client = (
            etcd_client
            if etcd_client
            else EtcdClient(etcd_host, etcd_port, namespace=namespace)
        )
        self._names = {}
        self._addr_cls = addr_cls or PlainAddress

        if start_listener:
            self.start_listener()

    def resolve(self, name: str) -> list:
        """Resolve gRPC service name.

        :param name: gRPC service name.
        :rtype list: A collection gRPC server address.

        """
        with self._lock:
            try:
                return self._names[name]
            except KeyError:
                addrs = self.get(name)
                self._names[name] = addrs
                return addrs

    def get(self, name: str):
        """Get values from Etcd.

        :param name: Etcd key prefix name.
        :rtype list: A collection of Etcd values.

        """
        keys = self._client.get_prefix(name)
        vals = []
        plain = True
        if self._addr_cls != PlainAddress:
            plain = False

        for val, metadata in keys:
            if plain:
                vals.append(self._addr_cls.from_value(val))
            else:
                add, addr = self._addr_cls.from_value(val)
                if add:
                    vals.append(addr)

        return vals

    def update(self, **kwargs):
        """Add or delete service address.

        :param kwargs: Dictionary of ``'service_name': ((add-address, delete-address)).``

        """
        with self._lock:
            for name, (add, delete) in kwargs.items():
                try:
                    self._names[name].extend(add)
                except KeyError:
                    self._names[name] = add

                for del_item in delete:
                    try:
                        self._names[name].remove(del_item)
                    except ValueError:
                        continue

    def listen(self):
        """Listen for change about service address."""
        while not self._stopped:
            for name in self._names:
                try:
                    vals = self.get(name)
                except:
                    continue
                else:
                    with self._lock:
                        self._names[name] = vals

            time.sleep(self._listen_timeout)

    def watch_service(self, service_name: str, event_callback: callable):
        watch_id = self._client.add_watch_prefix_callback(service_name, event_callback)
        print("Watching service:", service_name, watch_id)

    def start_listener(self, daemon=True):
        """Start listen thread.

        :param daemon: Indicate whether start thread as a daemon.

        """
        if self._listening:
            return

        thread_name = 'Thread-resolver-listener'
        self._listen_thread = threading.Thread(target=self.listen, name=thread_name)
        self._listen_thread.daemon = daemon
        self._listen_thread.start()
        self._listening = True

    def stop(self):
        """Stop service resolver."""
        if self._stopped:
            return

        self._stopped = True

    def __del__(self):
        self.stop()


if __name__ == '__main__':
    resolver = EtcdServiceResolver(
        '127.0.0.1', 2379, namespace='marie', start_listener=False, listen_timeout=5
    )
    print(resolver.resolve('gateway/service_test'))

    resolver.watch_service(
        'gateway/service_test',
        lambda service, event: print("Event from service : ", service, event),
    )

    while True:
        print("Checking service address...")
        # print(resolver.resolve('service_test'))
        time.sleep(2)
