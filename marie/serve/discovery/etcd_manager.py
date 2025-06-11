from marie.serve.discovery.container import EtcdServiceContainer
from marie.serve.discovery.etcd_client import EtcdClient


def get_etcd_client(args: dict = None) -> EtcdClient:
    """Get the shared EtcdClient instance."""
    return EtcdServiceContainer.get_client(args)
