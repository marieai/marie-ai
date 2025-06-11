import warnings
from typing import Any, Dict

from marie.serve.discovery.container import EtcdServiceContainer
from marie.serve.discovery.etcd_client import EtcdClient


def convert_to_etcd_args(_args: Dict) -> Dict[str, Any]:
    discovery_host = _args.get('discovery_host')
    discovery_port = _args.get('discovery_port', 2379)
    if not discovery_host:
        warnings.warn(
            "The `discovery_host` is not defined. Defaulting to `127.0.0.1`. Please ensure this is intentional.",
            UserWarning,
        )
        discovery_host = '127.0.0.1'  # Default value

    etcd_args = {
        'host': discovery_host,
        'port': discovery_port,
        'endpoints': _args.get(
            'discovery_endpoints', [(discovery_host, discovery_port)]
        ),
        'namespace': _args.get('discovery_namespace', 'marie'),
        'timeout': _args.get('discovery_timeout_sec', 10),
        'retry_times': _args.get('discovery_retry_times', 5),
        'lease_sec': _args.get('discovery_lease_sec', 6),
        'heartbeat_sec': _args.get('discovery_heartbeat_sec', 1.5),
        'ca_cert': _args.get('discovery_ca_cert', None),
        'cert_key': _args.get('discovery_cert_key', None),
        'cert_cert': _args.get('discovery_cert_cert', None),
        'grpc_options': _args.get('discovery_grpc_options', None),
    }
    return etcd_args


def get_etcd_client(args: dict = None) -> EtcdClient:
    """Get the shared EtcdClient instance."""
    return EtcdServiceContainer.get_client(args)
