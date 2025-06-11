import os
import threading
from typing import Any, Dict, List, Optional, Tuple, Union

from marie.serve.discovery.etcd_client import EtcdClient


class EtcdConfig:
    """Configuration for EtcdClient instances."""

    @staticmethod
    def get_endpoints() -> List[Union[str, Tuple[str, int]]]:
        """Get etcd endpoints from environment or args."""
        endpoints_str = os.getenv('ETCD_ENDPOINTS')
        if endpoints_str:
            return endpoints_str.split(',')

        # Fallback to individual host/port
        host = os.getenv('ETCD_HOST', 'localhost')
        port = int(os.getenv('ETCD_PORT', '2379'))
        return [(host, port)]

    @staticmethod
    def get_namespace() -> str:
        return os.getenv('ETCD_NAMESPACE', 'marie')

    @staticmethod
    def get_timeout() -> float:
        return float(os.getenv('ETCD_TIMEOUT', '5.0'))

    @staticmethod
    def get_retry_times() -> int:
        return int(os.getenv('ETCD_RETRY_TIMES', '10'))

    @staticmethod
    def get_ca_cert() -> Optional[str]:
        """Get CA certificate file path from environment."""
        return os.getenv('ETCD_CA_CERT')

    @staticmethod
    def get_cert_key() -> Optional[str]:
        """Get client certificate key file path from environment."""
        return os.getenv('ETCD_CERT_KEY')

    @staticmethod
    def get_cert_cert() -> Optional[str]:
        """Get client certificate file path from environment."""
        return os.getenv('ETCD_CERT_CERT')

    @staticmethod
    def get_credentials() -> Optional[Dict[str, str]]:
        """Get etcd credentials from environment."""
        user = os.getenv('ETCD_USER')
        password = os.getenv('ETCD_PASSWORD')
        if user and password:
            return {'user': user, 'password': password}
        return None

    @staticmethod
    def get_encoding() -> str:
        """Get encoding from environment."""
        return os.getenv('ETCD_ENCODING', 'utf8')

    @staticmethod
    def get_grpc_options() -> Optional[List[Tuple[str, Any]]]:
        """Get gRPC options from environment."""
        # Example: ETCD_GRPC_OPTIONS="grpc.keepalive_time_ms:30000,grpc.keepalive_timeout_ms:5000"
        options_str = os.getenv('ETCD_GRPC_OPTIONS')
        if not options_str:
            return None

        options = []
        for option_pair in options_str.split(','):
            if ':' in option_pair:
                key, value = option_pair.split(':', 1)
                # Try to convert to int/float if possible, otherwise keep as string
                try:
                    if '.' in value:
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    pass  # Keep as string
                options.append((key.strip(), value))

        return options if options else None


class EtcdServiceContainer:
    """Simple dependency injection container for EtcdClient instances."""

    _client = None
    _args = None
    _lock = threading.Lock()

    @classmethod
    def get_client(cls, args: dict = None) -> EtcdClient:
        """Get or create the shared EtcdClient instance."""
        if cls._client is None:
            with cls._lock:
                if cls._client is None:
                    cls._args = args
                    cls._client = cls._create_etcd_client()
        return cls._client

    @classmethod
    def _create_etcd_client(cls) -> EtcdClient:
        """Create EtcdClient with configuration priority: args > env > defaults."""
        # Priority: args > environment > defaults
        if cls._args:
            # Use args if available (backward compatibility)
            if 'discovery_host' in cls._args and 'discovery_port' in cls._args:
                return EtcdClient(
                    etcd_host=cls._args['discovery_host'],
                    etcd_port=cls._args['discovery_port'],
                    namespace=cls._args.get('namespace', 'marie'),
                    timeout=cls._args.get('timeout', 5.0),
                    retry_times=cls._args.get('retry_times', 10),
                    ca_cert=cls._args.get('ca_cert'),
                    cert_key=cls._args.get('cert_key'),
                    cert_cert=cls._args.get('cert_cert'),
                    grpc_options=cls._args.get('grpc_options'),
                )
            elif 'etcd_endpoints' in cls._args:
                return EtcdClient(
                    endpoints=cls._args['etcd_endpoints'],
                    namespace=cls._args.get('namespace', 'marie'),
                    timeout=cls._args.get('timeout', 5.0),
                    retry_times=cls._args.get('retry_times', 10),
                    ca_cert=cls._args.get('ca_cert'),
                    cert_key=cls._args.get('cert_key'),
                    cert_cert=cls._args.get('cert_cert'),
                    grpc_options=cls._args.get('grpc_options'),
                )

        # Use environment variables or defaults
        return EtcdClient(
            endpoints=EtcdConfig.get_endpoints(),
            namespace=EtcdConfig.get_namespace(),
            timeout=EtcdConfig.get_timeout(),
            retry_times=EtcdConfig.get_retry_times(),
            ca_cert=EtcdConfig.get_ca_cert(),
            cert_key=EtcdConfig.get_cert_key(),
            cert_cert=EtcdConfig.get_cert_cert(),
            grpc_options=EtcdConfig.get_grpc_options(),
        )

    @classmethod
    def reset(cls):
        """Reset the container (useful for testing)."""
        with cls._lock:
            cls._client = None
            cls._args = None


# Global container instance
etcd_container = EtcdServiceContainer()

# SHELL VARIABLES

# export ETCD_CA_CERT="/path/to/ca.crt"
# export ETCD_CERT_KEY="/path/to/client.key"
# export ETCD_CERT_CERT="/path/to/client.crt"
# export ETCD_USER="myuser"
# export ETCD_PASSWORD="mypassword"
# export ETCD_GRPC_OPTIONS="grpc.keepalive_time_ms:30000,grpc.keepalive_timeout_ms:5000"


# Kubernetes ConfigMap example:
# apiVersion: v1
# kind: ConfigMap
# metadata:
#   name: etcd-config
#   namespace: marie
# data:
#   ETCD_ENDPOINTS: "etcd-0.etcd-headless.marie.svc.cluster.local:2379,etcd-1.etcd-headless.marie.svc.cluster.local:2379,etcd-2.etcd-headless.marie.svc.cluster.local:2379"
#   ETCD_NAMESPACE: "marie"
#   ETCD_TIMEOUT: "10.0"
#   ETCD_RETRY_TIMES: "5"
#   ETCD_ENCODING: "utf8"
#   ETCD_GRPC_OPTIONS: "grpc.keepalive_time_ms:30000,grpc.keepalive_timeout_ms:5000"
