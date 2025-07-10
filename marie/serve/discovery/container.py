import os
import threading
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from marie.serve.discovery.etcd_client import EtcdClient


@dataclass
class EtcdConfig:
    """Configuration for EtcdClient instances."""

    host: str = field(default_factory=lambda: os.getenv('ETCD_HOST', 'localhost'))
    port: int = field(default_factory=lambda: int(os.getenv('ETCD_PORT', 2379)))
    endpoints: List[Union[str, Tuple[str, int]]] = field(
        default_factory=lambda: EtcdConfig._get_endpoints_from_env()
    )
    namespace: str = field(default_factory=lambda: os.getenv('ETCD_NAMESPACE', 'marie'))
    timeout_sec: float = field(
        default_factory=lambda: float(os.getenv('ETCD_TIMEOUT_SEC', '30.0'))
    )
    retry_times: int = field(
        default_factory=lambda: int(os.getenv('ETCD_RETRY_TIMES', '10'))
    )
    ca_cert: Optional[str] = field(default_factory=lambda: os.getenv('ETCD_CA_CERT'))
    cert_key: Optional[str] = field(default_factory=lambda: os.getenv('ETCD_CERT_KEY'))
    cert_cert: Optional[str] = field(
        default_factory=lambda: os.getenv('ETCD_CERT_CERT')
    )
    grpc_options: Optional[List[Tuple[str, Any]]] = field(
        default_factory=lambda: EtcdConfig._get_grpc_options_from_env()
    )

    # Service discovery-related fields
    lease_sec: int = field(
        default_factory=lambda: int(os.getenv('ETCD_LEASE_SEC', '6'))
    )
    heartbeat_sec: float = field(
        default_factory=lambda: float(os.getenv('ETCD_HEARTBEAT_SEC', '1.5'))
    )
    service_name: str = field(
        default_factory=lambda: os.getenv('ETCD_SERVICE_NAME', 'gateway/marie')
    )  # Updated name

    _used_env: bool = field(
        init=False, default=False
    )  # Tracks if environment variables were used

    def __post_init__(self):
        """Warn if environment variables were used during initialization."""
        if self._check_if_env_used():
            self._used_env = True
            warnings.warn(
                "Configuration values were provided via environment variables. Ensure this is intended.",
                UserWarning,
            )

    def _check_if_env_used(self) -> bool:
        """Check if any field used environment variables."""
        env_keys = {
            'ETCD_HOST': self.host,
            'ETCD_PORT': self.port,
            'ETCD_NAMESPACE': self.namespace,
            'ETCD_TIMEOUT_SEC': self.timeout_sec,
            'ETCD_RETRY_TIMES': self.retry_times,
            'ETCD_CA_CERT': self.ca_cert,
            'ETCD_CERT_KEY': self.cert_key,
            'ETCD_CERT_CERT': self.cert_cert,
            'ETCD_ENDPOINTS': self.endpoints,
            'ETCD_GRPC_OPTIONS': self.grpc_options,
            'ETCD_LEASE_SEC': self.lease_sec,
            'ETCD_HEARTBEAT_SEC': self.heartbeat_sec,
            'ETCD_SERVICE_NAME': self.service_name,  # Updated to match normalized field
        }
        for key, value in env_keys.items():
            env_value = os.getenv(key)
            if env_value is not None and env_value == str(value):
                return True
        return False

    @staticmethod
    def _get_endpoints_from_env() -> list[str] | None:
        """Get etcd endpoints from environment or return None."""
        endpoints_str = os.getenv('ETCD_ENDPOINTS')
        if endpoints_str:
            return endpoints_str.split(',')
        return None

    @staticmethod
    def _get_grpc_options_from_env() -> Optional[List[Tuple[str, Any]]]:
        """Get gRPC options from environment."""
        options_str = os.getenv('ETCD_GRPC_OPTIONS')
        if not options_str:
            return None

        options = []
        for option_pair in options_str.split(','):
            if ':' in option_pair:
                key, value = option_pair.split(':', 1)
                try:
                    if value.lower() == 'true':
                        value = True
                    elif value.lower() == 'false':
                        value = False
                    elif '.' in value:
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    pass  # Keep as string
                options.append((key.strip(), value))

        return options if options else None

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "EtcdConfig":
        """Create an EtcdConfig instance from a dictionary."""
        config = cls(
            host=config_dict.get('host', os.getenv('ETCD_HOST', 'localhost')),
            port=config_dict.get('port', int(os.getenv('ETCD_PORT', 2379))),
            endpoints=config_dict.get('endpoints', cls._get_endpoints_from_env()),
            namespace=config_dict.get(
                'namespace', os.getenv('ETCD_NAMESPACE', 'marie')
            ),
            timeout_sec=config_dict.get(
                'timeout_sec', float(os.getenv('ETCD_TIMEOUT_SEC', '30.0'))
            ),
            retry_times=config_dict.get(
                'retry_times', int(os.getenv('ETCD_RETRY_TIMES', '10'))
            ),
            ca_cert=config_dict.get('ca_cert', os.getenv('ETCD_CA_CERT')),
            cert_key=config_dict.get('cert_key', os.getenv('ETCD_CERT_KEY')),
            cert_cert=config_dict.get('cert_cert', os.getenv('ETCD_CERT_CERT')),
            grpc_options=config_dict.get(
                'grpc_options', cls._get_grpc_options_from_env()
            ),
            lease_sec=config_dict.get(
                'lease_sec', int(os.getenv('ETCD_LEASE_SEC', '6'))
            ),
            heartbeat_sec=config_dict.get(
                'heartbeat_sec', float(os.getenv('ETCD_HEARTBEAT_SEC', '1.5'))
            ),
            service_name=config_dict.get(
                'service_name', os.getenv('ETCD_SERVICE_NAME', 'gateway/marie')
            ),
        )
        return config


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
        """Create EtcdClient, prioritizing args > environment > defaults."""
        if cls._args:
            etcd_config = EtcdConfig.from_dict(cls._args)
        else:
            etcd_config = EtcdConfig()

        if etcd_config._used_env:
            warnings.warn(
                "No explicit arguments provided; configuration values were sourced from environment variables. "
                "Ensure this is intentional.",
                UserWarning,
            )

        grpc_options = etcd_config.grpc_options
        if isinstance(grpc_options, dict):
            grpc_options = [(k, v) for k, v in grpc_options.items()]

        return EtcdClient(
            etcd_host=etcd_config.host,
            etcd_port=etcd_config.port,
            endpoints=etcd_config.endpoints,
            namespace=etcd_config.namespace,
            timeout=etcd_config.timeout_sec,
            retry_times=etcd_config.retry_times,
            ca_cert=etcd_config.ca_cert,
            cert_key=etcd_config.cert_key,
            cert_cert=etcd_config.cert_cert,
            grpc_options=grpc_options,
        )

    @classmethod
    def reset(cls):
        """Reset the container (for testing purposes)."""
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
