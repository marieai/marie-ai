import pytest

from marie.serve.discovery.registry import EtcdServiceRegistry
from marie.serve.discovery.resolver import EtcdServiceResolver


@pytest.fixture(scope='function')
def grpc_resolver(mocker, grpc_addr):
    client = mocker.Mock()
    resolver = EtcdServiceResolver(etcd_client=client, start_listener=False)

    def resolve(service_name):
        return grpc_addr

    old_resolve = resolver.resolve
    setattr(resolver, 'resolve', resolve)
    yield resolver

    setattr(resolver, 'resolve', old_resolve)


@pytest.fixture(scope='function')
def etcd_registry(mocker):
    client = mocker.Mock()
    registry = EtcdServiceRegistry(etcd_client=client)

    return registry
