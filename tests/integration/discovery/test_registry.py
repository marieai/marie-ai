import time

import pytest

from marie.serve.discovery.etcd_manager import etcd_container


@pytest.mark.parametrize(
    'service_names, service_addr, service_ttl', (
            (('grpc.service_test', 'grpc.service_list'), '10.30.1.1.50011', 120),
            (('grpc.service_create', 'grpc.service_update'), '10.30.1.1.50011', 120),
    )
)
def test_service_registry(
        etcd_registry, service_names, service_addr, service_ttl):
    etcd_registry.register(service_names, service_addr, service_ttl)
    assert etcd_registry._services[service_addr] == set(service_names)
    assert service_addr in etcd_registry._leases

    etcd_registry.unregister((service_names[0],), service_addr)
    assert etcd_registry._services[service_addr] == {service_names[1]}
    assert service_addr in etcd_registry._leases


@pytest.fixture(scope='function')
def etcd_registry(mocker):
    # Reset container for each test
    etcd_container.reset()

    # Mock the EtcdClient
    client = mocker.Mock()
    etcd_container._etcd_client = client

    from marie.serve.discovery.registry import EtcdServiceRegistry
    registry = EtcdServiceRegistry(etcd_client=client)

    yield registry

    # Clean up after test
    etcd_container.reset()
