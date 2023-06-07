import asyncio

import pytest
from grpc import ChannelConnectivity

from marie.serve.networking.connection_stub import _ConnectionStubs
from marie.serve.networking.instrumentation import _NetworkingHistograms
from marie.serve.networking.replica_list import _ReplicaList


@pytest.fixture()
def replica_list(logger, metrics):
    return _ReplicaList(
        metrics=metrics,
        histograms=_NetworkingHistograms(),
        logger=logger,
        runtime_name='test',
        load_balancer_type='least_connection',
    )

def test_add_connection(replica_list):
    replica_list.add_connection('executor0', 'executor-0')
    assert replica_list.has_connections()
    assert replica_list.has_connection('executor0')
    assert len(replica_list.warmup_stubs)
    assert not replica_list.has_connection('random-address')
    assert len(replica_list.get_all_connections()) == 1


@pytest.mark.asyncio
async def test_remove_connection(replica_list):
    replica_list.add_connection('executor0', 'executor-0')
    assert replica_list.has_connections()
    await replica_list.remove_connection('executor0')
    assert not replica_list.has_connections()
    assert not replica_list.has_connection('executor0')
    # warmup stubs are not updated in the remove_connection method
    assert len(replica_list.warmup_stubs)
    # unknown/unmanaged connections
    removed_connection_invalid = await replica_list.remove_connection('random-address')
    assert removed_connection_invalid is None
    assert len(replica_list.get_all_connections()) == 0


@pytest.mark.asyncio
async def test_get_next_connection(replica_list):
    replica_list.add_connection('executor0', 'dep-0')
    replica_list.add_connection('executor1', 'dep-0')
    replica_list.add_connection('executor2', 'dep-0')
    replica_list.add_connection('executor3', 'dep-0')

    assert (await replica_list.get_next_connection()).address == 'executor0'
    assert (await replica_list.get_next_connection()).address == 'executor1'
    assert (await replica_list.get_next_connection()).address == 'executor2'
    assert (await replica_list.get_next_connection()).address == 'executor3'


@pytest.mark.asyncio
async def test_get_next_connection_counter(replica_list):
    replica_list.add_connection('executor0', 'dep-0')
    replica_list.add_connection('executor1', 'dep-0')
    load_balancer = replica_list.get_load_balancer()

    assert (await replica_list.get_next_connection()).address == 'executor0'
    load_balancer.incr_usage('executor0')
    assert (await replica_list.get_next_connection()).address == 'executor1'
    load_balancer.incr_usage('executor1')
    assert (await replica_list.get_next_connection()).address == 'executor0'
    load_balancer.incr_usage('executor0')
    assert (await replica_list.get_next_connection()).address == 'executor1'
    load_balancer.incr_usage('executor1')
    assert (await replica_list.get_next_connection()).address == 'executor0'
    load_balancer.incr_usage('executor0')

    assert replica_list.get_load_balancer().get_active_count('executor0') == 3
    assert replica_list.get_load_balancer().get_active_count('executor1') == 2

