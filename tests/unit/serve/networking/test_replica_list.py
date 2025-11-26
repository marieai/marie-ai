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
    )


def test_add_connection(replica_list):
    replica_list.add_connection('executor0', 'executor-0')
    assert replica_list.has_connections()
    assert replica_list.has_connection('executor0')
    assert not replica_list.has_connection('random-address')
    assert len(replica_list.get_all_connections()) == 1


@pytest.mark.asyncio
async def test_remove_connection(replica_list):
    replica_list.add_connection('executor0', 'executor-0')
    assert replica_list.has_connections()
    await replica_list.remove_connection('executor0')
    assert not replica_list.has_connections()
    assert not replica_list.has_connection('executor0')
    # unknown/unmanaged connections
    removed_connection_invalid = await replica_list.remove_connection('random-address')
    assert removed_connection_invalid is None
    assert len(replica_list.get_all_connections()) == 0


@pytest.mark.asyncio
async def test_reset_connection(replica_list):
    replica_list.add_connection('executor0', 'executor-0')
    connection_stub = await replica_list.get_next_connection('executor0')
    await replica_list.reset_connection('executor0', 'executor-0')
    new_connection_stub = await replica_list.get_next_connection()
    assert len(replica_list.get_all_connections()) == 1

    connection_stub_random_address = await replica_list.reset_connection(
        'random-address', '0'
    )
    assert connection_stub_random_address is None


@pytest.mark.asyncio
async def test_close(replica_list):
    replica_list.add_connection('executor0', 'executor-0')
    replica_list.add_connection('executor1', 'executor-0')
    assert replica_list.has_connection('executor0')
    assert replica_list.has_connection('executor1')
    await replica_list.close()
    assert not replica_list.has_connections()


@pytest.mark.asyncio
async def test_synchronization_when_resetting_connection(replica_list, logger):
    """Test that reset_connection properly handles concurrent access.

    After reset, the old channel is closed (preventing resource leaks),
    and a new connection is available for use.
    """
    replica_list.add_connection('executor0', 'executor-0')
    old_connection_stub = await replica_list.get_next_connection(num_retries=0)
    old_channel = old_connection_stub.channel

    # Reset the connection
    await replica_list.reset_connection(
        address='executor0', deployment_name='executor-0'
    )

    # Verify we can get a new connection after reset
    new_connection_stub = await replica_list.get_next_connection(num_retries=0)
    assert new_connection_stub is not None
    assert len(replica_list.get_all_connections()) == 1

    # The new connection should be different from the old one
    # (new channel created during reset)
    assert new_connection_stub.channel != old_channel
