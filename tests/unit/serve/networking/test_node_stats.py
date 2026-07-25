import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from marie.serve.networking import GrpcConnectionPool
from marie.serve.networking.balancer.least_connection_balancer import (
    LeastConnectionsLoadBalancer,
)
from marie.serve.networking.connection_pool_map import _ConnectionPoolMap


@pytest.mark.asyncio
async def test_node_stats_report_least_connection_load() -> None:
    load_balancer = LeastConnectionsLoadBalancer("executor", Mock())
    first = SimpleNamespace(address="node-a:5000")
    second = SimpleNamespace(address="node-b:5000")
    load_balancer.update_connections([first, second])
    load_balancer.incr_usage(first.address)

    selected = await load_balancer.get_next_connection()
    assert selected.address == second.address

    replica_list = Mock()
    replica_list.get_load_balancer.return_value = load_balancer
    connection_pool = object.__new__(_ConnectionPoolMap)
    connection_pool._lock = threading.RLock()
    connection_pool._deployments = {
        "executor": {"heads": {0: replica_list}, "shards": {}}
    }

    assert connection_pool.get_node_stats() == [
        {
            "executor": "executor",
            "address": "node-a:5000",
            "role": "head",
            "entity_id": 0,
            "active_requests": 1,
            "selection_count": 0,
            "accepting_traffic": True,
            "circuit_state": "disabled",
            "consecutive_failures": 0,
            "total_failures": 0,
            "total_successes": 0,
        },
        {
            "executor": "executor",
            "address": "node-b:5000",
            "role": "head",
            "entity_id": 0,
            "active_requests": 0,
            "selection_count": 1,
            "accepting_traffic": True,
            "circuit_state": "disabled",
            "consecutive_failures": 0,
            "total_failures": 0,
            "total_successes": 0,
        },
    ]


@pytest.mark.asyncio
async def test_single_document_requests_track_node_usage() -> None:
    events = []

    class Connection:
        address = "node-a:5000"
        deployment_name = "executor"

        async def send_single_doc_request(self, **kwargs):
            events.append("send")
            yield "response", "metadata"

    connection = Connection()
    replica_list = Mock()
    replica_list.get_all_connections.return_value = [connection]
    replica_list.get_next_connection = Mock(return_value=None)

    async def get_next_connection(**kwargs):
        return connection

    replica_list.get_next_connection.side_effect = get_next_connection
    replica_list.incr_usage.side_effect = lambda address: events.append("increment")
    replica_list.decr_usage.side_effect = lambda address: events.append("decrement")

    connection_pool = object.__new__(GrpcConnectionPool)
    connection_pool.compression = None
    request = SimpleNamespace(request_id="request-1")

    responses = [
        response
        async for response in connection_pool._send_single_doc_request(
            request=request,
            connections=replica_list,
            retries=0,
        )
    ]

    assert responses == [("response", "metadata")]
    assert events == ["increment", "send", "decrement"]
