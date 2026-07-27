from unittest.mock import AsyncMock, Mock

import pytest

from marie.serve.runtimes.gateway.marie.llm_dispatch_runtime import (  # noqa: F401
    GatewayLlmDispatchRuntime,
)
from marie.serve.runtimes.servers.marie_gateway import MarieServerGateway


def test_node_capacity_snapshot_classifies_live_router_load() -> None:
    gateway = object.__new__(MarieServerGateway)
    gateway.deployment_nodes = {
        "executor": [
            {
                "address": "grpc://node-a:5000",
                "gateway": "gateway-a:51000",
            },
            {
                "address": "grpc://node-b:5000",
                "gateway": "gateway-b:51000",
            },
        ]
    }
    gateway.desired_map = {}
    gateway.status_map = {}
    gateway._service_readiness = {}
    gateway.capacity_manager = Mock()
    gateway.capacity_manager.slots_per_node.return_value = 2
    gateway.streamer = Mock()
    gateway.streamer.get_node_stats.return_value = [
        {
            "executor": "executor",
            "address": "node-a:5000",
            "active_requests": 2,
            "selection_count": 12,
            "accepting_traffic": True,
        },
        {
            "executor": "executor",
            "address": "node-b:5000",
            "active_requests": 0,
            "selection_count": 3,
            "accepting_traffic": True,
        },
    ]

    snapshot = gateway._node_capacity_snapshot()

    assert snapshot[0] == {
        "executor": "executor",
        "address": "node-a:5000",
        "active_requests": 2,
        "selection_count": 12,
        "accepting_traffic": True,
        "gateways": ["gateway-a:51000"],
        "slot_capacity": 2,
        "slot_available": 0,
        "utilization_pct": 100.0,
        "routing_state": "saturated",
    }
    assert snapshot[1]["gateways"] == ["gateway-b:51000"]
    assert snapshot[1]["routing_state"] == "idle"
    assert snapshot[1]["utilization_pct"] == 0.0


def test_node_observations_use_executor_and_address_labels() -> None:
    gateway = object.__new__(MarieServerGateway)
    gateway._set_node_observations(
        [
            {
                "executor": "executor",
                "address": "node-a:5000",
                "active_requests": 2,
                "slot_capacity": 4,
                "accepting_traffic": False,
                "selection_count": 12,
            }
        ]
    )

    key = ("executor", "node-a:5000")
    assert gateway._node_observations == {
        "active_requests": {key: 2},
        "slot_capacity": {key: 4},
        "accepting_traffic": {key: 0},
        "selection_count": {key: 12},
    }


@pytest.mark.asyncio
async def test_capacity_event_logs_slot_table_every_ten_seconds(
    monkeypatch,
) -> None:
    gateway = object.__new__(MarieServerGateway)
    rows = [("executor", 40, 40, 25, 15, 25, "")]
    totals = {
        "capacity": 40,
        "used": 25,
        "available": 15,
        "holder_count": 25,
    }
    gateway.capacity_manager = Mock()
    gateway.capacity_manager.refresh_from_nodes.return_value = (
        rows,
        totals,
        "full capacity table",
    )
    gateway._routable_deployment_nodes = Mock(return_value={})
    gateway._node_capacity_snapshot = Mock(return_value=[])
    gateway._set_node_observations = Mock()
    gateway._slot_observations = {"capacity": {}, "used": {}, "available": {}}
    gateway._last_capacity_info_log_at = 0.0
    gateway.logger = Mock()

    now = [10.0]
    monkeypatch.setattr(
        "marie.serve.runtimes.servers.marie_gateway.time.monotonic",
        lambda: now[0],
    )
    notify = AsyncMock()
    monkeypatch.setattr(
        "marie.serve.runtimes.servers.marie_gateway.Toast.notify", notify
    )

    await gateway._publish_capacity_event()
    now[0] = 15.0
    await gateway._publish_capacity_event()
    now[0] = 20.0
    await gateway._publish_capacity_event()

    assert gateway.logger.info.call_args_list == [
        (("full capacity table",), {}),
        (("full capacity table",), {}),
    ]
    assert (
        sum(
            call.args == ("full capacity table",)
            for call in gateway.logger.debug.call_args_list
        )
        == 1
    )
    assert notify.await_count == 3
