from unittest.mock import Mock

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
