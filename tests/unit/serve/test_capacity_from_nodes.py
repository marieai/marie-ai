import inspect
from unittest.mock import Mock

import pytest

from marie.state.semaphore_store import SemaphoreStore
from marie.state.slot_capacity_manager import SlotCapacityManager


@pytest.fixture
def mgr(mocker, monkeypatch):
    monkeypatch.delenv("MARIE_SLOTS_PER_NODE", raising=False)
    sem = mocker.Mock(spec=SemaphoreStore)
    return SlotCapacityManager(sem, logger=Mock())


def test_targets_derive_only_from_nodes(mgr):
    nodes = {
        "vector_store_executor": [
            {"address": "grpc://10.0.0.1:58285", "gateway": "10.0.0.1:60817"}
        ],
        "extract_executor": [
            {"address": "grpc://10.0.0.1:53756", "gateway": "10.0.0.1:53756"},
            {"address": "grpc://10.0.0.2:53756", "gateway": "10.0.0.2:53756"},
        ],
    }
    targets = mgr._capacity_targets_from_nodes(nodes)
    assert targets == {"vector_store_executor": 1, "extract_executor": 2}


def test_duplicate_addresses_count_once(mgr):
    nodes = {
        "extract_executor": [
            {"address": "grpc://10.0.0.1:53756", "gateway": "a"},
            {"address": "10.0.0.1:53756", "gateway": "b"},  # same netloc
        ]
    }
    assert mgr._capacity_targets_from_nodes(nodes) == {"extract_executor": 1}


def test_empty_node_list_yields_zero_target(mgr):
    assert mgr._capacity_targets_from_nodes({"extract_executor": []}) == {
        "extract_executor": 0
    }
    assert mgr._capacity_targets_from_nodes({}) == {}


def test_worker_serving_status_is_not_a_capacity_input():
    """Guardrail: NOT_SERVING means idle/ready in this codebase
    (request_handling.py:1937). Capacity must never read worker status —
    it derives from deployment_nodes membership only."""
    src = inspect.getsource(SlotCapacityManager)
    assert "NOT_SERVING" not in src
    assert "ServingStatus" not in src
    assert "status_store" not in src.lower()
