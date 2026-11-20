import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from marie.serve.runtimes.gateway.marie.infrastructure_metrics import (
    capacity_pool_identity,
    gateway_metric_attributes,
)
from marie.serve.runtimes.servers.marie_gateway import MarieServerGateway


def test_metric_identity_keeps_configured_gateway_separate_from_process() -> None:
    attributes = gateway_metric_attributes(
        "default", "gateway-localhost", "process-a", 100.0
    )
    assert attributes == {
        "fabric_group_id": "default",
        "gateway_id": "gateway-localhost",
        "gateway_instance_id": "process-a",
        "gateway_started_at": "100.0",
    }
    assert (
        gateway_metric_attributes("default", "gateway-localhost", "process-a", 200.0)
        != attributes
    )


@pytest.mark.parametrize(
    "fabric,gateway",
    [("bad fabric", "gateway-a"), ("default", None), ("default", "bad/gateway")],
)
def test_missing_or_invalid_identity_is_not_assigned_to_another_gateway(
    fabric: str, gateway: str | None
) -> None:
    attributes = gateway_metric_attributes(fabric, gateway, "process-a", 100.0)
    assert not attributes["fabric_group_id"] or not attributes["gateway_id"]


def test_capacity_pool_identity_tracks_cluster_and_effective_storage_prefix() -> None:
    pool = capacity_pool_identity(123, "marie")
    assert pool != capacity_pool_identity(123, "/marie/")
    assert pool != capacity_pool_identity(456, "marie")
    assert pool != capacity_pool_identity(123, "other")
    assert len(pool) == 64


@pytest.mark.parametrize("queue_enabled", [True, False])
def test_all_infrastructure_metrics_include_fabric_gateway_and_process(
    monkeypatch: pytest.MonkeyPatch, queue_enabled: bool
) -> None:
    monkeypatch.delenv("MARIE_GATEWAY_ID", raising=False)
    monkeypatch.delenv("LLM_QUEUE_FABRIC_GROUP_ID", raising=False)
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    gateway = object.__new__(MarieServerGateway)
    gateway.meter = provider.get_meter("gateway-test")
    gateway.args = {
        "gateway_id": "gateway-a",
        "llm_queue": {"fabric_group_id": "fabric-a"},
    }
    gateway.gateway_instance_id = "process-a"
    gateway._infrastructure_started_at = 100.0
    gateway._metric_capacity_pool_id = "shared-pool"
    gateway._slot_observed_at = time.monotonic()
    gateway._slot_observed_at_unix = 1000.0
    gateway.logger = Mock()
    gateway.llm_dispatch_runtime = SimpleNamespace(
        config=SimpleNamespace(fabric_group_id="fabric-a", gateway_id="gateway-a"),
        enabled=queue_enabled,
    )
    gateway._slot_observations = {
        "capacity": {"executor": 4, "_total": 4},
        "used": {"executor": 1, "_total": 1},
        "available": {"executor": 3, "_total": 3},
    }
    gateway._node_observations = {
        "active_requests": {("executor", "node-a:5000"): 1},
        "slot_capacity": {("executor", "node-a:5000"): 4},
        "accepting_traffic": {("executor", "node-a:5000"): 1},
        "selection_count": {("executor", "node-a:5000"): 8},
    }
    gateway._setup_infrastructure_metrics()
    gateway._gateway_request_seconds.record(
        0.25,
        {
            **gateway._infrastructure_metric_scope,
            "endpoint": "/api/v1/invoke",
            "status": "success",
        },
    )
    data = reader.get_metrics_data()
    metrics = {
        metric.name: metric
        for resource in data.resource_metrics
        for scope in resource.scope_metrics
        for metric in scope.metrics
    }
    assert len(metrics) == 10
    assert (
        list(metrics["marie_executor_slot_observed_at"].data.data_points)[0].value
        == 1000.0
    )
    assert "marie_gateway_up" in metrics
    for metric in metrics.values():
        for point in metric.data.data_points:
            assert point.attributes["fabric_group_id"] == "fabric-a"
            assert point.attributes["gateway_id"] == "gateway-a"
            assert point.attributes["gateway_instance_id"] == "process-a"
            assert point.attributes["gateway_started_at"] == "100.0"
            if metric.name.startswith("marie_executor_slot_"):
                assert point.attributes["capacity_pool_id"] == "shared-pool"
    repeated = reader.get_metrics_data()
    repeated_timestamps = [
        point.value
        for resource in repeated.resource_metrics
        for scope in resource.scope_metrics
        for metric in scope.metrics
        if metric.name == "marie_executor_slot_observed_at"
        for point in metric.data.data_points
    ]
    assert repeated_timestamps == [1000.0, 1000.0]
    gateway._slot_observed_at = time.monotonic() - 181
    data = reader.get_metrics_data()
    assert not any(
        metric.data.data_points
        for resource in data.resource_metrics
        for scope in resource.scope_metrics
        for metric in scope.metrics
        if metric.name.startswith("marie_executor_slot_")
    )
    gateway._slot_observed_at = time.monotonic()
    gateway._metric_capacity_pool_id = None
    data = reader.get_metrics_data()
    assert not any(
        metric.data.data_points
        for resource in data.resource_metrics
        for scope in resource.scope_metrics
        for metric in scope.metrics
        if metric.name.startswith("marie_executor_slot_")
    )
    provider.shutdown()


def test_shared_capacity_pool_uses_etcd_cluster_identity_without_writes() -> None:
    gateway = object.__new__(MarieServerGateway)
    gateway._metric_capacity_pool_id = None
    gateway.etcd_client = Mock(ns="marie")
    gateway.etcd_client.client.get_response.return_value = SimpleNamespace(
        header=SimpleNamespace(cluster_id=123)
    )
    gateway._refresh_metric_capacity_pool_identity()
    assert gateway._metric_capacity_pool_id == capacity_pool_identity(123, "marie")
    gateway.etcd_client.client.put.assert_not_called()
    gateway._refresh_metric_capacity_pool_identity()
    gateway.etcd_client.client.get_response.assert_called_once()


@pytest.mark.parametrize(
    "config,env_fabric,expected",
    [
        ({}, None, ""),
        ({"fabric_group_id": "fabric-a"}, None, "fabric-a"),
        ({}, "fabric-env", "fabric-env"),
        (
            {"scheduler": {"fabric_group_id": "fabric-scheduler"}},
            None,
            "fabric-scheduler",
        ),
        ({"fabric_group_id": "fabric-a"}, "fabric-b", ""),
    ],
)
def test_disabled_queue_metrics_require_explicit_consistent_fabric(
    monkeypatch: pytest.MonkeyPatch,
    config: dict,
    env_fabric: str | None,
    expected: str,
) -> None:
    from marie.serve.runtimes.gateway.marie.llm_dispatch_runtime import (
        GatewayLlmDispatchRuntime,
    )

    monkeypatch.setenv("LLM_QUEUE_ENABLED", "false")
    monkeypatch.delenv("LLM_QUEUE_FABRIC_GROUP_ID", raising=False)
    if env_fabric is not None:
        monkeypatch.setenv("LLM_QUEUE_FABRIC_GROUP_ID", env_fabric)
    gateway = object.__new__(MarieServerGateway)
    gateway.meter = None
    gateway.args = {"gateway_id": "gateway-a", "llm_queue": config}
    gateway.gateway_instance_id = "process-a"
    gateway._infrastructure_started_at = 100.0
    gateway.llm_dispatch_runtime = GatewayLlmDispatchRuntime(config=config)
    gateway._setup_infrastructure_metrics()
    assert gateway._infrastructure_metric_scope["fabric_group_id"] == expected


def test_telemetry_does_not_require_a_second_configured_gateway_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from marie.serve.runtimes.gateway.marie.llm_dispatch_runtime import (
        GatewayLlmDispatchRuntime,
    )

    monkeypatch.setenv('LLM_QUEUE_ENABLED', 'false')
    monkeypatch.setenv('LLM_QUEUE_FABRIC_GROUP_ID', 'default')
    monkeypatch.delenv('LLM_QUEUE_GATEWAY_ID', raising=False)
    monkeypatch.delenv('MARIE_GATEWAY_ID', raising=False)
    gateway = object.__new__(MarieServerGateway)
    gateway.meter = None
    gateway.args = {'llm_queue': {'fabric_group_id': 'default'}}
    gateway.gateway_instance_id = 'already-existing-runtime-instance'
    gateway._infrastructure_started_at = 100.0
    gateway.llm_dispatch_runtime = GatewayLlmDispatchRuntime(
        config=gateway.args['llm_queue']
    )
    gateway._setup_infrastructure_metrics()
    assert gateway._infrastructure_metric_scope['gateway_id'] == ''
    assert (
        gateway._infrastructure_metric_scope['gateway_instance_id']
        == 'already-existing-runtime-instance'
    )
