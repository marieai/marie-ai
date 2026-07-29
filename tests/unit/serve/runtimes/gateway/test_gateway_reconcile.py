import asyncio
from unittest import mock

import pytest
from grpc_health.v1.health_pb2 import HealthCheckResponse

from marie.serve.runtimes.gateway.marie.llm_dispatch_runtime import (  # noqa: F401
    GatewayLlmDispatchRuntime,
)
from marie.serve.runtimes.servers.cluster_state import ClusterState
from marie.serve.runtimes.servers.marie_gateway import (
    MAX_AGE_S,
    MAX_MISSES,
    STATUS_DEGRADED_LIVE_MISSING,
    STATUS_DEGRADED_REASON,
    STATUS_DEGRADED_SINCE,
    EventKind,
    MarieServerGateway,
    StateEvent,
)
from marie.state.state_store import DesiredDoc, StatusDoc


class _DesiredStore:
    def __init__(self, doc: DesiredDoc):
        self.doc = doc
        self.update_calls = 0
        self.bump_calls = 0

    def update_params(self, node, depl, updater):
        self.update_calls += 1
        self.doc = DesiredDoc(
            phase=self.doc.phase,
            epoch=self.doc.epoch,
            params=updater(dict(self.doc.params or {})),
            updated_at=self.doc.updated_at,
        )
        return self.doc

    def list_pairs(self):
        return [("node:1", "extract_executor")]

    def get(self, node, depl):
        return self.doc

    def bump_epoch(self, node, depl):
        self.bump_calls += 1
        self.doc = DesiredDoc(
            phase=self.doc.phase,
            epoch=self.doc.epoch + 1,
            params=dict(self.doc.params or {}),
            updated_at=self.doc.updated_at,
        )
        return self.doc


class _Etcd:
    def __init__(self):
        self.deleted_prefixes = []

    def delete_prefix(self, prefix):
        self.deleted_prefixes.append(prefix)


class _Logger:
    def __init__(self):
        self.records = []

    def _record(self, level, args, kwargs):
        self.records.append((level, args, kwargs))

    def debug(self, *args, **kwargs):
        self._record("debug", args, kwargs)

    def info(self, *args, **kwargs):
        self._record("info", args, kwargs)

    def warning(self, *args, **kwargs):
        self._record("warning", args, kwargs)

    def error(self, *args, **kwargs):
        self._record("error", args, kwargs)


class _SemaphoreStore:
    def reconcile_all(self, **kwargs):
        return {}


class _StatusStore:
    def __init__(self, status):
        self.status = status

    def read(self, node, depl):
        return self.status


def _gateway(doc: DesiredDoc):
    gateway = object.__new__(MarieServerGateway)
    gateway.desired_store = _DesiredStore(doc)
    gateway.status_store = _StatusStore(None)
    gateway.semaphore_store = _SemaphoreStore()
    gateway.etcd_client = _Etcd()
    gateway.desired_map = {("node:1", "extract_executor"): {}}
    gateway.status_map = {("node:1", "extract_executor"): {}}
    gateway.deployment_nodes = {}
    gateway._service_readiness = {}
    gateway.logger = _Logger()
    return gateway


def test_status_miss_sets_and_preserves_missing_since():
    gateway = _gateway(
        DesiredDoc(
            phase="SCHEDULED",
            epoch=10,
            params={},
            updated_at="2000-01-01T00:00:00Z",
        )
    )
    stale_calls = []

    def is_stale(ts, timeout):
        stale_calls.append((ts, timeout))
        return False

    with (
        mock.patch(
            "marie.serve.runtimes.servers.marie_gateway._now_iso",
            side_effect=["2026-05-14T17:00:00Z", "2026-05-14T17:01:00Z"],
        ),
        mock.patch(
            "marie.serve.runtimes.servers.marie_gateway.is_stale",
            side_effect=is_stale,
        ),
    ):
        assert gateway._incr_miss_and_maybe_gc("node:1", "extract_executor") is False
        assert gateway._incr_miss_and_maybe_gc("node:1", "extract_executor") is False

    params = gateway.desired_store.doc.params
    assert params["misses"] == 2
    assert params["missing_since"] == "2026-05-14T17:00:00Z"
    assert stale_calls == [
        ("2026-05-14T17:00:00Z", MAX_AGE_S),
        ("2026-05-14T17:00:00Z", MAX_AGE_S),
    ]
    assert gateway.etcd_client.deleted_prefixes == []


def test_status_miss_gc_uses_missing_since_not_desired_updated_at():
    gateway = _gateway(
        DesiredDoc(
            phase="SCHEDULED",
            epoch=10,
            params={"missing_since": "2000-01-01T00:00:00Z", "misses": 0},
            updated_at="2999-01-01T00:00:00Z",
        )
    )

    assert gateway._incr_miss_and_maybe_gc("node:1", "extract_executor") is True
    assert gateway.etcd_client.deleted_prefixes == [
        "deployments/node:1/extract_executor"
    ]


def test_live_status_miss_marks_degraded_instead_of_gc():
    gateway = _gateway(
        DesiredDoc(
            phase="SCHEDULED",
            epoch=10,
            params={
                "misses": MAX_MISSES - 1,
                "missing_since": "2026-05-14T17:00:00Z",
            },
            updated_at="2026-05-14T17:02:00Z",
        )
    )
    gateway.deployment_nodes = {
        "extract_executor": [{"address": "grpc://node:1"}],
    }

    with mock.patch(
        "marie.serve.runtimes.servers.marie_gateway._now_iso",
        side_effect=["2026-05-14T17:03:00Z", "2026-05-14T17:04:00Z"],
    ):
        assert gateway._incr_miss_and_maybe_gc("node:1", "extract_executor") is False

    params = gateway.desired_store.doc.params
    assert params["misses"] == MAX_MISSES
    assert params[STATUS_DEGRADED_SINCE] == "2026-05-14T17:04:00Z"
    assert params[STATUS_DEGRADED_REASON] == STATUS_DEGRADED_LIVE_MISSING
    assert gateway.etcd_client.deleted_prefixes == []
    assert any(
        kwargs.get("extra", {}).get("event_type") == "gateway_status_degraded_live_node"
        for _, _, kwargs in gateway.logger.records
    )


def test_degraded_registration_is_excluded_from_routing():
    gateway = _gateway(
        DesiredDoc(
            phase="SCHEDULED",
            epoch=10,
            params={},
            updated_at="2026-05-14T17:02:00Z",
        )
    )
    gateway.deployment_nodes = {
        "extract_executor": [
            {"address": "grpc://node-a:5000"},
            {"address": "grpc://node-b:5000"},
        ]
    }
    gateway.desired_map = {
        ("node-a:5000", "extract_executor"): DesiredDoc(
            phase="SCHEDULED",
            epoch=10,
            params={},
            updated_at="2026-05-14T17:02:00Z",
        ),
        ("node-b:5000", "extract_executor"): DesiredDoc(
            phase="SCHEDULED",
            epoch=20,
            params={STATUS_DEGRADED_SINCE: "2026-05-14T17:03:00Z"},
            updated_at="2026-05-14T17:03:00Z",
        ),
    }
    gateway.status_map = {
        ("node-a:5000", "extract_executor"): StatusDoc(
            status_code=HealthCheckResponse.SERVING,
            status_name="SERVING",
            owner="worker-a",
            epoch=10,
            updated_at="2026-05-14T17:03:00Z",
            heartbeat_at="2026-05-14T17:03:00Z",
            details=None,
        )
    }

    assert gateway._address_is_registered("node-b:5000", "extract_executor")
    assert gateway._address_is_live("node-a:5000", "extract_executor")
    assert not gateway._address_is_live("node-b:5000", "extract_executor")
    assert gateway._routable_deployment_nodes() == {
        "extract_executor": [{"address": "grpc://node-a:5000"}]
    }


@pytest.mark.asyncio
async def test_quarantine_and_recovery_refresh_routing_and_capacity():
    gateway = _gateway(
        DesiredDoc(
            phase="SCHEDULED",
            epoch=10,
            params={},
            updated_at="2026-05-14T17:02:00Z",
        )
    )
    gateway.deployment_nodes = {"extract_executor": [{"address": "grpc://node:1"}]}
    gateway.desired_map = {
        ("node:1", "extract_executor"): DesiredDoc(
            phase="SCHEDULED",
            epoch=10,
            params={},
            updated_at="2026-05-14T17:02:00Z",
        )
    }
    gateway.state_events_queue = asyncio.Queue()
    gateway._schedule_rebuild = mock.Mock()
    gateway._publish_capacity_event = mock.AsyncMock()
    processor = asyncio.create_task(gateway.process_state_events())

    await gateway.state_events_queue.put(
        StateEvent(
            kind=EventKind.DESIRED,
            node="node:1",
            deployment="extract_executor",
            ev_type="put",
            value={
                "node:1": {
                    "extract_executor": {
                        "phase": "SCHEDULED",
                        "epoch": 10,
                        "params": {STATUS_DEGRADED_SINCE: "2026-05-14T17:03:00Z"},
                        "updated_at": "2026-05-14T17:03:00Z",
                    }
                }
            },
            key="deployments/node:1/extract_executor/desired",
        )
    )
    await asyncio.wait_for(gateway.state_events_queue.join(), timeout=1)

    gateway._schedule_rebuild.assert_called_once_with(True)
    gateway._publish_capacity_event.assert_awaited_once()

    gateway._schedule_rebuild.reset_mock()
    gateway._publish_capacity_event.reset_mock()
    await gateway.state_events_queue.put(
        StateEvent(
            kind=EventKind.DESIRED,
            node="node:1",
            deployment="extract_executor",
            ev_type="put",
            value={
                "node:1": {
                    "extract_executor": {
                        "phase": "SCHEDULED",
                        "epoch": 10,
                        "params": {},
                        "updated_at": "2026-05-14T17:04:00Z",
                    }
                }
            },
            key="deployments/node:1/extract_executor/desired",
        )
    )
    await asyncio.wait_for(gateway.state_events_queue.join(), timeout=1)

    gateway._schedule_rebuild.assert_called_once_with(True)
    gateway._publish_capacity_event.assert_awaited_once()

    processor.cancel()
    await asyncio.gather(processor, return_exceptions=True)


@pytest.mark.asyncio
async def test_status_updates_notify_only_for_admission_state_changes():
    gateway = _gateway(
        DesiredDoc(
            phase="SCHEDULED",
            epoch=10,
            params={},
            updated_at="2026-05-14T17:02:00Z",
        )
    )
    gateway.status_map = {
        ("node:1", "extract_executor"): StatusDoc(
            status_code=HealthCheckResponse.SERVING,
            status_name="SERVING",
            owner="worker-a",
            epoch=10,
            updated_at="2026-05-14T17:03:00Z",
            heartbeat_at="2026-05-14T17:03:00Z",
            details=None,
        )
    }
    gateway.state_events_queue = asyncio.Queue()
    gateway._schedule_rebuild = mock.Mock()
    gateway._publish_capacity_event = mock.AsyncMock()

    def status_event(
        status_code: int, status_name: str, heartbeat_at: str
    ) -> StateEvent:
        return StateEvent(
            kind=EventKind.STATUS,
            node="node:1",
            deployment="extract_executor",
            ev_type="put",
            value={
                "node:1": {
                    "extract_executor": {
                        "status": {
                            "status_code": status_code,
                            "status_name": status_name,
                            "owner": "worker-a",
                            "epoch": 10,
                            "updated_at": heartbeat_at,
                            "heartbeat_at": heartbeat_at,
                        }
                    }
                }
            },
            key="deployments/node:1/extract_executor/status",
        )

    with mock.patch.object(ClusterState, "notify_deployment_update") as notify:
        processor = asyncio.create_task(gateway.process_state_events())
        await gateway.state_events_queue.put(
            status_event(
                HealthCheckResponse.SERVING,
                "SERVING",
                "2026-05-14T17:04:00Z",
            )
        )
        await asyncio.wait_for(gateway.state_events_queue.join(), timeout=1)
        notify.assert_not_called()

        await gateway.state_events_queue.put(
            status_event(
                HealthCheckResponse.NOT_SERVING,
                "NOT_SERVING",
                "2026-05-14T17:05:00Z",
            )
        )
        await asyncio.wait_for(gateway.state_events_queue.join(), timeout=1)
        notify.assert_called_once_with()

        processor.cancel()
        await asyncio.gather(processor, return_exceptions=True)


@pytest.mark.asyncio
async def test_gateway_streamer_receives_only_routable_nodes():
    gateway = _gateway(
        DesiredDoc(
            phase="SCHEDULED",
            epoch=10,
            params={},
            updated_at="2026-05-14T17:02:00Z",
        )
    )
    gateway.deployment_nodes = {
        "extract_executor": [
            {"address": "grpc://node-a:5000"},
            {"address": "grpc://node-b:5000"},
        ]
    }
    gateway.desired_map = {
        ("node-b:5000", "extract_executor"): DesiredDoc(
            phase="SCHEDULED",
            epoch=20,
            params={STATUS_DEGRADED_SINCE: "2026-05-14T17:03:00Z"},
            updated_at="2026-05-14T17:03:00Z",
        )
    }
    gateway._can_update_incrementally = mock.Mock(return_value=True)
    gateway._apply_incremental_updates = mock.AsyncMock()
    gateway.streamer = mock.Mock()
    gateway.streamer.topology_graph.all_nodes = []
    gateway.distributor = mock.Mock()

    await gateway.update_gateway_streamer()

    gateway._apply_incremental_updates.assert_awaited_once_with(
        {"extract_executor": ["node-a:5000"]}
    )
    assert gateway.distributor.deployment_nodes == {
        "extract_executor": [{"address": "grpc://node-a:5000"}]
    }


def test_reset_miss_metadata_removes_only_reconcile_metadata():
    gateway = _gateway(
        DesiredDoc(
            phase="SCHEDULED",
            epoch=10,
            params={
                "misses": 3,
                "missing_since": "2026-05-14T17:00:00Z",
                STATUS_DEGRADED_SINCE: "2026-05-14T17:01:00Z",
                STATUS_DEGRADED_REASON: STATUS_DEGRADED_LIVE_MISSING,
                "keep": "value",
            },
            updated_at="2026-05-14T17:02:00Z",
        )
    )

    gateway._reset_miss_metadata("node:1", "extract_executor")

    doc = gateway.desired_store.doc
    assert doc.epoch == 10
    assert doc.phase == "SCHEDULED"
    assert doc.updated_at == "2026-05-14T17:02:00Z"
    assert doc.params == {"keep": "value"}


@pytest.mark.asyncio
async def test_healthy_reconcile_resets_miss_metadata():
    gateway = _gateway(
        DesiredDoc(
            phase="SCHEDULED",
            epoch=10,
            params={
                "misses": 3,
                "missing_since": "2026-05-14T17:00:00Z",
                STATUS_DEGRADED_SINCE: "2026-05-14T17:01:00Z",
                STATUS_DEGRADED_REASON: STATUS_DEGRADED_LIVE_MISSING,
                "keep": "value",
            },
            updated_at="2026-05-14T17:02:00Z",
        )
    )
    gateway.status_store = _StatusStore(
        StatusDoc(
            status_code=HealthCheckResponse.SERVING,
            status_name="SERVING",
            owner="worker",
            epoch=10,
            updated_at="2026-05-14T17:03:00Z",
            heartbeat_at="2026-05-14T17:03:00Z",
            details=None,
        )
    )

    async def stop_after_one_sleep(_interval):
        raise RuntimeError("stop-loop")

    with (
        mock.patch(
            "marie.serve.runtimes.servers.marie_gateway.is_stale",
            return_value=False,
        ),
        mock.patch(
            "marie.serve.runtimes.servers.marie_gateway.asyncio.sleep",
            side_effect=stop_after_one_sleep,
        ),
    ):
        with pytest.raises(RuntimeError, match="stop-loop"):
            await gateway._reconcile_loop(interval_s=0)

    assert gateway.desired_store.doc.params == {"keep": "value"}


@pytest.mark.asyncio
async def test_degraded_live_missing_status_suppresses_future_bumps():
    gateway = _gateway(
        DesiredDoc(
            phase="SCHEDULED",
            epoch=10,
            params={
                "misses": MAX_MISSES,
                "missing_since": "2026-05-14T17:00:00Z",
                STATUS_DEGRADED_SINCE: "2026-05-14T17:01:00Z",
                STATUS_DEGRADED_REASON: STATUS_DEGRADED_LIVE_MISSING,
            },
            updated_at="2026-05-14T17:02:00Z",
        )
    )
    gateway.deployment_nodes = {
        "extract_executor": [{"address": "node:1"}],
    }

    async def stop_after_one_sleep(_interval):
        raise RuntimeError("stop-loop")

    with (
        mock.patch(
            "marie.serve.runtimes.servers.marie_gateway.is_stale",
            return_value=True,
        ),
        mock.patch(
            "marie.serve.runtimes.servers.marie_gateway.asyncio.sleep",
            side_effect=stop_after_one_sleep,
        ),
    ):
        with pytest.raises(RuntimeError, match="stop-loop"):
            await gateway._reconcile_loop(interval_s=0)

    assert gateway.desired_store.bump_calls == 0
    assert gateway.desired_store.doc.epoch == 10


@pytest.mark.asyncio
async def test_epoch_mismatch_waits_during_claim_window():
    gateway = _gateway(
        DesiredDoc(
            phase="SCHEDULED",
            epoch=10,
            params={},
            updated_at="2026-05-14T17:02:00Z",
        )
    )
    gateway.status_store = _StatusStore(
        StatusDoc(
            status_code=HealthCheckResponse.SERVING,
            status_name="SERVING",
            owner="worker",
            epoch=9,
            updated_at="2026-05-14T17:03:00Z",
            heartbeat_at="2026-05-14T17:03:00Z",
            details=None,
        )
    )

    async def stop_after_one_sleep(_interval):
        raise RuntimeError("stop-loop")

    with (
        mock.patch(
            "marie.serve.runtimes.servers.marie_gateway.is_stale",
            return_value=False,
        ),
        mock.patch(
            "marie.serve.runtimes.servers.marie_gateway.asyncio.sleep",
            side_effect=stop_after_one_sleep,
        ),
    ):
        with pytest.raises(RuntimeError, match="stop-loop"):
            await gateway._reconcile_loop(interval_s=0)

    assert gateway.desired_store.bump_calls == 0
    assert gateway.desired_store.doc.params == {}


@pytest.mark.asyncio
async def test_epoch_mismatch_after_claim_timeout_bumps_and_records_miss():
    gateway = _gateway(
        DesiredDoc(
            phase="SCHEDULED",
            epoch=10,
            params={},
            updated_at="2026-05-14T17:02:00Z",
        )
    )
    gateway.status_store = _StatusStore(
        StatusDoc(
            status_code=HealthCheckResponse.SERVING,
            status_name="SERVING",
            owner="worker",
            epoch=9,
            updated_at="2026-05-14T17:03:00Z",
            heartbeat_at="2026-05-14T17:03:00Z",
            details=None,
        )
    )

    async def stop_after_one_sleep(_interval):
        raise RuntimeError("stop-loop")

    with (
        mock.patch(
            "marie.serve.runtimes.servers.marie_gateway.is_stale",
            side_effect=[True, False],
        ),
        mock.patch(
            "marie.serve.runtimes.servers.marie_gateway._now_iso",
            return_value="2026-05-14T17:04:00Z",
        ),
        mock.patch(
            "marie.serve.runtimes.servers.marie_gateway.asyncio.sleep",
            side_effect=stop_after_one_sleep,
        ),
    ):
        with pytest.raises(RuntimeError, match="stop-loop"):
            await gateway._reconcile_loop(interval_s=0)

    assert gateway.desired_store.bump_calls == 1
    assert gateway.desired_store.doc.epoch == 11
    assert gateway.desired_store.doc.params["misses"] == 1
    assert gateway.desired_store.doc.params["missing_since"] == "2026-05-14T17:04:00Z"

    bump_logs = [
        kwargs["extra"]
        for level, _, kwargs in gateway.logger.records
        if level == "warning"
        and kwargs.get("extra", {}).get("event_type") == "gateway_status_reconcile_bump"
    ]
    assert bump_logs
    assert bump_logs[-1]["desired_epoch"] == 10
    assert bump_logs[-1]["new_desired_epoch"] == 11
    assert bump_logs[-1]["current_desired_epoch"] == 11
    assert bump_logs[-1]["status_epoch"] == 9
    assert bump_logs[-1]["misses"] == 1
    assert bump_logs[-1]["node_live"] is False
