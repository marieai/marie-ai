import time
from types import SimpleNamespace

import marie.orchestrate.flow.base as flow_module
from marie.orchestrate.flow.base import Flow


def test_flow_stops_discovery_before_deployments(monkeypatch):
    order: list[str] = []
    flow = Flow()
    flow._etcd_registry = SimpleNamespace(
        shutdown=lambda: order.append("discovery")
    )
    flow.sd_state = "ready"
    flow._start_time = time.time()
    flow.callback(lambda: order.append("deployments"))
    monkeypatch.setattr(flow_module, "send_telemetry_event", lambda **_kwargs: None)

    flow.__exit__(None, None, None)

    assert order == ["discovery", "deployments"]
