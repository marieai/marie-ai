import pytest

import marie.engine.llm_queue.metrics as metrics_module


def test_dispatch_metrics_observe_drr_lane_values(monkeypatch):
    if metrics_module.Observation is None:
        pytest.skip("OpenTelemetry Observation is unavailable")

    monkeypatch.setattr(
        metrics_module,
        "dispatch_runtime_snapshot",
        lambda: {
            "contract_version": "v2",
            "dispatchers": [
                {
                    "dispatcher_id": "default:drr:abc",
                    "fabric_group_id": "fabric-a",
                    "gateway_id": "gateway-a",
                    "scheduler_policy": "drr",
                    "lanes": [
                        {
                            "pool_id": "interactive",
                            "request_queue_depth": 3,
                            "inflight": 2,
                            "deficit": 5,
                            "quantum": 8,
                            "oldest_pending_age_seconds": 12.5,
                        }
                    ],
                }
            ],
        },
    )

    depth = metrics_module.dispatch_metrics._observe_lane_depth(None)
    inflight = metrics_module.dispatch_metrics._observe_lane_inflight(None)
    oldest = metrics_module.dispatch_metrics._observe_lane_oldest_pending_seconds(None)

    assert depth[0].value == 3
    assert inflight[0].value == 2
    assert oldest[0].value == 12.5
    assert depth[0].attributes == {
        "contract_version": "v2",
        "dispatcher_id": "default:drr:abc",
        "fabric_group_id": "fabric-a",
        "gateway_id": "gateway-a",
        "scheduler_policy": "drr",
        "pool_id": "interactive",
    }
