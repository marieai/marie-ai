import marie.engine.llm_queue.metrics as metrics_module
import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader


def test_dispatch_metrics_observe_drr_lane_values(monkeypatch):
    if metrics_module.Observation is None:
        pytest.skip("OpenTelemetry Observation is unavailable")

    monkeypatch.setattr(
        metrics_module,
        "dispatch_runtime_snapshot",
        lambda: {
            "mode": "queued-dispatch",
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
        "mode": "queued-dispatch",
        "dispatcher_id": "default:drr:abc",
        "fabric_group_id": "fabric-a",
        "gateway_id": "gateway-a",
        "scheduler_policy": "drr",
        "pool_id": "interactive",
    }


def test_dispatch_metrics_export_queued_mode_with_execution_and_outcomes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        metrics_module,
        'dispatch_runtime_snapshot',
        lambda: {
            'mode': 'queued-dispatch',
            'registered_dispatchers': 0,
            'running_dispatchers': 0,
            'dispatchers': [],
        },
    )
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    try:
        metrics = metrics_module.DispatchMetrics(provider.get_meter('dispatch-test'))
        metrics.record_request_execution(
            pool_id='pool',
            dispatcher_id='dispatcher',
            fabric_group_id='fabric',
            duration_seconds=0.5,
            queue_wait_seconds=0.25,
            ok=False,
            category='outcome_unknown',
            remote_settled=False,
        )
        metrics.record_request_outcome(
            pool_id='pool',
            dispatcher_id='dispatcher',
            fabric_group_id='fabric',
            disposition='deferred',
            category='pool_pressure',
        )
        exported = {
            metric.name: metric.data.data_points
            for resource in reader.get_metrics_data().resource_metrics
            for scope in resource.scope_metrics
            for metric in scope.metrics
        }
        for name in (
            'marie_llm_dispatch_request_seconds',
            'marie_llm_dispatch_queue_wait_seconds',
            'marie_llm_dispatch_execution_failures',
        ):
            point = exported[name][0]
            assert point.attributes == {
                'pool_id': 'pool',
                'dispatcher_id': 'dispatcher',
                'fabric_group_id': 'fabric',
                'mode': 'queued-dispatch',
                'status': 'error',
                'category': 'outcome_unknown',
                'remote_settled': False,
            }
        assert exported['marie_llm_dispatch_request_seconds'][0].sum == 0.5
        assert exported['marie_llm_dispatch_queue_wait_seconds'][0].sum == 0.25
        assert exported['marie_llm_dispatch_execution_failures'][0].value == 1
        outcome = exported['marie_llm_dispatch_request_outcomes'][0]
        assert outcome.value == 1
        assert outcome.attributes == {
            'pool_id': 'pool',
            'dispatcher_id': 'dispatcher',
            'fabric_group_id': 'fabric',
            'mode': 'queued-dispatch',
            'disposition': 'deferred',
            'category': 'pool_pressure',
        }
        for name in (
            'marie_llm_dispatch_registered_dispatchers',
            'marie_llm_dispatch_running_dispatchers',
        ):
            assert exported[name][0].value == 0
            assert exported[name][0].attributes == {'mode': 'queued-dispatch'}
    finally:
        provider.shutdown()
