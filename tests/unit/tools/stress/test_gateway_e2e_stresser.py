from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.stress.gateway_e2e_stresser import (
    GatewayE2EStresser,
    InputAsset,
    JobRun,
    _build_debug_snapshot,
    _build_input_assets,
    _coerce_gateway_debug_payload,
    _extract_failure_error,
    _extract_ref_id_from_event,
    _extract_template,
    _resolve_inputs,
    _resolve_s3_inputs,
)


class _FakeDebugResponse:
    def __init__(self, status: int, payload: dict) -> None:
        self.status = status
        self._payload = payload

    async def __aenter__(self) -> "_FakeDebugResponse":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False

    async def text(self) -> str:
        return json.dumps(self._payload)


class _FakeDebugSession:
    def __init__(self, response: _FakeDebugResponse) -> None:
        self.response = response
        self.closed = False
        self.last_url: str | None = None

    def get(self, url: str, **_: object) -> _FakeDebugResponse:
        self.last_url = url
        return self.response

    async def close(self) -> None:
        self.closed = True


def test_extract_template_from_invoke_action_payload() -> None:
    name, metadata = _extract_template(
        {
            "invoke_action": {
                "name": "mock_parallel_subgraphs",
                "metadata": {
                    "planner": "mock_parallel_subgraphs",
                    "ref_type": "stress",
                },
            }
        }
    )

    assert name == "mock_parallel_subgraphs"
    assert metadata == {
        "planner": "mock_parallel_subgraphs",
        "ref_type": "stress",
    }


def test_resolve_inputs_supports_manifest_and_absolute_glob(tmp_path: Path) -> None:
    first = tmp_path / "one.tif"
    second = tmp_path / "two.tif"
    ignored = tmp_path / "note.txt"
    first.write_text("a")
    second.write_text("b")
    ignored.write_text("c")

    manifest = tmp_path / "manifest.txt"
    manifest.write_text(f"{first}\n# comment\n{second}\n")

    manifest_inputs = _resolve_inputs(
        input_glob=None,
        input_dir=None,
        input_manifest=str(manifest),
    )
    glob_inputs = _resolve_inputs(
        input_glob=str(tmp_path / "*.tif"),
        input_dir=None,
        input_manifest=None,
    )

    assert manifest_inputs == [first.resolve(), second.resolve()]
    assert glob_inputs == [first.resolve(), second.resolve()]


def test_extract_ref_id_and_failure_reason_from_scheduler_event() -> None:
    message = {
        "event": "extract.failed",
        "payload": json.dumps(
            {
                "metadata": {
                    "ref_id": "job-123-sample.tif",
                },
                "error": "backend timeout",
            }
        ),
    }

    assert _extract_ref_id_from_event(message) == "job-123-sample.tif"
    assert _extract_failure_error(message) == "backend timeout"


def test_resolve_s3_inputs_supports_direct_uri_and_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "uris.txt"
    manifest.write_text(
        "s3://marie/gen5_extract/a.tif\n# comment\ns3://marie/gen5_extract/b.tif\n"
    )

    assert _resolve_s3_inputs(
        s3_uri="s3://marie/gen5_extract/single.tif",
        s3_uri_manifest=None,
    ) == ["s3://marie/gen5_extract/single.tif"]
    assert _resolve_s3_inputs(
        s3_uri=None,
        s3_uri_manifest=str(manifest),
    ) == [
        "s3://marie/gen5_extract/a.tif",
        "s3://marie/gen5_extract/b.tif",
    ]


def test_build_input_assets_supports_existing_s3_mode(tmp_path: Path) -> None:
    asset_path = tmp_path / "sample.tif"
    asset_path.write_text("x")

    assets = _build_input_assets(
        input_glob=None,
        input_dir=None,
        input_manifest=None,
        s3_uri="s3://marie/gen5_extract/sample.tif",
        s3_uri_manifest=None,
    )
    local_assets = _build_input_assets(
        input_glob=str(asset_path),
        input_dir=None,
        input_manifest=None,
        s3_uri=None,
        s3_uri_manifest=None,
    )

    assert len(assets) == 1
    assert assets[0].existing_s3_uri == "s3://marie/gen5_extract/sample.tif"
    assert assets[0].local_path is None
    assert assets[0].source_name == "sample.tif"

    assert len(local_assets) == 1
    assert local_assets[0].local_path == asset_path.resolve()
    assert local_assets[0].existing_s3_uri is None


def test_job_run_latency_properties() -> None:
    run = JobRun(
        request_id="job-1",
        job_index=0,
        source_path="/tmp/sample.tif",
        source_name="sample.tif",
        input_mode="existing_s3",
        ref_id="job-1-sample.tif",
        ref_type="gen5_extract",
        s3_uri="s3://marie/gen5_extract/job-1-sample/job-1-sample.tif",
        planner="extract",
        job_name="gen5_extract",
        fault_profile="normal",
    )
    run.submit_started_at = 10.0
    run.submit_finished_at = 12.0
    run.scheduled_at = 13.0
    run.started_at = 15.5
    run.completed_at = 19.0

    assert run.scheduling_ms == 1000.0
    assert run.queue_wait_ms == 3500.0
    assert run.execution_ms == 3500.0
    assert run.end_to_end_ms == 9000.0
    assert run.terminal_status == "completed"


def test_build_debug_snapshot_extracts_scheduler_fields() -> None:
    snapshot = _build_debug_snapshot(
        stage="periodic",
        status_code=200,
        payload={
            "scheduler_info": {
                "running": True,
                "paused": False,
                "active_dags_count": 7,
                "max_concurrent_dags": 32,
            },
            "counters": {
                "fetch_counter": 11,
                "pending_requests": 2,
            },
            "queues": {
                "request_queue_size": 5,
                "event_queue_size": 1,
            },
            "queue_status": {
                "queue_size": 5,
                "workers": {"total": 10, "active": 4, "utilization": "40.0%"},
            },
            "llm_dispatch": {
                "contract_version": "v2",
                "registered_dispatchers": 2,
                "running_dispatchers": 1,
            },
        },
    )

    assert snapshot.ok is True
    assert snapshot.scheduler_running is True
    assert snapshot.scheduler_paused is False
    assert snapshot.active_dags_count == 7
    assert snapshot.max_concurrent_dags == 32
    assert snapshot.fetch_counter == 11
    assert snapshot.pending_requests == 2
    assert snapshot.request_queue_size == 5
    assert snapshot.event_queue_size == 1
    assert snapshot.queue_status == {
        "queue_size": 5,
        "workers": {"total": 10, "active": 4, "utilization": "40.0%"},
    }
    assert snapshot.llm_dispatch_registered_dispatchers == 2
    assert snapshot.llm_dispatch_running_dispatchers == 1


@pytest.mark.asyncio
async def test_capture_debug_snapshot_records_gateway_debug_state() -> None:
    stresser = GatewayE2EStresser(
        gateway_host="localhost",
        gateway_port=52000,
        http_port=51000,
        protocol="grpc",
        endpoint="/api/v1/invoke",
        api_key="system:gateway",
        queue_name="extract",
        planner="extract",
        input_assets=[
            InputAsset(
                source_name="sample.tif",
                source_path="s3://marie/sample.tif",
                existing_s3_uri="s3://marie/sample.tif",
            )
        ],
        job_count=1,
        submit_concurrency=1,
        submit_rate=1.0,
        timeout=10.0,
        terminal_timeout=10.0,
        s3_config=None,
        queue_config=None,
        metadata_template=None,
        template_job_name=None,
        fault_profile="normal",
        debug_sample_interval=5.0,
    )
    stresser._http_session = _FakeDebugSession(
        _FakeDebugResponse(
            200,
            {
                "status": "OK",
                "result": {
                    "scheduler_info": {
                        "running": True,
                        "paused": False,
                        "active_dags_count": 3,
                        "max_concurrent_dags": 16,
                    },
                    "counters": {"fetch_counter": 9, "pending_requests": 1},
                    "queues": {"request_queue_size": 2, "event_queue_size": 0},
                    "queue_status": {"queue_size": 2},
                    "llm_dispatch": {
                        "registered_dispatchers": 1,
                        "running_dispatchers": 1,
                    },
                },
            },
        )
    )

    await stresser._capture_debug_snapshot("start")

    assert len(stresser._debug_samples) == 1
    snapshot = stresser._debug_samples[0]
    assert snapshot.stage == "start"
    assert snapshot.ok is True
    assert snapshot.active_dags_count == 3
    assert snapshot.fetch_counter == 9
    assert snapshot.request_queue_size == 2
    assert snapshot.queue_status == {"queue_size": 2}
    assert snapshot.llm_dispatch_registered_dispatchers == 1
    assert snapshot.llm_dispatch_running_dispatchers == 1
    assert stresser._http_session.last_url == "http://localhost:51000/api/debug"


def test_write_json_report_includes_debug_samples(tmp_path: Path) -> None:
    stresser = GatewayE2EStresser(
        gateway_host="localhost",
        gateway_port=51000,
        http_port=51000,
        protocol="http",
        endpoint="/api/v1/invoke",
        api_key="system:gateway",
        queue_name="extract",
        planner="extract",
        input_assets=[
            InputAsset(
                source_name="sample.tif",
                source_path="s3://marie/sample.tif",
                existing_s3_uri="s3://marie/sample.tif",
            )
        ],
        job_count=1,
        submit_concurrency=1,
        submit_rate=1.0,
        timeout=10.0,
        terminal_timeout=10.0,
        s3_config=None,
        queue_config=None,
        metadata_template=None,
        template_job_name=None,
        fault_profile="normal",
        debug_sample_interval=5.0,
    )
    stresser._debug_samples.append(
        _build_debug_snapshot(
            stage="end",
            status_code=200,
            payload={
                "scheduler_info": {
                    "running": True,
                    "active_dags_count": 4,
                    "max_concurrent_dags": 16,
                },
                "counters": {"fetch_counter": 12},
                "queues": {"request_queue_size": 0, "event_queue_size": 0},
                "queue_status": {"queue_size": 0},
                "llm_dispatch": {
                    "registered_dispatchers": 1,
                    "running_dispatchers": 1,
                },
            },
        )
    )

    output_path = tmp_path / "gateway-e2e-report.json"
    stresser.write_json_report(str(output_path))
    payload = json.loads(output_path.read_text())

    assert payload["summary"]["debug_sample_count"] == 1
    assert payload["debug_sampling"]["enabled"] is True
    assert payload["debug_sampling"]["sample_interval_seconds"] == 5.0
    assert payload["debug_sampling"]["samples"][0]["stage"] == "end"
    assert payload["debug_sampling"]["samples"][0]["active_dags_count"] == 4
    assert payload["debug_sampling"]["samples"][0]["fetch_counter"] == 12
    assert payload["debug_sampling"]["samples"][0]["llm_dispatch_running_dispatchers"] == 1


def test_coerce_gateway_debug_payload_unwraps_gateway_result() -> None:
    payload = _coerce_gateway_debug_payload(
        {
            "status": "OK",
            "result": {"scheduler_info": {"running": True}},
        }
    )

    assert payload == {"scheduler_info": {"running": True}}
