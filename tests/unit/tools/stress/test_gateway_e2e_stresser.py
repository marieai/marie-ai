from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.stress.gateway_e2e_reporting import resolve_report_format
from tools.stress.gateway_e2e_stresser import (
    REDACTED_SECRET,
    GatewayE2EStresser,
    InputAsset,
    JobRun,
    SubmitResult,
    _build_debug_snapshot,
    _build_input_assets,
    _coerce_gateway_debug_payload,
    _extract_failure_error,
    _extract_ref_id_from_event,
    _extract_template,
    _parse_duration_seconds,
    _resolve_inputs,
    _resolve_runtime_config,
    _resolve_s3_inputs,
)

VALID_FAKE_API_KEY = "mau_" + ("A" * 54)


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


def test_mock_annotator_llm_template_builds_upload_shaped_dry_run(
    tmp_path: Path,
) -> None:
    template_payload = json.loads(
        Path("tools/stress/mock_annotator_llm.invoke.json").read_text()
    )
    template_job_name, metadata_template = _extract_template(template_payload)
    source_path = tmp_path / "sample.tif"
    source_path.write_text("fake image bytes")

    stresser = GatewayE2EStresser(
        gateway_host="localhost",
        gateway_port=51000,
        http_port=51000,
        protocol="http",
        endpoint="/api/v1/invoke",
        api_key=VALID_FAKE_API_KEY,
        queue_name=template_job_name or "gen5_extract",
        planner="mock_annotator_llm",
        input_assets=[
            InputAsset(
                source_name="sample.tif",
                source_path=str(source_path),
                local_path=source_path,
            )
        ],
        job_count=1,
        run_time_seconds=None,
        submit_concurrency=1,
        submit_rate=1.0,
        timeout=10.0,
        terminal_timeout=10.0,
        s3_config=None,
        queue_config=None,
        metadata_template=metadata_template,
        template_job_name=template_job_name,
        fault_profile="normal",
        ref_type="stress",
    )

    plan = stresser.build_dry_run_plan()
    submission = plan["submissions"][0]
    invoke_action = submission["request_payload"]["parameters"]["invoke_action"]

    assert template_job_name == "gen5_extract"
    assert plan["planner"] == "mock_annotator_llm"
    assert plan["job_name"] == "gen5_extract"
    assert submission["input_mode"] == "upload"
    assert submission["upload_planned"] is True
    assert submission["source_path"] == str(source_path)
    assert invoke_action["name"] == "gen5_extract"
    assert invoke_action["api_key"] == REDACTED_SECRET
    assert invoke_action["metadata"]["planner"] == "mock_annotator_llm"
    assert invoke_action["metadata"]["ref_id"] == "sample.tif"
    assert invoke_action["metadata"]["ref_type"] == "stress"
    assert invoke_action["metadata"]["uri"].startswith("s3://")
    assert invoke_action["metadata"]["uri"].endswith("sample.tif")
    assert "/sample/" in invoke_action["metadata"]["uri"]
    assert invoke_action["metadata"]["features"] == [
        {
            "type": "pipeline",
            "name": "stress-purge-annotators",
            "purge_annotators": ["mock-llm"],
        }
    ]
    assert VALID_FAKE_API_KEY not in json.dumps(plan)


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


def test_resolve_runtime_config_rejects_explicit_empty_api_key(tmp_path: Path) -> None:
    config = tmp_path / "gateway-e2e.json"
    config.write_text(
        json.dumps(
            {
                "api_base_url": "http://127.0.0.1:51000",
                "api_key": VALID_FAKE_API_KEY,
            }
        )
    )

    args = SimpleNamespace(
        config=str(config),
        protocol=None,
        gateway_host=None,
        gateway_port=None,
        http_port=None,
        api_key="",
    )

    with pytest.raises(ValueError, match="API key is required"):
        _resolve_runtime_config(args)


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


def test_parse_duration_seconds_supports_suffixes() -> None:
    assert _parse_duration_seconds("30s") == 30.0
    assert _parse_duration_seconds("2m") == 120.0
    assert _parse_duration_seconds("1h") == 3600.0
    assert _parse_duration_seconds("1.5m") == 90.0


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


def test_build_dry_run_plan_includes_resolved_payload_for_local_input(
    tmp_path: Path,
) -> None:
    asset_path = tmp_path / "sample.tif"
    asset_path.write_text("x")
    companion_meta = tmp_path / "sample.tif.meta.json"
    companion_meta.write_text(json.dumps({"doc_id": "sample"}))

    stresser = GatewayE2EStresser(
        gateway_host="localhost",
        gateway_port=51000,
        http_port=51000,
        protocol="http",
        endpoint="/api/v1/invoke",
        api_key=VALID_FAKE_API_KEY,
        queue_name="extract",
        planner="extract",
        input_assets=[
            InputAsset(
                source_name=asset_path.name,
                source_path=str(asset_path),
                local_path=asset_path.resolve(),
            )
        ],
        job_count=1,
        run_time_seconds=None,
        submit_concurrency=1,
        submit_rate=1.0,
        timeout=10.0,
        terminal_timeout=10.0,
        s3_config={"S3_STORAGE_BUCKET_NAME": "stress-bucket"},
        queue_config=None,
        metadata_template=None,
        template_job_name=None,
        fault_profile="normal",
        upload_companion_meta=True,
    )

    plan = stresser.build_dry_run_plan()

    assert plan["dry_run"] is True
    assert plan["job_count"] == 1
    assert len(plan["submissions"]) == 1
    submission = plan["submissions"][0]
    assert submission["input_mode"] == "upload"
    assert submission["upload_planned"] is True
    assert submission["upload_companion_meta_planned"] is True
    assert submission["s3_uri"].startswith("s3://stress-bucket/extract/")
    assert submission["metadata"]["uri"] == submission["s3_uri"]
    assert (
        submission["request_payload"]["parameters"]["invoke_action"]["metadata"]["uri"]
        == submission["s3_uri"]
    )
    assert submission["transport"]["url"] == "http://localhost:51000/api/v1/invoke"


def test_upload_to_s3_preserves_source_ref_id_for_companion_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asset_path = tmp_path / "sample.tif"
    asset_path.write_text("image")
    companion_meta = tmp_path / "sample.tif.meta.json"
    companion_meta.write_text(json.dumps({"doc_id": "sample"}))

    writes: list[tuple[str, str, dict]] = []

    def fake_write(source: str, destination: str, **kwargs: object) -> bool:
        payload = {}
        if source.endswith(".stress.meta.json"):
            payload = json.loads(Path(source).read_text())
        writes.append((source, destination, payload))
        return True

    monkeypatch.setattr(
        "tools.stress.gateway_e2e_stresser.StorageManager.write",
        fake_write,
    )

    stresser = GatewayE2EStresser(
        gateway_host="localhost",
        gateway_port=51000,
        http_port=51000,
        protocol="http",
        endpoint="/api/v1/invoke",
        api_key=VALID_FAKE_API_KEY,
        queue_name="gen5_extract",
        planner="mock_annotator_llm",
        input_assets=[
            InputAsset(
                source_name=asset_path.name,
                source_path=str(asset_path),
                local_path=asset_path,
            )
        ],
        job_count=1,
        run_time_seconds=None,
        submit_concurrency=1,
        submit_rate=1.0,
        timeout=10.0,
        terminal_timeout=10.0,
        s3_config=None,
        queue_config=None,
        metadata_template=None,
        template_job_name=None,
        fault_profile="normal",
        ref_type="stress",
    )

    run = stresser._build_run(stresser.input_assets[0], 0)
    stresser._upload_to_s3(run)

    assert run.ref_id == "sample.tif"
    assert writes[0][1].endswith("/stress/sample/sample.tif")
    assert writes[1][1].endswith("/stress/sample/sample.tif.meta.json")
    assert writes[1][2]["ref_id"] == "sample.tif"
    assert writes[1][2]["uri"] == writes[0][1]


def test_build_dry_run_plan_previews_duration_mode(tmp_path: Path) -> None:
    asset_path = tmp_path / "sample.tif"
    asset_path.write_text("x")

    stresser = GatewayE2EStresser(
        gateway_host="localhost",
        gateway_port=51000,
        http_port=51000,
        protocol="http",
        endpoint="/api/v1/invoke",
        api_key=VALID_FAKE_API_KEY,
        queue_name="extract",
        planner="extract",
        input_assets=[
            InputAsset(
                source_name=asset_path.name,
                source_path=str(asset_path),
                local_path=asset_path.resolve(),
            )
        ],
        job_count=None,
        run_time_seconds=60.0,
        submit_concurrency=1,
        submit_rate=10.0,
        timeout=10.0,
        terminal_timeout=10.0,
        s3_config={"S3_STORAGE_BUCKET_NAME": "stress-bucket"},
        queue_config=None,
        metadata_template=None,
        template_job_name=None,
        fault_profile="normal",
        upload_companion_meta=False,
        dry_run_preview_count=2,
    )

    plan = stresser.build_dry_run_plan()

    assert plan["run_mode"] == "duration"
    assert plan["job_count"] is None
    assert plan["run_time_seconds"] == 60.0
    assert plan["estimated_job_count"] == 600
    assert plan["preview_job_count"] == 2
    assert len(plan["submissions"]) == 2


def test_live_status_payload_and_report_file(tmp_path: Path) -> None:
    asset_path = tmp_path / "sample.tif"
    asset_path.write_text("x")
    live_report_path = tmp_path / "live-status.json"

    stresser = GatewayE2EStresser(
        gateway_host="localhost",
        gateway_port=51000,
        http_port=51000,
        protocol="http",
        endpoint="/api/v1/invoke",
        api_key=VALID_FAKE_API_KEY,
        queue_name="extract",
        planner="extract",
        input_assets=[
            InputAsset(
                source_name=asset_path.name,
                source_path=str(asset_path),
                local_path=asset_path.resolve(),
            )
        ],
        job_count=5,
        run_time_seconds=None,
        submit_concurrency=1,
        submit_rate=2.0,
        timeout=10.0,
        terminal_timeout=10.0,
        s3_config={"S3_STORAGE_BUCKET_NAME": "stress-bucket"},
        queue_config=None,
        metadata_template=None,
        template_job_name=None,
        fault_profile="normal",
        soft_sla_seconds=15.0,
        hard_sla_seconds=45.0,
        min_soft_sla_compliance_pct=95.0,
        min_hard_sla_compliance_pct=99.0,
        upload_companion_meta=False,
        progress_interval=2.5,
        live_report_path=str(live_report_path),
        live_report_format="json",
    )

    stresser.metrics.start_time = 100.0
    run = stresser._build_run(stresser.input_assets[0], 0)
    run.soft_sla_at = 116.0
    run.hard_sla_at = 146.0
    run.soft_sla_offset_seconds = 15.0
    run.hard_sla_offset_seconds = 45.0
    run.job_id = "job-a"
    run.submit_started_at = 101.0
    run.submit_finished_at = 102.0
    run.completed_at = 104.0
    stresser._register_run(run)

    payload = stresser._build_live_status_payload("running")
    stresser._write_live_report_payload(payload)
    written = json.loads(live_report_path.read_text())

    assert payload["status"] == "running"
    assert payload["counts"]["created_jobs"] == 1
    assert payload["counts"]["submitted_jobs"] == 1
    assert payload["counts"]["completed_jobs"] == 1
    assert payload["progress_interval_seconds"] == 2.5
    assert payload["live_report_format"] == "json"
    assert payload["debug_sampling"]["enabled"] is False
    assert payload["debug_sampling"]["interval_seconds"] == 0.0
    assert payload["run_health"]["open_jobs"] == 0
    assert payload["run_health"]["inflight_jobs"] == 0
    assert payload["throughput_jobs_per_second"]["completed"] is not None
    assert payload["latency_stats_ms"]["end_to_end"] is not None
    assert payload["sla"]["soft"]["configured_seconds"] == 15.0
    assert payload["sla"]["soft"]["configured_jobs"] == 1
    assert payload["sla"]["soft"]["met_jobs"] == 1
    assert payload["sla"]["soft"]["terminal_compliance_pct"] == 100.0
    assert payload["sla"]["hard"]["configured_seconds"] == 45.0
    assert payload["recent_jobs"][0]["job_id"] == "job-a"
    assert written["counts"]["completed_jobs"] == 1
    assert written["sla"]["soft"]["met_jobs"] == 1
    assert written["recent_jobs"][0]["request_id"] == run.request_id


def test_live_report_supports_html_output(tmp_path: Path) -> None:
    asset_path = tmp_path / "sample.tif"
    asset_path.write_text("x")
    live_report_path = tmp_path / "live-status.html"

    stresser = GatewayE2EStresser(
        gateway_host="localhost",
        gateway_port=51000,
        http_port=51000,
        protocol="http",
        endpoint="/api/v1/invoke",
        api_key=VALID_FAKE_API_KEY,
        queue_name="extract",
        planner="extract",
        input_assets=[
            InputAsset(
                source_name=asset_path.name,
                source_path=str(asset_path),
                local_path=asset_path.resolve(),
            )
        ],
        job_count=5,
        run_time_seconds=None,
        submit_concurrency=1,
        submit_rate=2.0,
        timeout=10.0,
        terminal_timeout=10.0,
        s3_config={"S3_STORAGE_BUCKET_NAME": "stress-bucket"},
        queue_config=None,
        metadata_template=None,
        template_job_name=None,
        fault_profile="normal",
        soft_sla_seconds=15.0,
        hard_sla_seconds=45.0,
        upload_companion_meta=False,
        progress_interval=2.5,
        live_report_path=str(live_report_path),
        live_report_format="html",
    )

    stresser.metrics.start_time = 100.0
    run = stresser._build_run(stresser.input_assets[0], 0)
    run.soft_sla_at = 116.0
    run.hard_sla_at = 146.0
    run.soft_sla_offset_seconds = 15.0
    run.hard_sla_offset_seconds = 45.0
    run.job_id = "job-a"
    run.submit_started_at = 101.0
    run.submit_finished_at = 102.0
    run.completed_at = 104.0
    stresser._register_run(run)

    payload = stresser._build_live_status_payload("running")
    stresser._write_live_report_payload(payload)
    written = live_report_path.read_text()

    assert "Gateway E2E Live Report" in written
    assert (
        "Auto-refreshing aggregate view of throughput, backlog, and latency." in written
    )
    assert 'http-equiv="refresh"' in written
    assert "Run Health" in written
    assert "Throughput and Flow" in written
    assert "Live SLA Status" in written
    assert "Observed Latency" in written
    assert "Debug sampling disabled." in written
    assert "No recent failures." in written


def test_write_report_supports_html_output(tmp_path: Path) -> None:
    asset_path = tmp_path / "sample.tif"
    asset_path.write_text("x")
    report_path = tmp_path / "final-report.html"

    stresser = GatewayE2EStresser(
        gateway_host="localhost",
        gateway_port=51000,
        http_port=51000,
        protocol="http",
        endpoint="/api/v1/invoke",
        api_key=VALID_FAKE_API_KEY,
        queue_name="extract",
        planner="extract",
        input_assets=[
            InputAsset(
                source_name=asset_path.name,
                source_path=str(asset_path),
                local_path=asset_path.resolve(),
            )
        ],
        job_count=1,
        run_time_seconds=None,
        submit_concurrency=1,
        submit_rate=1.0,
        timeout=10.0,
        terminal_timeout=10.0,
        s3_config={"S3_STORAGE_BUCKET_NAME": "stress-bucket"},
        queue_config=None,
        metadata_template=None,
        template_job_name=None,
        fault_profile="normal",
        soft_sla_seconds=15.0,
        hard_sla_seconds=45.0,
        upload_companion_meta=False,
    )

    stresser.metrics.start_time = 100.0
    stresser.metrics.end_time = 104.0
    run = stresser._build_run(stresser.input_assets[0], 0)
    run.job_id = "job-a"
    run.submit_started_at = 100.5
    run.submit_finished_at = 101.0
    run.scheduled_at = 101.2
    run.started_at = 101.5
    run.completed_at = 104.0
    run.soft_sla_at = 115.0
    run.hard_sla_at = 145.0
    stresser._register_run(run)
    stresser._finalize_metrics()
    payload = stresser.build_report_payload()

    stresser.write_report(str(report_path), "html")
    written = report_path.read_text()

    assert payload["sla"]["soft"]["met_jobs"] == 1
    assert payload["sla"]["hard"]["met_jobs"] == 1
    assert "Gateway E2E Stress Report" in written
    assert "Latency Breakdown" in written
    assert "SLA Outcome" in written
    assert "Worst SLA Misses" in written
    assert "job-a" in written
    assert "Recent Jobs" in written


def test_resolve_report_format_defaults_from_extension() -> None:
    assert resolve_report_format("/tmp/report.html", "auto") == "html"
    assert resolve_report_format("/tmp/report.json", "auto") == "json"
    assert resolve_report_format("/tmp/report", "auto") == "json"


def test_log_submit_summary_includes_source_and_s3_uri(
    caplog: pytest.LogCaptureFixture,
) -> None:
    stresser = GatewayE2EStresser(
        gateway_host="localhost",
        gateway_port=51000,
        http_port=51000,
        protocol="http",
        endpoint="/api/v1/invoke",
        api_key=VALID_FAKE_API_KEY,
        queue_name="extract",
        planner="extract",
        input_assets=[],
        job_count=1,
        run_time_seconds=None,
        submit_concurrency=1,
        submit_rate=1.0,
        timeout=10.0,
        terminal_timeout=10.0,
        s3_config={"S3_STORAGE_BUCKET_NAME": "stress-bucket"},
        queue_config=None,
        metadata_template=None,
        template_job_name=None,
        fault_profile="normal",
        upload_companion_meta=False,
    )
    run = JobRun(
        request_id="job-1",
        job_index=0,
        source_path="/tmp/sample.tif",
        source_name="sample.tif",
        input_mode="upload",
        ref_id="job-1-sample.tif",
        ref_type="extract",
        s3_uri="s3://stress-bucket/extract/job-1-sample/job-1-sample.tif",
        planner="extract",
        job_name="extract",
        fault_profile="normal",
    )

    with caplog.at_level(logging.INFO, logger="GatewayE2EStresser"):
        stresser._log_submit_summary(
            run,
            SubmitResult(
                request_id=run.request_id,
                success=True,
                latency_ms=12.5,
                job_id="gateway-job-1",
            ),
        )

    assert "Submitted job_index=0 request_id=job-1 job_id=gateway-job-1" in caplog.text
    assert "source=/tmp/sample.tif" in caplog.text
    assert (
        "s3_uri=s3://stress-bucket/extract/job-1-sample/job-1-sample.tif" in caplog.text
    )


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


def test_job_run_clamps_negative_scheduling_and_queue_wait() -> None:
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
    run.submit_finished_at = 12.750
    run.scheduled_at = 12.000
    run.started_at = 12.000

    assert run.scheduling_ms == 0.0
    assert run.queue_wait_ms == 0.0


def test_job_run_sla_properties_mark_deadline_miss() -> None:
    run = JobRun(
        request_id="job-2",
        job_index=1,
        source_path="/tmp/sample.tif",
        source_name="sample.tif",
        input_mode="existing_s3",
        ref_id="job-2-sample.tif",
        ref_type="gen5_extract",
        s3_uri="s3://marie/gen5_extract/job-2-sample/job-2-sample.tif",
        planner="extract",
        job_name="gen5_extract",
        fault_profile="normal",
        soft_sla_at=14.0,
        hard_sla_at=20.0,
    )
    run.submit_started_at = 10.0
    run.submit_finished_at = 12.0
    run.completed_at = 19.0

    assert run.soft_sla_status == "deadline_missed"
    assert run.soft_sla_met is False
    assert run.soft_sla_lateness_ms == 5000.0
    assert run.hard_sla_status == "met"
    assert run.hard_sla_met is True
    assert run.hard_sla_lateness_ms == 0.0


def test_build_metadata_applies_relative_sla_offsets() -> None:
    stresser = GatewayE2EStresser(
        gateway_host="localhost",
        gateway_port=51000,
        http_port=51000,
        protocol="http",
        endpoint="/api/v1/invoke",
        api_key=VALID_FAKE_API_KEY,
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
        run_time_seconds=None,
        submit_concurrency=1,
        submit_rate=1.0,
        timeout=10.0,
        terminal_timeout=10.0,
        s3_config=None,
        queue_config=None,
        metadata_template={"policy": "strict"},
        template_job_name=None,
        fault_profile="normal",
        soft_sla_seconds=30.0,
        hard_sla_seconds=120.0,
    )
    run = stresser._build_run(stresser.input_assets[0], 0)

    metadata = stresser._build_metadata(run, sla_anchor_at=1000.0)

    assert metadata["soft_sla"] == "1970-01-01T00:17:10+00:00"
    assert metadata["hard_sla"] == "1970-01-01T00:18:40+00:00"
    assert run.soft_sla_at == 1030.0
    assert run.hard_sla_at == 1120.0
    assert metadata["policy"] == "strict"


def test_build_metadata_injects_mock_failure_controls() -> None:
    stresser = GatewayE2EStresser(
        gateway_host="localhost",
        gateway_port=51000,
        http_port=51000,
        protocol="http",
        endpoint="/api/v1/invoke",
        api_key=VALID_FAKE_API_KEY,
        queue_name="extract",
        planner="extract",
        input_assets=[
            InputAsset(
                source_name="sample.tif",
                source_path="s3://marie/sample.tif",
                existing_s3_uri="s3://marie/sample.tif",
            )
        ],
        job_count=2,
        run_time_seconds=None,
        submit_concurrency=1,
        submit_rate=1.0,
        timeout=10.0,
        terminal_timeout=10.0,
        s3_config=None,
        queue_config=None,
        metadata_template=None,
        template_job_name=None,
        fault_profile="normal",
        mock_failure_rate=0.25,
        mock_failure_mode="timeout",
        force_failure_every=2,
    )
    first_run = stresser._build_run(stresser.input_assets[0], 0)
    second_run = stresser._build_run(stresser.input_assets[0], 1)

    first_metadata = stresser._build_metadata(first_run, sla_anchor_at=1000.0)
    second_metadata = stresser._build_metadata(second_run, sla_anchor_at=1000.0)

    assert first_metadata["failure_rate"] == 0.25
    assert first_metadata["failure_mode"] == "timeout"
    assert "force_fail" not in first_metadata
    assert first_metadata["stress_failure_simulation"]["force_fail"] is False
    assert first_run.mock_failure_rate == 0.25
    assert first_run.mock_failure_mode == "timeout"
    assert first_run.force_fail is False

    assert second_metadata["failure_rate"] == 0.25
    assert second_metadata["failure_mode"] == "timeout"
    assert second_metadata["force_fail"] is True
    assert second_metadata["stress_failure_simulation"] == {
        "source": "gateway_e2e_stresser",
        "mock_failure_rate": 0.25,
        "mock_failure_mode": "timeout",
        "force_failure_every": 2,
        "force_fail": True,
    }
    assert second_run.force_fail is True


def test_build_metadata_injects_fixed_llm_pool_controls() -> None:
    stresser = GatewayE2EStresser(
        gateway_host="localhost",
        gateway_port=51000,
        http_port=51000,
        protocol="http",
        endpoint="/api/v1/invoke",
        api_key=VALID_FAKE_API_KEY,
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
        run_time_seconds=None,
        submit_concurrency=1,
        submit_rate=1.0,
        timeout=10.0,
        terminal_timeout=10.0,
        s3_config=None,
        queue_config=None,
        metadata_template={"source": "unit-test"},
        template_job_name=None,
        fault_profile="normal",
        llm_pool_id="document-small",
    )
    run = stresser._build_run(stresser.input_assets[0], 0)

    metadata = stresser._build_metadata(run, sla_anchor_at=1000.0)

    assert run.llm_pool_id == "document-small"
    assert metadata["pool_id"] == "document-small"


def test_build_metadata_cycles_llm_pool_controls() -> None:
    stresser = GatewayE2EStresser(
        gateway_host="localhost",
        gateway_port=51000,
        http_port=51000,
        protocol="http",
        endpoint="/api/v1/invoke",
        api_key=VALID_FAKE_API_KEY,
        queue_name="extract",
        planner="extract",
        input_assets=[
            InputAsset(
                source_name="sample.tif",
                source_path="s3://marie/sample.tif",
                existing_s3_uri="s3://marie/sample.tif",
            )
        ],
        job_count=3,
        run_time_seconds=None,
        submit_concurrency=1,
        submit_rate=1.0,
        timeout=10.0,
        terminal_timeout=10.0,
        s3_config=None,
        queue_config=None,
        metadata_template=None,
        template_job_name=None,
        fault_profile="normal",
        llm_pool_cycle=["document-small", "document-medium"],
    )
    runs = [stresser._build_run(stresser.input_assets[0], index) for index in range(3)]

    metadata = [stresser._build_metadata(run, sla_anchor_at=1000.0) for run in runs]

    assert [run.llm_pool_id for run in runs] == [
        "document-small",
        "document-medium",
        "document-small",
    ]
    assert [item["pool_id"] for item in metadata] == [
        "document-small",
        "document-medium",
        "document-small",
    ]


def test_build_metadata_injects_purge_annotators_feature_for_mock_llm() -> None:
    stresser = GatewayE2EStresser(
        gateway_host="localhost",
        gateway_port=51000,
        http_port=51000,
        protocol="http",
        endpoint="/api/v1/invoke",
        api_key=VALID_FAKE_API_KEY,
        queue_name="gen5_extract",
        planner="mock_annotator_llm",
        input_assets=[
            InputAsset(
                source_name="sample.tif",
                source_path="s3://marie/sample.tif",
                existing_s3_uri="s3://marie/sample.tif",
            )
        ],
        job_count=1,
        run_time_seconds=None,
        submit_concurrency=1,
        submit_rate=1.0,
        timeout=10.0,
        terminal_timeout=10.0,
        s3_config=None,
        queue_config=None,
        metadata_template=None,
        template_job_name=None,
        fault_profile="normal",
        purge_annotators=["mock-llm"],
    )
    run = stresser._build_run(stresser.input_assets[0], 0)

    metadata = stresser._build_metadata(run, sla_anchor_at=1000.0)

    assert metadata["features"] == [
        {
            "type": "pipeline",
            "name": "stress-purge-annotators",
            "purge_annotators": ["mock-llm"],
        }
    ]


def test_build_dry_run_plan_previews_failure_simulation(tmp_path: Path) -> None:
    asset_path = tmp_path / "sample.tif"
    asset_path.write_text("x")

    stresser = GatewayE2EStresser(
        gateway_host="localhost",
        gateway_port=51000,
        http_port=51000,
        protocol="http",
        endpoint="/api/v1/invoke",
        api_key=VALID_FAKE_API_KEY,
        queue_name="extract",
        planner="extract",
        input_assets=[
            InputAsset(
                source_name=asset_path.name,
                source_path=str(asset_path),
                local_path=asset_path.resolve(),
            )
        ],
        job_count=2,
        run_time_seconds=None,
        submit_concurrency=1,
        submit_rate=1.0,
        timeout=10.0,
        terminal_timeout=10.0,
        s3_config={"S3_STORAGE_BUCKET_NAME": "stress-bucket"},
        queue_config=None,
        metadata_template=None,
        template_job_name=None,
        fault_profile="normal",
        mock_failure_mode="exception",
        force_failure_every=2,
        upload_companion_meta=False,
    )

    plan = stresser.build_dry_run_plan()
    forced_submission = plan["submissions"][1]

    assert plan["mock_failure_rate"] is None
    assert plan["mock_failure_mode"] == "exception"
    assert plan["force_failure_every"] == 2
    assert forced_submission["force_fail"] is True
    assert forced_submission["metadata"]["force_fail"] is True
    assert (
        forced_submission["request_payload"]["parameters"]["invoke_action"]["metadata"][
            "force_fail"
        ]
        is True
    )


def test_build_metadata_applies_incremental_sla_offsets_with_cycle() -> None:
    stresser = GatewayE2EStresser(
        gateway_host="localhost",
        gateway_port=51000,
        http_port=51000,
        protocol="http",
        endpoint="/api/v1/invoke",
        api_key=VALID_FAKE_API_KEY,
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
        run_time_seconds=None,
        submit_concurrency=1,
        submit_rate=1.0,
        timeout=10.0,
        terminal_timeout=10.0,
        s3_config=None,
        queue_config=None,
        metadata_template=None,
        template_job_name=None,
        fault_profile="normal",
        soft_sla_seconds=20.0,
        hard_sla_seconds=60.0,
        soft_sla_step_seconds=5.0,
        hard_sla_step_seconds=10.0,
        sla_step_every_jobs=2,
        sla_step_cycle=3,
    )
    run_bucket_0 = stresser._build_run(stresser.input_assets[0], 0)
    run_bucket_2 = stresser._build_run(stresser.input_assets[0], 4)
    run_wrapped = stresser._build_run(stresser.input_assets[0], 6)

    metadata_0 = stresser._build_metadata(run_bucket_0, sla_anchor_at=1000.0)
    metadata_2 = stresser._build_metadata(run_bucket_2, sla_anchor_at=1000.0)
    metadata_wrapped = stresser._build_metadata(run_wrapped, sla_anchor_at=1000.0)

    assert run_bucket_0.sla_bucket_index == 0
    assert run_bucket_0.soft_sla_offset_seconds == 20.0
    assert run_bucket_0.hard_sla_offset_seconds == 60.0
    assert metadata_0["soft_sla"] == "1970-01-01T00:17:00+00:00"
    assert metadata_0["hard_sla"] == "1970-01-01T00:17:40+00:00"

    assert run_bucket_2.sla_bucket_index == 2
    assert run_bucket_2.soft_sla_offset_seconds == 30.0
    assert run_bucket_2.hard_sla_offset_seconds == 80.0
    assert metadata_2["soft_sla"] == "1970-01-01T00:17:10+00:00"
    assert metadata_2["hard_sla"] == "1970-01-01T00:18:00+00:00"

    assert run_wrapped.sla_bucket_index == 0
    assert run_wrapped.soft_sla_offset_seconds == 20.0
    assert run_wrapped.hard_sla_offset_seconds == 60.0
    assert metadata_wrapped["soft_sla"] == "1970-01-01T00:17:00+00:00"
    assert metadata_wrapped["hard_sla"] == "1970-01-01T00:17:40+00:00"


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
        api_key=VALID_FAKE_API_KEY,
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
        run_time_seconds=None,
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
        api_key=VALID_FAKE_API_KEY,
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
        run_time_seconds=None,
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
    assert (
        payload["debug_sampling"]["samples"][0]["llm_dispatch_running_dispatchers"] == 1
    )


def test_finalize_metrics_tracks_sla_compliance_and_verification() -> None:
    stresser = GatewayE2EStresser(
        gateway_host="localhost",
        gateway_port=51000,
        http_port=51000,
        protocol="http",
        endpoint="/api/v1/invoke",
        api_key=VALID_FAKE_API_KEY,
        queue_name="extract",
        planner="extract",
        input_assets=[
            InputAsset(
                source_name="sample.tif",
                source_path="s3://marie/sample.tif",
                existing_s3_uri="s3://marie/sample.tif",
            )
        ],
        job_count=2,
        run_time_seconds=None,
        submit_concurrency=1,
        submit_rate=1.0,
        timeout=10.0,
        terminal_timeout=10.0,
        s3_config=None,
        queue_config=None,
        metadata_template=None,
        template_job_name=None,
        fault_profile="normal",
        min_soft_sla_compliance_pct=100.0,
        min_hard_sla_compliance_pct=100.0,
    )
    met_run = stresser._build_run(stresser.input_assets[0], 0)
    met_run.job_id = "job-a"
    met_run.submit_started_at = 10.0
    met_run.submit_finished_at = 11.0
    met_run.completed_at = 18.0
    met_run.soft_sla_at = 20.0
    met_run.hard_sla_at = 30.0

    missed_run = stresser._build_run(stresser.input_assets[0], 1)
    missed_run.job_id = "job-b"
    missed_run.submit_started_at = 10.0
    missed_run.submit_finished_at = 11.0
    missed_run.completed_at = 28.0
    missed_run.soft_sla_at = 20.0
    missed_run.hard_sla_at = 25.0

    stresser._register_run(met_run)
    stresser._register_run(missed_run)
    stresser._finalize_metrics()

    assert stresser.metrics.soft_sla_configured_jobs == 2
    assert stresser.metrics.soft_sla_met_jobs == 1
    assert stresser.metrics.soft_sla_missed_jobs == 1
    assert stresser.metrics.soft_sla_compliance_pct == 50.0
    assert stresser.metrics.hard_sla_configured_jobs == 2
    assert stresser.metrics.hard_sla_met_jobs == 1
    assert stresser.metrics.hard_sla_missed_jobs == 1
    assert stresser.metrics.hard_sla_compliance_pct == 50.0
    assert len(stresser.verification_errors) == 2
    assert "Soft SLA compliance 50.00%" in stresser.verification_errors[0]
    assert "Hard SLA compliance 50.00%" in stresser.verification_errors[1]


def test_coerce_gateway_debug_payload_unwraps_gateway_result() -> None:
    payload = _coerce_gateway_debug_payload(
        {
            "status": "OK",
            "result": {"scheduler_info": {"running": True}},
        }
    )

    assert payload == {"scheduler_info": {"running": True}}
