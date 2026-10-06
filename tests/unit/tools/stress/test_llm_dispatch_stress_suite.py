from __future__ import annotations

import json
import subprocess
from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from marie.extract.readers.meta_reader.meta_reader import MetaReader
from tools.stress.gateway_e2e_stresser import document_page_count_from_uri
from tools.stress.llm_dispatch_stress_suite import (
    Phase,
    _pool_latency_breakdown,
    _provider_metrics,
    _result_row,
    _validate_provider_capacity,
    _write_run_index,
    build_stresser_command,
    default_phases,
    expected_pool_counts,
    parse_args,
    preflight_dispatch_runtime,
    run,
    validate_clean_dispatch_runtime,
    validate_fixture_manifest,
    validate_report,
    write_fixtures,
    write_html_report,
)


def _runtime_snapshot(*, reserved_items: int = 0) -> dict[str, object]:
    return {
        "observation": {"stale": False},
        "runtime_summary": {
            "registered_dispatchers": 1,
            "running_dispatchers": 1,
            "pending_request_count": 0,
            "inflight_request_count": reserved_items,
        },
        "pools": [
            {
                "pool_id": "document-small",
                "request_queue_depth": 0,
                "reserved_items": reserved_items,
                "state_counts": {
                    "ready": 0,
                    "claimed": 0,
                    "running": 0,
                    "unknown": 0,
                },
            }
        ],
        "endpoint_groups": [
            {
                "group_id": "local-mock",
                "replicas": [
                    {
                        "replica_id": "aimock-local",
                        "enabled": True,
                        "execution_limit": 4,
                        "circuit": "closed" if reserved_items == 0 else "open",
                        "reserved_items": reserved_items,
                        "reserved_bytes": reserved_items * 1024,
                    }
                ],
            }
        ],
    }


def test_dispatch_preflight_accepts_an_idle_healthy_runtime() -> None:
    summary = validate_clean_dispatch_runtime(_runtime_snapshot())

    assert summary == {
        "registered_dispatchers": 1,
        "running_dispatchers": 1,
        "pending_requests": 0,
        "inflight_requests": 0,
        "endpoint_replicas": 1,
        "dispatch_execution_limit": 4,
    }


def test_dispatch_preflight_rejects_provider_limit_above_dispatch_capacity() -> None:
    with pytest.raises(
        RuntimeError,
        match="dispatch capacity 4 is lower than requested provider concurrency 10",
    ):
        validate_clean_dispatch_runtime(
            _runtime_snapshot(), required_execution_capacity=10
        )


def test_dispatch_preflight_rejects_stale_reservations_and_open_circuit() -> None:
    with pytest.raises(
        RuntimeError,
        match=(
            "dispatch runtime is not clean.*in-flight requests=4.*"
            "document-small reserved items=4.*"
            "aimock-local reserved items=4"
        ),
    ):
        validate_clean_dispatch_runtime(_runtime_snapshot(reserved_items=4))


def test_dispatch_preflight_uses_gateway_config_without_exposing_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    token = "mas_" + "s" * 54
    config = tmp_path / "gateway.json"
    config.write_text(
        json.dumps({"api_base_url": "http://gateway.test:51000", "api_key": token})
    )
    observed: dict[str, object] = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps({"status": "OK", "result": _runtime_snapshot()}).encode()

    def open_request(request, timeout: int):
        observed["url"] = request.full_url
        observed["authorization"] = request.headers["Authorization"]
        observed["timeout"] = timeout
        return Response()

    monkeypatch.setattr(
        "tools.stress.llm_dispatch_stress_suite.urllib.request.urlopen",
        open_request,
    )

    result = preflight_dispatch_runtime(config, "default")

    assert result["pending_requests"] == 0
    assert observed == {
        "url": (
            "http://gateway.test:51000/api/llm-dispatch/runtime?"
            "fabric_group_id=default&limit=10"
        ),
        "authorization": f"Bearer {token}",
        "timeout": 10,
    }
    assert token not in json.dumps(result)


def test_default_phases_preserve_the_qualification_matrix() -> None:
    phases = {phase.name: phase for phase in default_phases()}

    assert list(phases) == [
        "normal",
        "unreachable",
        "retryable",
        "delay",
        "hang",
        "error",
        "max-output",
        "repetition",
        "repetition-terminal",
        "chaos",
    ]
    assert phases["normal"].jobs == 60
    assert phases["unreachable"].fault == {
        "profile": "normal",
        "outageMs": 5000,
    }
    assert phases["unreachable"].retry_evidence == "prewrite_refund"
    assert phases["unreachable"].direct_expected_outcome == "settled"
    assert phases["retryable"].fault == {
        "profile": "transient_error",
        "transientErrors": 3,
    }
    assert phases["retryable"].retry_evidence == "deferred_retry"
    assert phases["retryable"].direct_expected_outcome == "settled"
    assert phases["delay"].fault == {"profile": "timeout", "timeoutMs": 5000}
    assert phases["hang"].fault == {"profile": "timeout", "timeoutMs": 35000}
    assert phases["hang"].expected_outcome == "failed"
    assert phases["hang"].direct_expected_outcome == "completed"
    assert phases["hang"].retry_evidence == "no_retry"
    assert phases["error"].fault == {"profile": "terminal_error"}
    assert phases["error"].expected_outcome == "failed"
    assert phases["max-output"].fault == {"profile": "max_tokens"}
    assert phases["max-output"].expected_outcome == "failed"
    assert phases["max-output"].retry_evidence == "max_output_terminal"
    assert phases["repetition"].fault == {"profile": "repetition"}
    assert phases["repetition"].expected_outcome == "completed"
    assert phases["repetition"].retry_evidence == "repetition_recovery"
    assert phases["repetition-terminal"].fault == {"profile": "persistent_repetition"}
    assert phases["repetition-terminal"].expected_outcome == "failed"
    assert phases["repetition-terminal"].retry_evidence == "repetition_terminal"
    assert phases["chaos"].jobs == 15
    assert phases["chaos"].fault == {
        "profile": "chaos",
        "timeoutMs": 6000,
        "chaosErrorRate": 0.2,
        "chaosTimeoutRate": 0.2,
        "chaosSlowRate": 0.3,
        "chaosSlowMs": 3000,
    }


def test_normal_job_count_is_configurable_and_balanced_for_any_count() -> None:
    phases = {phase.name: phase for phase in default_phases(normal_jobs=1000)}

    assert phases["normal"].jobs == 1000
    assert phases["unreachable"].jobs == 6
    assert phases["retryable"].jobs == 6
    assert phases["delay"].jobs == 6
    assert expected_pool_counts(1000) == {
        "document-small": 334,
        "document-medium": 333,
        "document-large": 333,
    }
    assert phases["normal"].terminal_timeout == 8000
    assert (
        parse_args(
            ["--runtime-mode", "queued-dispatch", "--job-count", "2500"]
        ).job_count
        == 2500
    )


def test_pool_counts_define_the_normal_phase_partition() -> None:
    args = parse_args(
        [
            "--runtime-mode",
            "queued-dispatch",
            "--pool-counts",
            "document-small=80,document-medium=15,document-large=5",
        ]
    )
    phase = default_phases(
        normal_jobs=args.job_count,
        normal_pool_counts=args.pool_counts,
    )[0]

    assert args.job_count == 100
    assert phase.jobs == 100
    assert phase.pool_counts == {
        "document-small": 80,
        "document-medium": 15,
        "document-large": 5,
    }


def test_provider_capacity_is_configurable() -> None:
    defaults = parse_args(["--runtime-mode", "queued-dispatch"])
    configured = parse_args(
        [
            "--runtime-mode",
            "direct-batch",
            "--provider-concurrency",
            "20",
            "--provider-delay-ms",
            "250",
        ]
    )

    assert defaults.provider_concurrency == 4
    assert defaults.provider_delay_ms == 100
    assert configured.provider_concurrency == 20
    assert configured.provider_delay_ms == 250

    with pytest.raises(SystemExit):
        parse_args(["--runtime-mode", "queued-dispatch", "--provider-concurrency", "0"])
    with pytest.raises(SystemExit):
        parse_args(["--runtime-mode", "queued-dispatch", "--provider-delay-ms", "-1"])


def test_provider_metrics_report_observed_capacity() -> None:
    metrics = _provider_metrics(
        {
            "maxConcurrentCalls": 10,
            "processingDelayMs": 125,
            "peakConcurrentCalls": 8,
            "capacityWaitCount": 31,
            "capacityWaitMs": 4200,
            "requestCount": 250,
        },
        duration_seconds=25.0,
    )

    assert metrics == {
        "provider_concurrency_limit": 10,
        "provider_processing_delay_ms": 125,
        "provider_peak_concurrency": 8,
        "provider_waited_calls": 31,
        "provider_wait_ms": 4200,
        "provider_calls": 250,
        "provider_calls_per_second": 10.0,
    }


def test_pool_latency_breakdown_uses_submitted_fixture_pool() -> None:
    report = {
        "jobs": [
            {
                "source_path": "/tmp/document-small-1p.tif",
                "queue_wait_ms": 10.0,
                "execution_ms": 20.0,
                "end_to_end_ms": 35.0,
            },
            {
                "source_path": "/tmp/document-small-1p.tif",
                "queue_wait_ms": 30.0,
                "execution_ms": 40.0,
                "end_to_end_ms": 75.0,
            },
            {
                "source_path": "/tmp/document-large-30p.tif",
                "queue_wait_ms": 50.0,
                "execution_ms": 60.0,
                "end_to_end_ms": 115.0,
            },
        ]
    }

    breakdown = _pool_latency_breakdown(report)

    assert breakdown["document-small"] == {
        "jobs": 2,
        "queue_wait_p50_ms": 20.0,
        "queue_wait_p95_ms": 29.0,
        "execution_p50_ms": 30.0,
        "execution_p95_ms": 39.0,
        "end_to_end_p50_ms": 55.0,
        "end_to_end_p95_ms": 73.0,
    }
    assert breakdown["document-medium"]["jobs"] == 0
    assert breakdown["document-large"]["end_to_end_p95_ms"] == 115.0


def test_queued_provider_capacity_requires_enough_dispatch_slots() -> None:
    state = {
        "maxConcurrentCalls": 10,
        "processingDelayMs": 100,
        "peakConcurrentCalls": 10,
    }
    report = {
        "routing_qualification": {
            "endpoint_groups": [
                {
                    "replicas": [
                        {
                            "enabled": True,
                            "execution_limit": 4,
                        }
                    ]
                }
            ]
        }
    }

    with pytest.raises(
        AssertionError,
        match="queued dispatch capacity is lower than the shared provider limit",
    ):
        _validate_provider_capacity(
            state=state,
            report=report,
            runtime_mode="queued-dispatch",
            concurrency=10,
            processing_delay_ms=100,
        )

    report["routing_qualification"]["endpoint_groups"][0]["replicas"][0][
        "execution_limit"
    ] = 20
    _validate_provider_capacity(
        state=state,
        report=report,
        runtime_mode="queued-dispatch",
        concurrency=10,
        processing_delay_ms=100,
    )


def test_pool_counts_reject_job_count_mismatch() -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--runtime-mode",
                "queued-dispatch",
                "--job-count",
                "99",
                "--pool-counts",
                "document-small=80,document-medium=15,document-large=5",
            ]
        )


def test_normal_terminal_timeout_can_be_overridden() -> None:
    args = parse_args(
        [
            "--runtime-mode",
            "queued-dispatch",
            "--job-count",
            "10000",
            "--normal-terminal-timeout",
            "90000",
        ]
    )
    phase = default_phases(
        normal_jobs=args.job_count,
        normal_terminal_timeout=args.normal_terminal_timeout,
    )[0]

    assert phase.terminal_timeout == 90000


def test_write_fixtures_creates_balanced_page_classes(tmp_path: Path) -> None:
    manifest = write_fixtures(tmp_path)
    sources = [Path(line) for line in manifest.read_text().splitlines()]

    assert [source.name for source in sources] == [
        "document-small-1p.tif",
        "document-medium-8p.tif",
        "document-large-30p.pdf",
    ]
    assert [
        json.loads(Path(f"{source}.meta.json").read_text())["pages"]
        for source in sources
    ] == [
        "1",
        "8",
        "30",
    ]
    assert [
        document_page_count_from_uri(str(source), 100 * 1024 * 1024, 30)
        for source in sources
    ] == [
        1,
        8,
        30,
    ]

    for source, expected_pages in zip(sources, (1, 8, 30), strict=True):
        metadata = json.loads(Path(f"{source}.meta.json").read_text())
        assert metadata["extraction"] == {}
        frames = [
            np.zeros((800, 640, 3), dtype=np.uint8) for _ in range(expected_pages)
        ]
        document = MetaReader.from_data(
            frames=frames,
            ocr_meta=metadata["ocr"],
            unstructured_meta={"source_metadata": metadata},
        )

        assert document.page_count == expected_pages
        assert document.to_text(page_number=0) == ("Marie LLM dispatch fixture page 1")

    validate_fixture_manifest(manifest)


def test_validate_fixture_manifest_rejects_incomplete_ocr_metadata(
    tmp_path: Path,
) -> None:
    source = tmp_path / "broken.tif"
    image = Image.new("RGB", (10, 10), "white")
    image.save(source)
    Path(f"{source}.meta.json").write_text(
        json.dumps(
            {
                "pages": "1",
                "ocr": [
                    {
                        "meta": {"page": 0},
                        "words": [],
                    }
                ],
            }
        )
    )
    manifest = tmp_path / "manifest.txt"
    manifest.write_text(f"{source}\n")

    with pytest.raises(ValueError, match="invalid OCR metadata"):
        validate_fixture_manifest(manifest)


def test_validate_fixture_manifest_rejects_list_extraction(tmp_path: Path) -> None:
    manifest = write_fixtures(tmp_path)
    source = Path(manifest.read_text().splitlines()[0])
    metadata_path = Path(f"{source}.meta.json")
    metadata = json.loads(metadata_path.read_text())
    metadata["extraction"] = []
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="extraction must be a JSON object"):
        validate_fixture_manifest(manifest)


def test_build_command_keeps_pool_selection_automatic(tmp_path: Path) -> None:
    phase = next(phase for phase in default_phases() if phase.name == "delay")
    command = build_stresser_command(
        python=Path(".venv/bin/python"),
        config=Path("tools/stress/gateway-e2e.config.json"),
        manifest=Path("fixtures/mixed.manifest.txt"),
        request_template=Path("tools/stress/mock_annotator_llm.invoke.json"),
        artifact_dir=tmp_path,
        run_id="replay-queued-dispatch-delay",
        phase=phase,
        routing_settle_timeout=30,
        debug_sample_interval=30,
        max_retained_jobs=1000,
        max_metric_samples=10000,
    )

    assert "--llm-pool-id" not in command
    assert command[command.index("--job-count") + 1] == "6"
    assert command[command.index("--fault-profile") + 1] == "timeout"
    assert command[command.index("--min-terminal-completion-pct") + 1] == "100"
    assert command[command.index("--max-open-jobs") + 1] == "0"
    assert command[command.index("--routing-settle-timeout") + 1] == "30"
    assert command[command.index("--debug-sample-interval") + 1] == "30"
    assert command[command.index("--max-retained-jobs") + 1] == "1000"
    assert command[command.index("--max-metric-samples") + 1] == "10000"


@pytest.mark.parametrize(
    ("phase_name", "minimum_completion"),
    [("hang", "100"), ("retryable", "0")],
)
def test_build_command_uses_direct_batch_outcome(
    tmp_path: Path, phase_name: str, minimum_completion: str
) -> None:
    phase = next(phase for phase in default_phases() if phase.name == phase_name)

    command = build_stresser_command(
        python=Path(".venv/bin/python"),
        config=Path("tools/stress/gateway-e2e.config.json"),
        manifest=Path("fixtures/mixed.manifest.txt"),
        request_template=Path("tools/stress/mock_annotator_llm.invoke.json"),
        artifact_dir=tmp_path,
        run_id=f"replay-direct-batch-{phase_name}",
        phase=phase,
        runtime_mode="direct-batch",
    )

    assert (
        command[command.index("--min-terminal-completion-pct") + 1]
        == minimum_completion
    )


def test_build_command_passes_exact_input_counts(tmp_path: Path) -> None:
    phase = default_phases(
        normal_pool_counts={
            "document-small": 80,
            "document-medium": 15,
            "document-large": 5,
        }
    )[0]
    command = build_stresser_command(
        python=Path(".venv/bin/python"),
        config=Path("tools/stress/gateway-e2e.config.json"),
        manifest=Path("fixtures/mixed.manifest.txt"),
        request_template=Path("tools/stress/mock_annotator_llm.invoke.json"),
        artifact_dir=tmp_path,
        run_id="weighted-normal",
        phase=phase,
    )

    assert command[command.index("--job-count") + 1] == "100"
    assert command[command.index("--input-counts") + 1] == "80,15,5"


def test_validate_report_requires_balanced_pool_deltas() -> None:
    phase = default_phases()[0]
    report = {
        "summary": {
            "submitted_jobs": 60,
            "completed_jobs": 60,
            "failed_jobs": 0,
            "event_timeout_jobs": 0,
        },
        "reliability": {"observed": {"open_jobs": 0}},
        "routing_qualification": {
            "drr": {
                pool: {"delta": {"accepted": 20, "completed": 20}}
                for pool in (
                    "document-small",
                    "document-medium",
                    "document-large",
                )
            },
            "policy": {"synchronized": True},
            "projection": {"pending_at_end": 0},
        },
    }

    validate_report(phase, report)
    report["routing_qualification"]["drr"]["document-large"]["delta"]["accepted"] = 19

    with pytest.raises(AssertionError, match="document-large accepted"):
        validate_report(phase, report)


def test_validate_report_rejects_unsettled_runtime_counters() -> None:
    phase = default_phases()[0]
    report = {
        "summary": {
            "submitted_jobs": 60,
            "completed_jobs": 60,
            "failed_jobs": 0,
            "event_timeout_jobs": 0,
        },
        "reliability": {"observed": {"open_jobs": 0}},
        "routing_qualification": {
            "counters_settled": False,
            "latest_observation": {
                "ok": False,
                "error": "HTTP 503: runtime_unavailable",
            },
            "drr": {},
            "policy": {"synchronized": True},
            "projection": {"pending_at_end": 0},
        },
    }

    with pytest.raises(
        AssertionError,
        match="runtime counters did not settle.*runtime_unavailable",
    ):
        validate_report(phase, report)


def test_validate_direct_batch_report_ignores_admission_counters() -> None:
    phase = default_phases()[0]
    report = {
        "summary": {
            "submitted_jobs": 60,
            "completed_jobs": 60,
            "failed_jobs": 0,
            "event_timeout_jobs": 0,
        },
        "reliability": {"observed": {"open_jobs": 0}},
        "latency_stats_ms": {
            name: {"p50": 1.0, "p95": 2.0}
            for name in ("queue_wait", "execution", "end_to_end")
        },
        "routing_qualification": {
            "available": True,
            "dispatcher_counters": {
                "delta": {"claims": 0, "provider_starts": 0, "completed": 0}
            },
            "drr": {"document-small": {"delta": {"accepted": 60, "completed": 60}}},
        },
    }

    validate_report(phase, report, runtime_mode="direct-batch")

    report["routing_qualification"]["dispatcher_counters"]["delta"][
        "provider_starts"
    ] = 60
    with pytest.raises(
        AssertionError, match="direct-batch benchmark invalid.*ran queued-dispatch"
    ):
        validate_report(phase, report, runtime_mode="direct-batch")

    report["routing_qualification"]["dispatcher_counters"]["delta"].update(
        {"provider_starts": 0, "claims": 1}
    )
    with pytest.raises(
        AssertionError, match="direct-batch benchmark invalid.*ran queued-dispatch"
    ):
        validate_report(phase, report, runtime_mode="direct-batch")


@pytest.mark.parametrize(
    ("phase_name", "completed", "failed"),
    [
        ("hang", 3, 0),
        ("unreachable", 6, 0),
        ("unreachable", 5, 1),
        ("retryable", 5, 1),
    ],
)
def test_validate_direct_batch_uses_runtime_specific_outcome(
    phase_name: str, completed: int, failed: int
) -> None:
    phase = next(phase for phase in default_phases() if phase.name == phase_name)
    report = {
        "summary": {
            "submitted_jobs": phase.jobs,
            "completed_jobs": completed,
            "failed_jobs": failed,
            "event_timeout_jobs": 0,
        },
        "reliability": {"observed": {"open_jobs": 0}},
        "routing_qualification": {
            "available": True,
            "dispatcher_counters": {
                "delta": {"claims": 0, "provider_starts": 0, "completed": 0}
            },
            "drr": {},
        },
    }

    validate_report(phase, report, runtime_mode="direct-batch")


def _failure_phase_report(
    phase_name: str,
    *,
    completed: int,
    failed: int,
    retries: int,
    refunded_cost: int,
) -> tuple[Phase, dict[str, Any]]:
    phase = next(phase for phase in default_phases() if phase.name == phase_name)
    per_pool = expected_pool_counts(phase.jobs)
    report: dict[str, Any] = {
        "summary": {
            "submitted_jobs": phase.jobs,
            "completed_jobs": completed,
            "failed_jobs": failed,
            "event_timeout_jobs": 0,
        },
        "reliability": {"observed": {"open_jobs": 0}},
        "routing_qualification": {
            "drr": {
                pool: {
                    "delta": {
                        "accepted": count,
                        "completed": count if completed else 0,
                        "refunded_cost": refunded_cost
                        if pool == "document-small"
                        else 0,
                    }
                }
                for pool, count in per_pool.items()
            },
            "dispatcher_counters": {"delta": {"retries": retries}},
            "policy": {"synchronized": True},
            "projection": {"pending_at_end": 0},
        },
    }
    return phase, report


def test_validate_report_requires_prewrite_refund_for_unreachable_phase() -> None:
    phase, report = _failure_phase_report(
        "unreachable", completed=6, failed=0, retries=0, refunded_cost=1
    )

    validate_report(phase, report)
    report["routing_qualification"]["drr"]["document-small"]["delta"][
        "refunded_cost"
    ] = 0

    with pytest.raises(AssertionError, match="prewrite retry evidence"):
        validate_report(phase, report)


def test_validate_report_requires_deferred_retry_for_503_phase() -> None:
    phase, report = _failure_phase_report(
        "retryable", completed=6, failed=0, retries=3, refunded_cost=0
    )

    validate_report(phase, report)
    report["routing_qualification"]["dispatcher_counters"]["delta"]["retries"] = 0

    with pytest.raises(AssertionError, match="deferred retry evidence"):
        validate_report(phase, report)


def test_validate_report_rejects_retry_after_ambiguous_hang() -> None:
    phase, report = _failure_phase_report(
        "hang", completed=0, failed=3, retries=0, refunded_cost=0
    )

    validate_report(phase, report)
    report["routing_qualification"]["dispatcher_counters"]["delta"]["retries"] = 1

    with pytest.raises(AssertionError, match="ambiguous hang retry"):
        validate_report(phase, report)


def test_validate_report_requires_terminal_max_output_evidence() -> None:
    phase, report = _failure_phase_report(
        "max-output", completed=0, failed=6, retries=0, refunded_cost=0
    )
    report["routing_qualification"]["dispatcher_counters"]["delta"].update(
        {
            "completed": 6,
            "execution_errors": 0,
            "remote_outcomes_failed": 0,
        }
    )

    validate_report(phase, report)
    report["routing_qualification"]["dispatcher_counters"]["delta"][
        "execution_errors"
    ] = 1

    with pytest.raises(AssertionError, match="max-output transport errors"):
        validate_report(phase, report)


def test_validate_report_requires_local_repetition_recovery() -> None:
    phase, report = _failure_phase_report(
        "repetition", completed=6, failed=0, retries=0, refunded_cost=0
    )
    report["routing_qualification"]["dispatcher_counters"]["delta"][
        "repetition_retries"
    ] = 6

    validate_report(phase, report)
    report["routing_qualification"]["dispatcher_counters"]["delta"][
        "repetition_retries"
    ] = 0

    with pytest.raises(AssertionError, match="repetition recovery evidence"):
        validate_report(phase, report)


def test_validate_report_requires_terminal_repetition_evidence() -> None:
    phase, report = _failure_phase_report(
        "repetition-terminal", completed=0, failed=6, retries=0, refunded_cost=0
    )
    report["routing_qualification"]["dispatcher_counters"]["delta"].update(
        {
            "completed": 6,
            "execution_errors": 6,
            "provider_starts": 12,
            "remote_outcomes_failed": 0,
            "repetition_retries": 6,
        }
    )

    validate_report(phase, report)
    report["routing_qualification"]["dispatcher_counters"]["delta"][
        "execution_errors"
    ] = 5

    with pytest.raises(
        AssertionError, match="terminal repetition execution error accounting"
    ):
        validate_report(phase, report)


def test_write_html_report_contains_run_details_and_escapes_values(
    tmp_path: Path,
) -> None:
    summary = {
        "suite_id": "suite-<script>",
        "runtime_mode": "queued-dispatch",
        "status": "failed",
        "manifest": "/tmp/mixed-pools.manifest.txt",
        "provider": {"max_concurrent_calls": 20, "processing_delay_ms": 125},
        "error": "provider <unavailable>",
        "phases": [
            {
                "phase": "normal",
                "status": "failed",
                "expected_outcome": "completed",
                "jobs": 60,
                "completed": 41,
                "failed": 19,
                "duration_seconds": 106.25,
                "throughput_jobs_per_second": 0.56,
                "queue_wait_p95_ms": 1200.0,
                "execution_p95_ms": 5100.0,
                "end_to_end_p50_ms": 3200.0,
                "end_to_end_p95_ms": 6300.0,
                "observed_runtime_mode": "queued-dispatch",
                "provider_concurrency_limit": 20,
                "provider_peak_concurrency": 18,
                "provider_waited_calls": 32,
                "provider_calls_per_second": 14.5,
                "pool_latency": {
                    "document-small": {
                        "jobs": 41,
                        "queue_wait_p95_ms": 1200.0,
                        "execution_p95_ms": 5100.0,
                        "end_to_end_p95_ms": 6300.0,
                    }
                },
            }
        ],
    }

    path = tmp_path / "report.html"
    write_html_report(path, summary)
    rendered = path.read_text()

    assert "suite-&lt;script&gt;" in rendered
    assert "provider &lt;unavailable&gt;" in rendered
    assert "normal-live.json" in rendered
    assert "normal-jobs.jsonl" in rendered
    assert "Completed" in rendered
    assert "Retries" in rendered
    assert "Refunded cost" in rendered
    assert "41" in rendered
    assert "How to read these metrics" in rendered
    assert "95% of requests completed at or below" in rendered
    assert "total system output" in rendered
    assert "P95 values from different columns cannot be added" in rendered
    assert "Execution path" in rendered
    assert "queued-dispatch" in rendered
    assert "Provider cap 20 · delay 125 ms" in rendered
    assert "Provider peak" in rendered
    assert "Provider calls/s" in rendered
    assert "Per-pool latency" in rendered
    assert "document-small: 41 jobs" in rendered
    assert "http-equiv=\"refresh\"" not in rendered


@pytest.mark.parametrize('phase_fails', [False, True])
@pytest.mark.parametrize('reset_fails', [False, True])
def test_run_keeps_phase_result_when_reset_succeeds_or_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    phase_fails: bool,
    reset_fails: bool,
) -> None:
    fault_settings: list[dict[str, object]] = []
    monkeypatch.setattr(
        "tools.stress.llm_dispatch_stress_suite.preflight_dispatch_runtime",
        lambda _config, _fabric, **_kwargs: {
            "registered_dispatchers": 1,
            "running_dispatchers": 1,
            "pending_requests": 0,
            "inflight_requests": 0,
            "endpoint_replicas": 1,
            "dispatch_execution_limit": 4,
        },
    )

    def set_fault(_url: str, settings: dict[str, object]) -> None:
        fault_settings.append(settings.copy())
        if reset_fails and settings['profile'] == 'normal':
            raise RuntimeError('AIMock reset unavailable')

    monkeypatch.setattr('tools.stress.llm_dispatch_stress_suite.set_fault', set_fault)
    monkeypatch.setattr(
        'tools.stress.llm_dispatch_stress_suite.get_fault_state',
        lambda _url: {
            'maxConcurrentCalls': 4,
            'processingDelayMs': 100,
            'peakConcurrentCalls': 4,
        },
    )

    def execute_phase(command: list[str], **_kwargs: object) -> None:
        report_path = Path(command[command.index("--report") + 1])
        if not phase_fails:
            _, report = _failure_phase_report(
                'error', completed=0, failed=6, retries=0, refunded_cost=0
            )
            report['summary']['duration_seconds'] = 1.0
            report['latency_stats_ms'] = {
                key: {'p50': 1.0, 'p95': 1.0}
                for key in ('queue_wait', 'execution', 'end_to_end')
            }
            report['routing_qualification']['endpoint_groups'] = [
                {'replicas': [{'enabled': True, 'execution_limit': 4}]}
            ]
            report_path.write_text(json.dumps(report))
            return
        report_path.write_text(
            json.dumps(
                {
                    "summary": {
                        "submitted_jobs": 6,
                        "completed_jobs": 1,
                        "failed_jobs": 0,
                        "event_timeout_jobs": 5,
                    },
                    "reliability": {
                        "observed": {"open_jobs": 5},
                    },
                    "verification": {
                        "passed": False,
                        "errors": ["Open jobs after drain 5 exceed the allowed 0"],
                    },
                }
            )
        )
        raise subprocess.CalledProcessError(2, command)

    monkeypatch.setattr(
        "tools.stress.llm_dispatch_stress_suite.subprocess.run", execute_phase
    )
    args = Namespace(
        suite_id="restore-test",
        runtime_mode="queued-dispatch",
        artifact_root=tmp_path,
        input_manifest=None,
        fabric_group_id="default",
        phases="error",
        job_count=60,
        aimock_admin_url="http://127.0.0.1:4011",
        python=Path(".venv/bin/python"),
        config=Path("config.json"),
        request_template=Path("request.json"),
    )

    if phase_fails:
        with pytest.raises(
            RuntimeError, match="Open jobs after drain 5 exceed the allowed 0"
        ):
            run(args)
    else:
        summary_path = run(args)
        assert summary_path.is_file()

    summary = json.loads(
        (tmp_path / 'restore-test-queued-dispatch' / 'comparison.json').read_text()
    )
    expected_status = 'failed' if phase_fails else 'passed'
    assert summary['status'] == expected_status
    assert summary['phases'][0]['status'] == expected_status
    if phase_fails:
        assert 'Open jobs after drain 5' in summary['error']
    else:
        assert 'error' not in summary
    if reset_fails:
        assert (
            'Failed to reset AIMock fault profile: AIMock reset unavailable'
            in caplog.text
        )
    else:
        assert 'Failed to reset AIMock fault profile' not in caplog.text

    assert fault_settings == [
        {
            "profile": "terminal_error",
            "maxConcurrentCalls": 4,
            "processingDelayMs": 100,
        },
        {
            "profile": "normal",
            "timeoutMs": 180000,
            "maxConcurrentCalls": 0,
            "processingDelayMs": 0,
        },
    ]
    report = tmp_path / "restore-test-queued-dispatch" / "report.html"
    assert report.is_file()
    assert expected_status in report.read_text()
    run_index = tmp_path / "index.html"
    assert run_index.is_file()
    assert "restore-test-queued-dispatch/report.html" in run_index.read_text()


def test_run_index_compares_matching_runtime_modes(tmp_path: Path) -> None:
    def write_summary(mode: str, *, throughput: float, e2e_p95: float) -> None:
        run_dir = tmp_path / f"comparison-{mode}"
        run_dir.mkdir()
        (run_dir / "comparison.json").write_text(
            json.dumps(
                {
                    "suite_id": "comparison",
                    "runtime_mode": mode,
                    "status": "passed",
                    "provider": {
                        "max_concurrent_calls": 4,
                        "processing_delay_ms": 100,
                    },
                    "phases": [
                        {
                            "phase": "normal",
                            "status": "passed",
                            "jobs": 100,
                            "completed": 100,
                            "failed": 0,
                            "throughput_jobs_per_second": throughput,
                            "queue_wait_p95_ms": 20.0,
                            "execution_p95_ms": 800.0,
                            "end_to_end_p95_ms": e2e_p95,
                            "observed_runtime_mode": mode,
                            "provider_concurrency_limit": 4,
                            "provider_processing_delay_ms": 100,
                            "dispatch_execution_limit": (
                                4 if mode == "queued-dispatch" else None
                            ),
                        }
                    ],
                }
            )
        )

    write_summary("queued-dispatch", throughput=12.0, e2e_p95=1200.0)
    write_summary("direct-batch", throughput=10.0, e2e_p95=1000.0)

    _write_run_index(tmp_path)

    index = (tmp_path / "index.html").read_text()
    assert "Runtime mode comparison" in index
    assert "comparison-queued-dispatch/report.html" in index
    assert "comparison-direct-batch/report.html" in index
    assert "Queued 20.00% higher" in index
    assert "Direct 200.00 ms lower" in index
    assert "How to read this comparison" in index
    assert "The table names the better result" in index
    assert "Higher throughput is better; lower latency is better" in index
    assert "Higher throughput can coexist with higher per-request latency" in index


def test_result_row_reports_completed_job_throughput_and_open_circuit() -> None:
    phase = Phase("error", 6, {"profile": "terminal_error"}, "failed", 60)
    empty_latency = {"p50": None, "p95": None}
    report = {
        "summary": {
            "completed_jobs": 0,
            "failed_jobs": 6,
            "duration_seconds": 10.0,
            "throughput": 0.6,
        },
        "latency_stats_ms": {
            "queue_wait": empty_latency,
            "execution": empty_latency,
            "end_to_end": empty_latency,
        },
        "reliability": {"observed": {"open_jobs": 0}},
        "debug_sampling": {
            "samples": [
                {
                    "llm_dispatch": {
                        "endpoint_groups": [
                            {"replicas": [{"circuit": "open"}]}
                        ]
                    }
                }
            ]
        },
        "jobs": [],
    }

    row = _result_row(phase, report)

    assert row["throughput_jobs_per_second"] == 0.0
    assert row["circuit_open_observed"] is True


def test_run_index_does_not_rank_different_terminal_outcomes(tmp_path: Path) -> None:
    outcomes = {
        "queued-dispatch": (6, 0),
        "direct-batch": (5, 1),
    }
    for mode, (completed, failed) in outcomes.items():
        run_dir = tmp_path / f"outcomes-{mode}"
        run_dir.mkdir()
        (run_dir / "comparison.json").write_text(
            json.dumps(
                {
                    "suite_id": "outcomes",
                    "runtime_mode": mode,
                    "status": "passed",
                    "phases": [
                        {
                            "phase": "unreachable",
                            "status": "passed",
                            "jobs": 6,
                            "completed": completed,
                            "failed": failed,
                            "unresolved": 0,
                            "throughput_jobs_per_second": 10.0,
                            "end_to_end_p95_ms": 1000.0,
                            "observed_runtime_mode": mode,
                            "provider_concurrency_limit": 4,
                            "provider_processing_delay_ms": 100,
                            "dispatch_execution_limit": (
                                4 if mode == "queued-dispatch" else None
                            ),
                        }
                    ],
                }
            )
        )

    _write_run_index(tmp_path)

    index = (tmp_path / "index.html").read_text()
    assert "Different outcomes" in index
    assert index.count("Not comparable") == 2


def test_run_index_does_not_rank_failure_only_runs(tmp_path: Path) -> None:
    for mode in ("queued-dispatch", "direct-batch"):
        run_dir = tmp_path / f"failures-{mode}"
        run_dir.mkdir()
        (run_dir / "comparison.json").write_text(
            json.dumps(
                {
                    "suite_id": "failures",
                    "runtime_mode": mode,
                    "status": "passed",
                    "phases": [
                        {
                            "phase": "error",
                            "status": "passed",
                            "jobs": 6,
                            "completed": 0,
                            "failed": 6,
                            "unresolved": 0,
                            "throughput_jobs_per_second": 0.0,
                            "end_to_end_p95_ms": 1000.0,
                            "observed_runtime_mode": mode,
                            "provider_concurrency_limit": 4,
                            "provider_processing_delay_ms": 100,
                            "dispatch_execution_limit": (
                                4 if mode == "queued-dispatch" else None
                            ),
                        }
                    ],
                }
            )
        )

    _write_run_index(tmp_path)

    index = (tmp_path / "index.html").read_text()
    assert "Same terminal outcome" in index
    assert index.count("Not comparable") == 2


def test_run_index_rejects_mislabeled_direct_batch_result(tmp_path: Path) -> None:
    for mode in ("queued-dispatch", "direct-batch"):
        run_dir = tmp_path / f"comparison-{mode}"
        run_dir.mkdir()
        (run_dir / "comparison.json").write_text(
            json.dumps(
                {
                    "suite_id": "comparison",
                    "runtime_mode": mode,
                    "status": "passed",
                    "provider": {
                        "max_concurrent_calls": 4,
                        "processing_delay_ms": 100,
                    },
                    "phases": [
                        {
                            "phase": "normal",
                            "status": "passed",
                            "jobs": 100,
                            "completed": 100,
                            "failed": 0,
                            "throughput_jobs_per_second": 10.0,
                            "end_to_end_p95_ms": 1000.0,
                            "observed_runtime_mode": "queued-dispatch",
                            "provider_concurrency_limit": 4,
                            "provider_processing_delay_ms": 100,
                            "dispatch_execution_limit": (
                                4 if mode == "queued-dispatch" else None
                            ),
                        }
                    ],
                }
            )
        )

    _write_run_index(tmp_path)

    index = (tmp_path / "index.html").read_text()
    assert (
        "Invalid baseline: requested Direct batch, but this run used Queued dispatch"
        in index
    )
    assert index.count("Not comparable") == 2
    assert (
        "Performance is ranked only when both modes completed every job with "
        "matching execution paths and provider capacity" in index
    )


def test_run_index_pairs_different_ids_with_the_same_workload(tmp_path: Path) -> None:
    for suite_id, mode in (
        ("small-heavy-20260924-135736", "queued-dispatch"),
        ("small-heavy-20260924-135922", "direct-batch"),
    ):
        run_dir = tmp_path / f"{suite_id}-{mode}"
        run_dir.mkdir()
        (run_dir / "comparison.json").write_text(
            json.dumps(
                {
                    "suite_id": suite_id,
                    "runtime_mode": mode,
                    "status": "passed",
                    "provider": {
                        "max_concurrent_calls": 4,
                        "processing_delay_ms": 100,
                    },
                    "pool_counts": {
                        "document-small": 80,
                        "document-medium": 15,
                        "document-large": 0,
                    },
                    "phases": [
                        {
                            "phase": "normal",
                            "expected_outcome": "completed",
                            "status": "passed",
                            "jobs": 95,
                            "completed": 95,
                            "failed": 0,
                            "pool_counts": {
                                "document-small": 80,
                                "document-medium": 15,
                                "document-large": 0,
                            },
                            "throughput_jobs_per_second": 10.0,
                            "end_to_end_p95_ms": 1000.0,
                            "observed_runtime_mode": mode,
                            "provider_concurrency_limit": 4,
                            "provider_processing_delay_ms": 100,
                            "dispatch_execution_limit": (
                                4 if mode == "queued-dispatch" else None
                            ),
                        }
                    ],
                }
            )
        )

    _write_run_index(tmp_path)

    index = (tmp_path / "index.html").read_text()
    assert "small-heavy-20260924-135736" in index
    assert "small-heavy-20260924-135922" in index
    assert "Workload match" in index


def test_run_index_refuses_to_compare_different_provider_capacity(
    tmp_path: Path,
) -> None:
    for mode, concurrency in (("queued-dispatch", 4), ("direct-batch", 20)):
        run_dir = tmp_path / f"capacity-{mode}"
        run_dir.mkdir()
        (run_dir / "comparison.json").write_text(
            json.dumps(
                {
                    "suite_id": "capacity",
                    "runtime_mode": mode,
                    "status": "passed",
                    "provider": {
                        "max_concurrent_calls": concurrency,
                        "processing_delay_ms": 100,
                    },
                    "phases": [
                        {
                            "phase": "normal",
                            "status": "passed",
                            "jobs": 100,
                            "completed": 100,
                            "failed": 0,
                            "throughput_jobs_per_second": 10.0,
                            "end_to_end_p95_ms": 1000.0,
                            "observed_runtime_mode": mode,
                            "provider_concurrency_limit": concurrency,
                            "provider_processing_delay_ms": 100,
                            "dispatch_execution_limit": (
                                4 if mode == "queued-dispatch" else None
                            ),
                        }
                    ],
                }
            )
        )

    _write_run_index(tmp_path)

    index = (tmp_path / "index.html").read_text()
    assert "Capacity mismatch" in index
    assert index.count("Not comparable") == 2
