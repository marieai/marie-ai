#!/usr/bin/env python3
"""Replay the mixed-pool LLM dispatch qualification matrix."""

from __future__ import annotations

import argparse
import html
import json
import logging
import re
import subprocess
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw

from marie.extract.readers.meta_reader.meta_reader import MetaReader

POOLS = ("document-small", "document-medium", "document-large")


@dataclass(frozen=True)
class Phase:
    name: str
    jobs: int
    fault: dict[str, object]
    expected_outcome: str
    terminal_timeout: int
    retry_evidence: str | None = None
    pool_counts: dict[str, int] | None = None
    direct_expected_outcome: str | None = None


def expected_outcome_for(phase: Phase, runtime_mode: str) -> str:
    if runtime_mode == "direct-batch" and phase.direct_expected_outcome is not None:
        return phase.direct_expected_outcome
    return phase.expected_outcome


def default_phases(
    normal_jobs: int = 60,
    normal_terminal_timeout: int | None = None,
    normal_pool_counts: dict[str, int] | None = None,
) -> list[Phase]:
    if normal_pool_counts is not None:
        normal_jobs = sum(normal_pool_counts.values())
    resolved_normal_timeout = (
        normal_terminal_timeout
        if normal_terminal_timeout is not None
        else max(1800, normal_jobs * 8)
    )
    return [
        Phase(
            "normal",
            normal_jobs,
            {"profile": "normal"},
            "completed",
            resolved_normal_timeout,
            pool_counts=normal_pool_counts,
        ),
        Phase(
            "unreachable",
            6,
            {"profile": "normal", "outageMs": 5000},
            "completed",
            900,
            "prewrite_refund",
            direct_expected_outcome="settled",
        ),
        Phase(
            "retryable",
            6,
            {"profile": "transient_error", "transientErrors": 3},
            "completed",
            900,
            "deferred_retry",
            direct_expected_outcome="settled",
        ),
        Phase(
            "delay",
            6,
            {"profile": "timeout", "timeoutMs": 5000},
            "completed",
            900,
        ),
        Phase(
            "hang",
            3,
            {"profile": "timeout", "timeoutMs": 35000},
            "failed",
            600,
            "no_retry",
            direct_expected_outcome="completed",
        ),
        Phase("error", 6, {"profile": "terminal_error"}, "failed", 600),
        Phase(
            "max-output",
            6,
            {"profile": "max_tokens"},
            "failed",
            600,
            "max_output_terminal",
        ),
        Phase(
            "repetition",
            6,
            {"profile": "repetition"},
            "completed",
            600,
            "repetition_recovery",
        ),
        Phase(
            "repetition-terminal",
            6,
            {"profile": "persistent_repetition"},
            "failed",
            600,
            "repetition_terminal",
        ),
        Phase(
            "chaos",
            15,
            {
                "profile": "chaos",
                "timeoutMs": 6000,
                "chaosErrorRate": 0.2,
                "chaosTimeoutRate": 0.2,
                "chaosSlowRate": 0.3,
                "chaosSlowMs": 3000,
            },
            "mixed",
            1200,
        ),
    ]


def expected_pool_counts(job_count: int) -> dict[str, int]:
    pool_count = len(POOLS)
    return {
        pool: (job_count + pool_count - 1 - index) // pool_count
        for index, pool in enumerate(POOLS)
    }


def parse_pool_counts(value: str) -> dict[str, int]:
    counts = {pool: 0 for pool in POOLS}
    seen: set[str] = set()
    for raw_entry in value.split(","):
        entry = raw_entry.strip()
        if "=" not in entry:
            raise argparse.ArgumentTypeError("pool counts must use pool=count entries")
        pool, raw_count = (part.strip() for part in entry.split("=", 1))
        if pool not in counts:
            raise argparse.ArgumentTypeError(f"unknown pool: {pool}")
        if pool in seen:
            raise argparse.ArgumentTypeError(f"duplicate pool: {pool}")
        try:
            count = int(raw_count)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"invalid count for {pool}: {raw_count}"
            ) from exc
        if count < 0:
            raise argparse.ArgumentTypeError(
                f"count for {pool} must be greater than or equal to zero"
            )
        counts[pool] = count
        seen.add(pool)
    if not seen or sum(counts.values()) == 0:
        raise argparse.ArgumentTypeError("pool counts must include at least one job")
    return counts


def _page_image(page: int) -> Image.Image:
    image = Image.new("RGB", (640, 800), "white")
    draw = ImageDraw.Draw(image)
    draw.text((40, 40), f"Marie LLM dispatch fixture - page {page + 1}", fill="black")
    return image


def _metadata(page_count: int) -> dict[str, object]:
    return {
        "pages": str(page_count),
        "ocr": [
            {
                "meta": {
                    "page": page,
                    "lang": "en",
                    "imageSize": {"width": 640, "height": 800},
                    "lines": [0],
                    "lines_bboxes": [[40, 40, 360, 24]],
                    "format": "xywh",
                },
                "words": [
                    {
                        "id": 0,
                        "text": f"Marie LLM dispatch fixture page {page + 1}",
                        "confidence": 1.0,
                        "box": [40, 40, 360, 24],
                        "line": 0,
                        "word_index": 0,
                    }
                ],
                "lines": [
                    {
                        "line": 0,
                        "wordids": [0],
                        "text": f"Marie LLM dispatch fixture page {page + 1}",
                        "bbox": [40, 40, 360, 24],
                        "confidence": 1.0,
                    }
                ],
            }
            for page in range(page_count)
        ],
        "extraction": {},
    }


def write_fixtures(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    specs = (
        ("document-small-1p.tif", 1),
        ("document-medium-8p.tif", 8),
        ("document-large-30p.pdf", 30),
    )
    sources: list[Path] = []
    for name, page_count in specs:
        source = root / name
        pages = [_page_image(page) for page in range(page_count)]
        pages[0].save(source, save_all=True, append_images=pages[1:])
        Path(f"{source}.meta.json").write_text(
            json.dumps(_metadata(page_count), indent=2) + "\n",
            encoding="utf-8",
        )
        sources.append(source.resolve())
    manifest = root / "mixed-pools.manifest.txt"
    manifest.write_text("".join(f"{source}\n" for source in sources), encoding="utf-8")
    return manifest


def validate_fixture_manifest(manifest: Path) -> None:
    for raw_line in manifest.expanduser().read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        source = Path(line).expanduser().resolve()
        metadata_path = Path(f"{source}.meta.json")
        if not source.is_file():
            raise ValueError(f"fixture does not exist: {source}")
        if not metadata_path.is_file():
            raise ValueError(f"fixture metadata does not exist: {metadata_path}")

        try:
            metadata = json.loads(metadata_path.read_text())
            page_count = int(metadata["pages"])
            ocr = metadata["ocr"]
            extraction = metadata.get("extraction", {})
            if not isinstance(extraction, dict):
                raise ValueError("extraction must be a JSON object")
            if page_count != len(ocr):
                raise ValueError(
                    f"pages={page_count} but OCR contains {len(ocr)} pages"
                )
            frames = [np.zeros((1, 1, 3), dtype=np.uint8) for _ in ocr]
            document = MetaReader.from_data(
                frames=frames,
                ocr_meta=ocr,
                unstructured_meta={"source_metadata": metadata},
            )
            if document.page_count != page_count:
                raise ValueError(
                    f"reader found {document.page_count} pages, expected {page_count}"
                )
        except (AssertionError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid OCR metadata for {source}: {exc}") from exc


def _non_negative_int(value: object, label: str, issues: list[str]) -> int:
    if type(value) is not int or value < 0:
        issues.append(f"{label} is unavailable")
        return 0
    return value


def validate_clean_dispatch_runtime(
    snapshot: dict[str, Any], required_execution_capacity: int | None = None
) -> dict[str, int]:
    issues: list[str] = []
    observation = snapshot.get("observation")
    if not isinstance(observation, dict) or observation.get("stale") is not False:
        issues.append("runtime observation is unavailable or stale")

    runtime = snapshot.get("runtime_summary")
    if not isinstance(runtime, dict):
        raise RuntimeError(
            "dispatch runtime is not clean: runtime summary is unavailable"
        )

    registered = _non_negative_int(
        runtime.get("registered_dispatchers"), "registered dispatcher count", issues
    )
    running = _non_negative_int(
        runtime.get("running_dispatchers"), "running dispatcher count", issues
    )
    pending = _non_negative_int(
        runtime.get("pending_request_count"), "pending request count", issues
    )
    inflight = _non_negative_int(
        runtime.get("inflight_request_count"), "in-flight request count", issues
    )
    if registered == 0:
        issues.append("no dispatcher is registered")
    elif running != registered:
        issues.append(f"running dispatchers={running}/{registered}")
    if pending:
        issues.append(f"pending requests={pending}")
    if inflight:
        issues.append(f"in-flight requests={inflight}")

    pools = snapshot.get("pools")
    if not isinstance(pools, list):
        issues.append("pool state is unavailable")
        pools = []
    for pool in pools:
        if not isinstance(pool, dict):
            issues.append("pool state is malformed")
            continue
        pool_id = str(pool.get("pool_id") or "unknown pool")
        depth = _non_negative_int(
            pool.get("request_queue_depth"), f"{pool_id} queue depth", issues
        )
        reserved = _non_negative_int(
            pool.get("reserved_items"), f"{pool_id} reserved item count", issues
        )
        if depth:
            issues.append(f"{pool_id} queued items={depth}")
        if reserved:
            issues.append(f"{pool_id} reserved items={reserved}")
        state_counts = pool.get("state_counts")
        if not isinstance(state_counts, dict):
            issues.append(f"{pool_id} request states are unavailable")
            continue
        for state in ("ready", "claimed", "running", "unknown"):
            count = _non_negative_int(
                state_counts.get(state), f"{pool_id} {state} count", issues
            )
            if count:
                issues.append(f"{pool_id} {state} items={count}")

    groups = snapshot.get("endpoint_groups")
    if not isinstance(groups, list):
        issues.append("endpoint state is unavailable")
        groups = []
    replica_count = 0
    execution_limit = 0
    for group in groups:
        if not isinstance(group, dict):
            issues.append("endpoint group state is malformed")
            continue
        replicas = group.get("replicas")
        if not isinstance(replicas, list):
            issues.append(
                f"{group.get('group_id', 'endpoint group')} replicas unavailable"
            )
            continue
        for replica in replicas:
            if not isinstance(replica, dict):
                issues.append("endpoint replica state is malformed")
                continue
            replica_count += 1
            replica_id = str(replica.get("replica_id") or "unknown replica")
            if replica.get("enabled") is not True:
                issues.append(f"{replica_id} is disabled")
            reserved_items = _non_negative_int(
                replica.get("reserved_items"),
                f"{replica_id} reserved item count",
                issues,
            )
            reserved_bytes = _non_negative_int(
                replica.get("reserved_bytes"),
                f"{replica_id} reserved byte count",
                issues,
            )
            replica_limit = _non_negative_int(
                replica.get("execution_limit"),
                f"{replica_id} execution limit",
                issues,
            )
            execution_limit += replica_limit
            if replica_limit == 0:
                issues.append(f"{replica_id} has no execution capacity")
            if reserved_items:
                issues.append(f"{replica_id} reserved items={reserved_items}")
            if reserved_bytes:
                issues.append(f"{replica_id} reserved bytes={reserved_bytes}")

    if replica_count == 0:
        issues.append("no endpoint replica is configured")
    if (
        required_execution_capacity is not None
        and execution_limit < required_execution_capacity
    ):
        issues.append(
            f"dispatch capacity {execution_limit} is lower than requested provider "
            f"concurrency {required_execution_capacity}"
        )
    if issues:
        raise RuntimeError("dispatch runtime is not clean: " + "; ".join(issues))
    return {
        "registered_dispatchers": registered,
        "running_dispatchers": running,
        "pending_requests": pending,
        "inflight_requests": inflight,
        "endpoint_replicas": replica_count,
        "dispatch_execution_limit": execution_limit,
    }


def preflight_dispatch_runtime(
    config_path: Path,
    fabric_group_id: str,
    required_execution_capacity: int | None = None,
) -> dict[str, int]:
    config = json.loads(config_path.expanduser().read_text())
    base_url = config.get("api_base_url")
    api_key = config.get("api_key")
    if not isinstance(base_url, str) or not base_url:
        raise ValueError("stress config api_base_url is required")
    if not isinstance(api_key, str) or not api_key:
        raise ValueError("stress config api_key is required")

    query = urllib.parse.urlencode({"fabric_group_id": fabric_group_id, "limit": 10})
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/llm-dispatch/runtime?{query}",
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        payload = json.load(response)
    if not isinstance(payload, dict) or payload.get("status") != "OK":
        raise RuntimeError("gateway returned an invalid dispatch runtime response")
    snapshot = payload.get("result")
    if not isinstance(snapshot, dict):
        raise RuntimeError("gateway dispatch runtime snapshot is unavailable")
    return validate_clean_dispatch_runtime(
        snapshot,
        required_execution_capacity=required_execution_capacity,
    )


def build_stresser_command(
    *,
    python: Path,
    config: Path,
    manifest: Path,
    request_template: Path,
    artifact_dir: Path,
    run_id: str,
    phase: Phase,
    routing_settle_timeout: int = 0,
    debug_sample_interval: int = 30,
    max_retained_jobs: int = 1000,
    max_metric_samples: int = 10000,
    runtime_mode: str = "queued-dispatch",
) -> list[str]:
    expected_outcome = expected_outcome_for(phase, runtime_mode)
    min_completion = "100" if expected_outcome == "completed" else "0"
    command = [
        str(python),
        "tools/stress/gateway_e2e_stresser.py",
        "--config",
        str(config),
        "--input-manifest",
        str(manifest),
        "--job-count",
        str(phase.jobs),
        "--run-id",
        run_id,
        "--job-name",
        "gen5_extract",
        "--planner",
        "mock_annotator_llm",
        "--required-executor",
        "annotator_llm",
        "--ref-type",
        "stress",
        "--project-id",
        "mock-annotator-llm-stress",
        "--request-template",
        str(request_template),
        "--fault-profile",
        str(phase.fault["profile"]),
        "--submit-rate",
        "6" if phase.name == "normal" else "3",
        "--submit-concurrency",
        "6" if phase.name == "normal" else "3",
        "--terminal-timeout",
        str(phase.terminal_timeout),
        "--min-submission-acceptance-pct",
        "100",
        "--min-terminal-completion-pct",
        min_completion,
        "--max-event-timeout-jobs",
        "0",
        "--max-open-jobs",
        "0",
        "--require-event-order",
        "--max-duplicate-terminal-events",
        "0",
        "--max-conflicting-terminal-events",
        "0",
        "--debug-sample-interval",
        str(debug_sample_interval),
        "--max-retained-jobs",
        str(max_retained_jobs),
        "--max-metric-samples",
        str(max_metric_samples),
        "--progress-interval",
        "2",
        "--job-jsonl",
        str(artifact_dir / f"{phase.name}-jobs.jsonl"),
        "--live-report",
        str(artifact_dir / f"{phase.name}-live.json"),
        "--report",
        str(artifact_dir / f"{phase.name}-final.json"),
    ]
    if routing_settle_timeout > 0:
        command.extend(["--routing-settle-timeout", str(routing_settle_timeout)])
    if phase.pool_counts is not None:
        command.extend(
            ["--input-counts", ",".join(str(phase.pool_counts[pool]) for pool in POOLS)]
        )
    return command


def set_fault(admin_url: str, settings: dict[str, object]) -> dict[str, Any]:
    payload = json.dumps({**settings, "resetCounters": True}).encode()
    request = urllib.request.Request(
        f"{admin_url.rstrip('/')}/fault-profile",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.load(response)


def get_fault_state(admin_url: str) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{admin_url.rstrip('/')}/fault-profile",
        headers={"Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.load(response)


def _provider_metrics(
    state: dict[str, Any], *, duration_seconds: float
) -> dict[str, object]:
    request_count = int(state.get("requestCount") or 0)
    return {
        "provider_concurrency_limit": state.get("maxConcurrentCalls"),
        "provider_processing_delay_ms": state.get("processingDelayMs"),
        "provider_peak_concurrency": state.get("peakConcurrentCalls"),
        "provider_waited_calls": state.get("capacityWaitCount"),
        "provider_wait_ms": state.get("capacityWaitMs"),
        "provider_calls": request_count,
        "provider_calls_per_second": (
            request_count / duration_seconds if duration_seconds > 0 else None
        ),
    }


def _dispatch_execution_limit(report: dict[str, Any]) -> int | None:
    routing = report.get("routing_qualification")
    if not isinstance(routing, dict):
        return None
    limits = [
        replica.get("execution_limit")
        for group in routing.get("endpoint_groups", [])
        if isinstance(group, dict)
        for replica in group.get("replicas", [])
        if isinstance(replica, dict)
        and replica.get("enabled") is True
        and type(replica.get("execution_limit")) is int
    ]
    return sum(limits) if limits else None


def _pool_latency_breakdown(
    report: dict[str, Any],
) -> dict[str, dict[str, int | float | None]]:
    jobs_by_pool: dict[str, list[dict[str, Any]]] = {pool: [] for pool in POOLS}
    for job in report.get("jobs", []):
        if not isinstance(job, dict):
            continue
        source = Path(str(job.get("source_path") or "")).name
        pool = next((name for name in POOLS if source.startswith(name)), None)
        if pool is not None:
            jobs_by_pool[pool].append(job)

    breakdown: dict[str, dict[str, int | float | None]] = {}
    for pool, jobs in jobs_by_pool.items():
        values: dict[str, int | float | None] = {"jobs": len(jobs)}
        for source_key, output_key in (
            ("queue_wait_ms", "queue_wait"),
            ("execution_ms", "execution"),
            ("end_to_end_ms", "end_to_end"),
        ):
            samples = [
                float(job[source_key])
                for job in jobs
                if isinstance(job.get(source_key), (int, float))
            ]
            values[f"{output_key}_p50_ms"] = (
                float(np.percentile(samples, 50)) if samples else None
            )
            values[f"{output_key}_p95_ms"] = (
                float(np.percentile(samples, 95)) if samples else None
            )
        breakdown[pool] = values
    return breakdown


def _validate_provider_capacity(
    *,
    state: dict[str, Any],
    report: dict[str, Any],
    runtime_mode: str,
    concurrency: int,
    processing_delay_ms: int,
) -> None:
    assert state.get("maxConcurrentCalls") == concurrency, (
        "AIMock provider concurrency configuration was not applied"
    )
    assert state.get("processingDelayMs") == processing_delay_ms, (
        "AIMock provider delay configuration was not applied"
    )
    peak = state.get("peakConcurrentCalls")
    assert type(peak) is int and 0 <= peak <= concurrency, (
        "AIMock provider concurrency telemetry is invalid"
    )
    if runtime_mode == "queued-dispatch":
        dispatch_limit = _dispatch_execution_limit(report)
        assert dispatch_limit is not None and dispatch_limit >= concurrency, (
            "queued dispatch capacity is lower than the shared provider limit: "
            f"dispatch={dispatch_limit}, provider={concurrency}"
        )


def _runtime_path_observation(report: dict[str, Any]) -> dict[str, object]:
    routing = report.get("routing_qualification")
    if not isinstance(routing, dict) or routing.get("available") is not True:
        return {"observed_runtime_mode": "unverified"}

    dispatcher_delta = routing.get("dispatcher_counters", {}).get("delta", {})
    claims = int(dispatcher_delta.get("claims") or 0)
    provider_starts = int(dispatcher_delta.get("provider_starts") or 0)
    dispatcher_completed = int(dispatcher_delta.get("completed") or 0)
    accepted = sum(
        int(pool.get("delta", {}).get("accepted") or 0)
        for pool in routing.get("drr", {}).values()
        if isinstance(pool, dict)
    )
    observed = (
        "queued-dispatch"
        if claims > 0 or provider_starts > 0 or dispatcher_completed > 0
        else "direct-batch"
    )
    return {
        "observed_runtime_mode": observed,
        "dispatch_claims": claims,
        "dispatch_provider_starts": provider_starts,
        "dispatch_completed": dispatcher_completed,
        "admission_accepted": accepted,
    }


def _circuit_open_observed(report: dict[str, Any]) -> bool:
    samples = report.get("debug_sampling", {}).get("samples", [])
    return any(
        replica.get("circuit") in {"open", "half_open"}
        for sample in samples
        if isinstance(sample, dict)
        for group in (sample.get("llm_dispatch") or {}).get("endpoint_groups", [])
        if isinstance(group, dict)
        for replica in group.get("replicas", [])
        if isinstance(replica, dict)
    )


def validate_report(
    phase: Phase,
    report: dict[str, Any],
    *,
    runtime_mode: str = "queued-dispatch",
) -> None:
    summary = report["summary"]
    observed = report["reliability"]["observed"]
    assert summary["submitted_jobs"] == phase.jobs, "submitted job count"
    assert summary["event_timeout_jobs"] == 0, "terminal event timeouts"
    assert observed["open_jobs"] == 0, "open jobs after drain"

    completed = summary["completed_jobs"]
    failed = summary["failed_jobs"]
    expected_outcome = expected_outcome_for(phase, runtime_mode)
    if expected_outcome == "completed":
        assert completed == phase.jobs and failed == 0, "successful terminal outcomes"
    elif expected_outcome == "failed":
        assert failed == phase.jobs and completed == 0, "failed terminal outcomes"
    elif expected_outcome == "mixed":
        assert completed + failed == phase.jobs, "mixed terminal outcomes"
        assert completed > 0 and failed > 0, "chaos must exercise success and failure"
    elif expected_outcome == "settled":
        assert completed + failed == phase.jobs, "settled terminal outcomes"
    else:
        raise ValueError(f"Unknown expected outcome: {expected_outcome}")

    if runtime_mode == "direct-batch":
        observation = _runtime_path_observation(report)
        observed = observation["observed_runtime_mode"]
        assert observed == "direct-batch", (
            f"direct-batch benchmark invalid: requested direct-batch but ran {observed}; "
            f"dispatcher claims={observation.get('dispatch_claims', 'unavailable')}, "
            f"provider starts={observation.get('dispatch_provider_starts', 'unavailable')}, "
            f"dispatcher completed={observation.get('dispatch_completed', 'unavailable')}, "
            f"admission accepted={observation.get('admission_accepted', 'unavailable')}"
        )
        return
    if runtime_mode != "queued-dispatch":
        return

    routing = report["routing_qualification"]
    if routing.get("counters_settled") is False:
        latest = routing.get("latest_observation")
        error = latest.get("error") if isinstance(latest, dict) else None
        detail = f": {error}" if error else ""
        raise AssertionError(f"runtime counters did not settle{detail}")
    assert routing["policy"]["synchronized"] is True, "policy synchronization"
    assert routing["projection"]["pending_at_end"] == 0, "pending projections"

    expected_by_pool = phase.pool_counts or expected_pool_counts(phase.jobs)
    for pool, expected in expected_by_pool.items():
        delta = routing["drr"][pool]["delta"]
        assert delta["accepted"] == expected, f"{pool} accepted"
        if expected_outcome == "completed":
            assert delta["completed"] == expected, f"{pool} completed"

    dispatcher_delta = routing.get("dispatcher_counters", {}).get("delta", {})
    retry_count = int(dispatcher_delta.get("retries") or 0)
    if phase.retry_evidence == "prewrite_refund":
        refunded_cost = sum(
            int(pool["delta"].get("refunded_cost") or 0)
            for pool in routing["drr"].values()
        )
        assert refunded_cost > 0, "prewrite retry evidence"
    elif phase.retry_evidence == "deferred_retry":
        assert retry_count > 0, "deferred retry evidence"
    elif phase.retry_evidence == "no_retry":
        assert retry_count == 0, "ambiguous hang retry"
    elif phase.retry_evidence == "max_output_terminal":
        assert retry_count == 0, "max-output retry"
        assert int(dispatcher_delta.get("completed") or 0) > 0, (
            "max-output completion evidence"
        )
        assert int(dispatcher_delta.get("execution_errors") or 0) == 0, (
            "max-output transport errors"
        )
        assert int(dispatcher_delta.get("remote_outcomes_failed") or 0) == 0, (
            "max-output unknown outcomes"
        )
    elif phase.retry_evidence == "repetition_recovery":
        assert retry_count == 0, "repetition durable retry"
        assert int(dispatcher_delta.get("repetition_retries") or 0) > 0, (
            "repetition recovery evidence"
        )
    elif phase.retry_evidence == "repetition_terminal":
        assert retry_count == 0, "terminal repetition durable retry"
        repetition_retries = int(dispatcher_delta.get("repetition_retries") or 0)
        assert repetition_retries > 0, "terminal repetition evidence"
        assert int(dispatcher_delta.get("completed") or 0) == repetition_retries, (
            "terminal repetition completion evidence"
        )
        assert (
            int(dispatcher_delta.get("execution_errors") or 0) == repetition_retries
        ), "terminal repetition execution error accounting"
        assert int(dispatcher_delta.get("provider_starts") or 0) == (
            repetition_retries * 2
        ), "terminal repetition provider attempt accounting"
        assert int(dispatcher_delta.get("remote_outcomes_failed") or 0) == 0, (
            "terminal repetition unknown outcomes"
        )


def _result_row(phase: Phase, report: dict[str, Any]) -> dict[str, object]:
    summary = report["summary"]
    latency = report["latency_stats_ms"]
    routing = report.get("routing_qualification")
    if not isinstance(routing, dict):
        routing = {}
    retries = routing.get("dispatcher_counters", {}).get("delta", {}).get("retries")
    refunded_cost = sum(
        int(pool.get("delta", {}).get("refunded_cost") or 0)
        for pool in routing.get("drr", {}).values()
    )
    duration = float(summary["duration_seconds"])
    completed = int(summary["completed_jobs"])
    return {
        "phase": phase.name,
        "jobs": phase.jobs,
        "completed": completed,
        "failed": summary["failed_jobs"],
        "unresolved": report.get("reliability", {})
        .get("observed", {})
        .get("open_jobs"),
        "retries": retries,
        "refunded_cost": refunded_cost,
        "duration_seconds": duration,
        "throughput_jobs_per_second": completed / duration if duration > 0 else 0.0,
        "circuit_open_observed": _circuit_open_observed(report),
        "queue_wait_p50_ms": latency["queue_wait"]["p50"],
        "queue_wait_p95_ms": latency["queue_wait"]["p95"],
        "execution_p50_ms": latency["execution"]["p50"],
        "execution_p95_ms": latency["execution"]["p95"],
        "end_to_end_p50_ms": latency["end_to_end"]["p50"],
        "end_to_end_p95_ms": latency["end_to_end"]["p95"],
        "dispatch_execution_limit": _dispatch_execution_limit(report),
        "pool_latency": _pool_latency_breakdown(report),
        **_runtime_path_observation(report),
    }


def _display_number(value: object, decimals: int = 2) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:,.{decimals}f}"
    if isinstance(value, int):
        return f"{value:,}"
    return html.escape(str(value))


def _pool_latency_html(row: dict[str, object]) -> str:
    breakdown = row.get("pool_latency")
    if not isinstance(breakdown, dict):
        return ""
    lines = []
    for pool in POOLS:
        metrics = breakdown.get(pool)
        if not isinstance(metrics, dict) or not metrics.get("jobs"):
            continue
        lines.append(
            f"<div>{html.escape(pool)}: {_display_number(metrics.get('jobs'))} jobs · "
            f"queue p95 {_display_number(metrics.get('queue_wait_p95_ms'))} ms · "
            f"execution p95 {_display_number(metrics.get('execution_p95_ms'))} ms · "
            f"E2E p95 {_display_number(metrics.get('end_to_end_p95_ms'))} ms</div>"
        )
    if not lines:
        return ""
    return (
        '<details><summary>Per-pool latency</summary>' + "".join(lines) + "</details>"
    )


def _metric_guide_html(*, comparison: bool = False) -> str:
    title = "How to read this comparison" if comparison else "How to read these metrics"
    comparison_item = ""
    if comparison:
        comparison_item = (
            "<li><strong>Comparison:</strong> Higher throughput is better; lower latency is better. "
            "The table names the better result instead of requiring you to interpret a signed delta. "
            "Performance is ranked only when both modes completed every job with matching execution "
            "paths and provider capacity.</li>"
        )
    return (
        '<details class="metric-guide" open>'
        f"<summary>{title}</summary>"
        "<ul>"
        "<li><strong>Throughput:</strong> total system output over the full run, in completed jobs per second.</li>"
        "<li><strong>P50:</strong> the median. Half of requests completed faster and half completed slower.</li>"
        "<li><strong>P95:</strong> 95% of requests completed at or below this value; the slowest 5% took longer.</li>"
        "<li><strong>Latency:</strong> queue wait ends when execution starts; execution ends at the terminal result; "
        "end-to-end covers submission through the terminal result.</li>"
        "<li><strong>Throughput versus latency:</strong> Higher throughput can coexist with higher per-request latency "
        "when more work overlaps and keeps capacity busy while requests spend longer waiting or competing for resources.</li>"
        "<li>P95 values from different columns cannot be added because each percentile may describe a different request.</li>"
        f"{comparison_item}"
        "</ul></details>"
    )


def write_html_report(path: Path, summary: dict[str, object]) -> None:
    rows = summary.get("phases", [])
    if not isinstance(rows, list):
        raise ValueError("summary phases must be a list")

    status = str(summary.get("status", "unknown"))
    refresh = '<meta http-equiv="refresh" content="3">' if status == "running" else ""
    totals = {
        key: sum(int(row.get(key, 0) or 0) for row in rows if isinstance(row, dict))
        for key in ("jobs", "completed", "failed", "unresolved")
    }
    phase_rows: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        phase = str(row.get("phase", "unknown"))
        phase_name = html.escape(phase)
        links = " · ".join(
            (
                f'<a href="{phase_name}-live.json">live JSON</a>',
                f'<a href="{phase_name}-final.json">final JSON</a>',
                f'<a href="{phase_name}-jobs.jsonl">jobs JSONL</a>',
            )
        )
        phase_rows.append(
            "<tr>"
            f"<td><strong>{phase_name}</strong><br><span>{html.escape(str(row.get('expected_outcome', '—')))}</span></td>"
            f"<td><span class=\"badge {html.escape(str(row.get('status', 'pending')))}\">{html.escape(str(row.get('status', 'pending')))}</span></td>"
            f"<td>{_display_number(row.get('jobs'))}</td>"
            f"<td>{_display_number(row.get('completed'))}</td>"
            f"<td>{_display_number(row.get('failed'))}</td>"
            f"<td>{_display_number(row.get('unresolved'))}</td>"
            f"<td>{_display_number(row.get('retries'))}</td>"
            f"<td>{_display_number(row.get('refunded_cost'))}</td>"
            f"<td>{_display_number(row.get('duration_seconds'))}</td>"
            f"<td>{_display_number(row.get('throughput_jobs_per_second'))}</td>"
            f"<td>{_display_number(row.get('queue_wait_p95_ms'))}</td>"
            f"<td>{_display_number(row.get('execution_p95_ms'))}</td>"
            f"<td>{_display_number(row.get('end_to_end_p50_ms'))}</td>"
            f"<td>{_display_number(row.get('end_to_end_p95_ms'))}</td>"
            f"<td>{_display_number(row.get('provider_concurrency_limit'))}</td>"
            f"<td>{_display_number(row.get('provider_peak_concurrency'))}</td>"
            f"<td>{_display_number(row.get('provider_waited_calls'))}</td>"
            f"<td>{_display_number(row.get('provider_calls_per_second'))}</td>"
            f"<td>{html.escape(str(row.get('observed_runtime_mode', 'unverified')))}"
            f"{_pool_latency_html(row)}</td>"
            f"<td class=\"links\">{links}</td>"
            "</tr>"
        )

    error = summary.get("error")
    error_block = ""
    if error:
        error_block = (
            '<section class="error"><h2>Run error</h2>'
            f"<pre>{html.escape(str(error))}</pre></section>"
        )
    provider = summary.get("provider")
    provider = provider if isinstance(provider, dict) else {}
    generated_at = datetime.now(timezone.utc).isoformat()
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
{refresh}
<title>LLM dispatch stress report</title>
<style>
:root {{ color-scheme: light dark; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }}
body {{ margin: 0; background: #f4f6fb; color: #172033; }}
main {{ max-width: 1500px; margin: 0 auto; padding: 32px; }}
header {{ display: flex; justify-content: space-between; gap: 24px; align-items: start; margin-bottom: 24px; }}
h1 {{ margin: 0 0 8px; font-size: 28px; }} h2 {{ font-size: 18px; }}
.muted, td span {{ color: #667085; font-size: 13px; }}
.badge {{ display: inline-block; border-radius: 999px; padding: 4px 10px; background: #e5e7eb; font-size: 12px; font-weight: 700; }}
.badge.passed {{ background: #d1fadf; color: #05603a; }} .badge.failed {{ background: #fee4e2; color: #b42318; }}
.badge.running {{ background: #dbeafe; color: #175cd3; }} .badge.interrupted {{ background: #fef0c7; color: #93370d; }}
.metrics {{ display: grid; grid-template-columns: repeat(3, minmax(160px, 1fr)); gap: 12px; margin-bottom: 24px; }}
.metric, .table-wrap, .error {{ background: #fff; border: 1px solid #dfe3eb; border-radius: 10px; box-shadow: 0 1px 2px rgba(16,24,40,.05); }}
.metric {{ padding: 18px; }} .metric strong {{ display: block; font-size: 28px; margin-top: 4px; }}
.metric-guide {{ margin: 0 0 24px; padding: 14px 18px; background: #fff; border: 1px solid #dfe3eb; border-radius: 10px; }}
.metric-guide summary {{ cursor: pointer; font-weight: 700; }} .metric-guide ul {{ margin: 12px 0 0; padding-left: 20px; }} .metric-guide li + li {{ margin-top: 6px; }}
.table-wrap {{ overflow-x: auto; }} table {{ width: 100%; border-collapse: collapse; }}
th, td {{ padding: 13px 14px; border-bottom: 1px solid #eaecf0; text-align: right; white-space: nowrap; }}
th {{ background: #f8fafc; color: #475467; font-size: 11px; text-transform: uppercase; letter-spacing: .04em; }}
th:first-child, td:first-child, th:last-child, td:last-child {{ text-align: left; }}
.links a {{ color: #4f46e5; text-decoration: none; }} .error {{ margin-top: 20px; padding: 18px; border-color: #fda29b; }}
pre {{ margin: 0; white-space: pre-wrap; overflow-wrap: anywhere; }} footer {{ margin-top: 18px; color: #667085; font-size: 12px; }}
@media (prefers-color-scheme: dark) {{ body {{ background: #111827; color: #f9fafb; }} .metric, .metric-guide, .table-wrap, .error {{ background: #1f2937; border-color: #374151; }} th {{ background: #111827; color: #d1d5db; }} th, td {{ border-color: #374151; }} .muted, td span {{ color: #9ca3af; }} }}
</style>
</head>
<body><main>
<header><div><h1>{html.escape(str(summary.get('suite_id', 'LLM dispatch stress suite')))}</h1>
<div class="muted">{html.escape(str(summary.get('runtime_mode', 'unknown')))} · {html.escape(str(summary.get('manifest', 'manifest unavailable')))}</div>
<div class="muted">Provider cap {_display_number(provider.get('max_concurrent_calls'))} · delay {_display_number(provider.get('processing_delay_ms'))} ms</div></div>
<span class="badge {html.escape(status)}">{html.escape(status)}</span></header>
<section class="metrics">
<div class="metric"><span class="muted">Configured jobs</span><strong>{totals['jobs']:,}</strong></div>
<div class="metric"><span class="muted">Completed</span><strong>{totals['completed']:,}</strong></div>
<div class="metric"><span class="muted">Failed</span><strong>{totals['failed']:,}</strong></div>
</section>
{_metric_guide_html()}
<section class="table-wrap"><table><thead><tr>
<th>Phase</th><th>Status</th><th>Jobs</th><th>Completed</th><th>Failed</th><th>Unresolved</th><th>Retries</th><th>Refunded cost</th><th>Duration s</th><th>Jobs/s</th><th>Queue p95 ms</th><th>Execution p95 ms</th><th>E2E p50 ms</th><th>E2E p95 ms</th><th>Provider cap</th><th>Provider peak</th><th>Calls waited</th><th>Provider calls/s</th><th>Execution path</th><th>Artifacts</th>
</tr></thead><tbody>{''.join(phase_rows)}</tbody></table></section>
{error_block}
<footer>Generated {html.escape(generated_at)} UTC. Running reports refresh every three seconds.</footer>
</main></body></html>
"""
    path.write_text(document, encoding="utf-8")


def _write_markdown(
    path: Path, runtime_mode: str, rows: list[dict[str, object]]
) -> None:
    lines = [
        f"# LLM dispatch stress suite: {runtime_mode}",
        "",
        "| Phase | Status | Jobs | Completed | Failed | Unresolved | Retries | Refunded cost | Duration (s) | E2E p50 (ms) | E2E p95 (ms) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {phase} | {status} | {jobs} | {completed} | {failed} | {unresolved} | {retries} | "
            "{refunded_cost} | {duration} | "
            "{e2e_p50} | {e2e_p95} |".format(
                phase=row.get("phase", "unknown"),
                status=row.get("status", "pending"),
                jobs=_display_number(row.get("jobs")),
                completed=_display_number(row.get("completed")),
                failed=_display_number(row.get("failed")),
                unresolved=_display_number(row.get("unresolved")),
                retries=_display_number(row.get("retries")),
                refunded_cost=_display_number(row.get("refunded_cost")),
                duration=_display_number(row.get("duration_seconds")),
                e2e_p50=_display_number(row.get("end_to_end_p50_ms")),
                e2e_p95=_display_number(row.get("end_to_end_p95_ms")),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_run_index(root: Path) -> None:
    runs: list[dict[str, object]] = []
    suites: dict[str, dict[str, dict[str, object]]] = {}
    for comparison_path in root.glob("*/comparison.json"):
        try:
            summary = json.loads(comparison_path.read_text())
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
        run_dir = comparison_path.parent.name
        suite_id = str(summary.get("suite_id", run_dir))
        runtime_mode = str(summary.get("runtime_mode", "unknown"))
        record: dict[str, object] = {
            "mtime": comparison_path.stat().st_mtime,
            "run_dir": run_dir,
            "summary": summary,
            "suite_id": suite_id,
            "runtime_mode": runtime_mode,
        }
        runs.append(record)
        mode_runs = suites.setdefault(suite_id, {})
        previous = mode_runs.get(runtime_mode)
        if previous is None or float(record["mtime"]) > float(previous["mtime"]):
            mode_runs[runtime_mode] = record

    def phase_map(record: dict[str, object]) -> dict[str, dict[str, object]]:
        summary = record["summary"]
        assert isinstance(summary, dict)
        phases = summary.get("phases")
        if not isinstance(phases, list):
            return {}
        return {
            str(row.get("phase", "unknown")): row
            for row in phases
            if isinstance(row, dict)
        }

    def mode_cell(record: dict[str, object], row: dict[str, object]) -> str:
        run_dir = html.escape(str(record["run_dir"]))
        status = html.escape(str(row.get("status", "unknown")))
        expected_mode = str(record["runtime_mode"])
        observed_mode = str(row.get("observed_runtime_mode", "unverified"))
        if observed_mode == expected_mode:
            path_status = f"Execution path verified: {html.escape(observed_mode)}"
        else:
            display_modes = {
                "direct-batch": "Direct batch",
                "queued-dispatch": "Queued dispatch",
                "unverified": "an unverified path",
            }
            path_status = (
                '<span class="path-warning">Invalid baseline: requested '
                f"{html.escape(display_modes.get(expected_mode, expected_mode))}, "
                "but this run used "
                f"{html.escape(display_modes.get(observed_mode, observed_mode))}</span>"
            )
        duration = row.get("duration_seconds")
        completed = row.get("completed")
        completion_rate = (
            completed / duration
            if isinstance(completed, (int, float))
            and isinstance(duration, (int, float))
            and duration > 0
            else row.get("throughput_jobs_per_second")
        )
        circuit_status = (
            '<div class="circuit-warning">Endpoint circuit opened during run</div>'
            if row.get("circuit_open_observed") is True
            else ""
        )
        return (
            f'<span class="badge {status}">{status}</span>'
            f'<div>{_display_number(row.get("completed"))}/'
            f'{_display_number(row.get("jobs"))} completed</div>'
            f'<div>{_display_number(completion_rate)} completed jobs/s</div>'
            f'<div>queue p95 {_display_number(row.get("queue_wait_p95_ms"))} ms</div>'
            f'<div>execution p95 {_display_number(row.get("execution_p95_ms"))} ms</div>'
            f'<div>E2E p95 {_display_number(row.get("end_to_end_p95_ms"))} ms</div>'
            f'<div>provider cap {_display_number(row.get("provider_concurrency_limit"))} · '
            f'peak {_display_number(row.get("provider_peak_concurrency"))} · '
            f'{_display_number(row.get("provider_calls_per_second"))} calls/s</div>'
            f"{_pool_latency_html(row)}"
            f"{circuit_status}"
            f"<div>{path_status}</div>"
            f'<div class="links"><a href="{run_dir}/report.html">Report</a> · '
            f'<a href="{run_dir}/comparison.json">JSON</a></div>'
        )

    paired_runs: list[tuple[str, str, dict[str, object], dict[str, object]]] = []
    used_run_dirs: set[str] = set()
    for suite_id, modes in suites.items():
        queued = modes.get("queued-dispatch")
        direct = modes.get("direct-batch")
        if queued is None or direct is None:
            continue
        paired_runs.append((suite_id, "Suite ID match", queued, direct))
        used_run_dirs.update((str(queued["run_dir"]), str(direct["run_dir"])))

    def workload_signature(record: dict[str, object]) -> str | None:
        summary = record["summary"]
        assert isinstance(summary, dict)
        phases = summary.get("phases")
        if not isinstance(phases, list):
            return None
        comparable = []
        for row in phases:
            if not isinstance(row, dict):
                return None
            comparable.append(
                {
                    "phase": row.get("phase"),
                    "expected_outcome": row.get("expected_outcome"),
                    "jobs": row.get("jobs"),
                    "pool_counts": row.get("pool_counts"),
                }
            )
        family = re.sub(r"-\d{8}-\d{6}$", "", str(record["suite_id"]))
        return json.dumps(
            {
                "family": family,
                "phases": comparable,
                "provider": summary.get("provider"),
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    def capacity_match(
        queued_row: dict[str, object], direct_row: dict[str, object]
    ) -> tuple[bool, str]:
        queued_limit = queued_row.get("provider_concurrency_limit")
        direct_limit = direct_row.get("provider_concurrency_limit")
        queued_delay = queued_row.get("provider_processing_delay_ms")
        direct_delay = direct_row.get("provider_processing_delay_ms")
        dispatch_limit = queued_row.get("dispatch_execution_limit")
        if not all(
            type(value) is int
            for value in (
                queued_limit,
                direct_limit,
                queued_delay,
                direct_delay,
                dispatch_limit,
            )
        ):
            return False, "Capacity unverified"
        if queued_limit != direct_limit or queued_delay != direct_delay:
            return (
                False,
                "Capacity mismatch: "
                f"queued {queued_limit} calls / {queued_delay} ms, "
                f"direct {direct_limit} calls / {direct_delay} ms",
            )
        if dispatch_limit < queued_limit:
            return (
                False,
                "Capacity mismatch: queued dispatch cap "
                f"{dispatch_limit} is below provider cap {queued_limit}",
            )
        return (
            True,
            f"Capacity match: {queued_limit} provider calls, {queued_delay} ms delay",
        )

    workload_groups: dict[str, dict[str, list[dict[str, object]]]] = {}
    for record in runs:
        if str(record["run_dir"]) in used_run_dirs:
            continue
        signature = workload_signature(record)
        mode = str(record["runtime_mode"])
        if signature is None or mode not in {"queued-dispatch", "direct-batch"}:
            continue
        workload_groups.setdefault(signature, {}).setdefault(mode, []).append(record)
    for modes in workload_groups.values():
        queued_runs = sorted(
            modes.get("queued-dispatch", []),
            key=lambda record: float(record["mtime"]),
            reverse=True,
        )
        direct_runs = sorted(
            modes.get("direct-batch", []),
            key=lambda record: float(record["mtime"]),
            reverse=True,
        )
        for queued, direct in zip(queued_runs, direct_runs, strict=False):
            label = f'{queued["suite_id"]} ↔ {direct["suite_id"]}'
            paired_runs.append((label, "Workload match", queued, direct))

    comparison_rows: list[tuple[float, str]] = []
    for suite_label, match_kind, queued, direct in paired_runs:
        queued_phases = phase_map(queued)
        direct_phases = phase_map(direct)
        phase_names = list(queued_phases)
        phase_names.extend(name for name in direct_phases if name not in queued_phases)
        newest = max(float(queued["mtime"]), float(direct["mtime"]))
        for phase_name in phase_names:
            queued_row = queued_phases.get(phase_name)
            direct_row = direct_phases.get(phase_name)
            if queued_row is None or direct_row is None:
                continue
            same_jobs = queued_row.get("jobs") == direct_row.get("jobs")
            queued_duration = queued_row.get("duration_seconds")
            direct_duration = direct_row.get("duration_seconds")
            queued_completed = queued_row.get("completed")
            direct_completed = direct_row.get("completed")
            queued_rate = (
                queued_completed / queued_duration
                if isinstance(queued_completed, (int, float))
                and isinstance(queued_duration, (int, float))
                and queued_duration > 0
                else queued_row.get("throughput_jobs_per_second")
            )
            direct_rate = (
                direct_completed / direct_duration
                if isinstance(direct_completed, (int, float))
                and isinstance(direct_duration, (int, float))
                and direct_duration > 0
                else direct_row.get("throughput_jobs_per_second")
            )
            queued_p95 = queued_row.get("end_to_end_p95_ms")
            direct_p95 = direct_row.get("end_to_end_p95_ms")
            paths_verified = (
                queued_row.get("observed_runtime_mode") == "queued-dispatch"
                and direct_row.get("observed_runtime_mode") == "direct-batch"
            )
            capacity_verified, capacity_status = capacity_match(queued_row, direct_row)
            queued_outcome = (
                queued_row.get("completed"),
                queued_row.get("failed"),
                queued_row.get("unresolved") or 0,
            )
            direct_outcome = (
                direct_row.get("completed"),
                direct_row.get("failed"),
                direct_row.get("unresolved") or 0,
            )
            same_outcome = queued_outcome == direct_outcome
            all_completed = (
                same_jobs
                and queued_completed == queued_row.get("jobs")
                and direct_completed == direct_row.get("jobs")
                and queued_row.get("failed") == 0
                and direct_row.get("failed") == 0
                and queued_outcome[2] == 0
                and direct_outcome[2] == 0
            )
            outcome_result = (
                "Same terminal outcome" if same_outcome else "Different outcomes"
            )
            performance_comparable = (
                same_jobs
                and same_outcome
                and all_completed
                and paths_verified
                and capacity_verified
            )
            if (
                performance_comparable
                and isinstance(queued_rate, (int, float))
                and isinstance(direct_rate, (int, float))
            ):
                if queued_rate == direct_rate:
                    throughput_result = "Tie"
                elif queued_rate > direct_rate and direct_rate > 0:
                    throughput_result = (
                        f"Queued {(queued_rate / direct_rate - 1) * 100:,.2f}% higher"
                    )
                elif queued_rate > direct_rate:
                    throughput_result = "Queued higher"
                elif queued_rate > 0:
                    throughput_result = (
                        f"Direct {(direct_rate / queued_rate - 1) * 100:,.2f}% higher"
                    )
                else:
                    throughput_result = "Direct higher"
            else:
                throughput_result = "Not comparable"
            if (
                performance_comparable
                and isinstance(queued_p95, (int, float))
                and isinstance(direct_p95, (int, float))
            ):
                if queued_p95 == direct_p95:
                    latency_result = "Tie"
                elif queued_p95 < direct_p95:
                    latency_result = f"Queued {direct_p95 - queued_p95:,.2f} ms lower"
                else:
                    latency_result = f"Direct {queued_p95 - direct_p95:,.2f} ms lower"
            else:
                latency_result = "Not comparable"
            comparison_rows.append(
                (
                    newest,
                    "<tr>"
                    f"<td><strong>{html.escape(suite_label)}</strong>"
                    f'<div class="match-kind">{html.escape(match_kind)}</div>'
                    f'<div class="match-kind">{html.escape(capacity_status)}</div></td>'
                    f"<td>{html.escape(phase_name)}</td>"
                    f"<td>{mode_cell(queued, queued_row)}</td>"
                    f"<td>{mode_cell(direct, direct_row)}</td>"
                    f"<td>{outcome_result}</td>"
                    f"<td>{throughput_result}</td>"
                    f"<td>{latency_result}</td>"
                    "</tr>",
                )
            )

    comparison_body = "".join(row for _, row in sorted(comparison_rows, reverse=True))
    if not comparison_body:
        comparison_body = (
            '<tr><td colspan="7" class="empty">Run the same suite ID once with '
            'each runtime mode to create a comparison.</td></tr>'
        )

    run_rows: list[tuple[float, str]] = []
    for record in runs:
        summary = record["summary"]
        assert isinstance(summary, dict)
        status = html.escape(str(summary.get("status", "unknown")))
        run_dir = html.escape(str(record["run_dir"]))
        run_rows.append(
            (
                float(record["mtime"]),
                "<tr>"
                f'<td><a href="{run_dir}/report.html">'
                f'{html.escape(str(record["suite_id"]))}</a></td>'
                f'<td>{html.escape(str(record["runtime_mode"]))}</td>'
                f'<td><span class="badge {status}">{status}</span></td>'
                f'<td><a href="{run_dir}/comparison.json">JSON</a></td>'
                "</tr>",
            )
        )
    all_runs_body = "".join(row for _, row in sorted(run_rows, reverse=True))
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="5"><title>LLM dispatch stress runs</title>
<style>body{{font-family:Inter,system-ui,sans-serif;margin:0;background:#f4f6fb;color:#172033}}main{{max-width:1500px;margin:auto;padding:32px}}h2{{margin-top:30px}}.metric-guide{{margin:20px 0;padding:14px 18px;background:#fff;border:1px solid #dfe3eb;border-radius:10px}}.metric-guide summary{{cursor:pointer;font-weight:700}}.metric-guide ul{{margin:12px 0 0;padding-left:20px}}.metric-guide li+li{{margin-top:6px}}.table-wrap{{overflow-x:auto;background:#fff;border:1px solid #dfe3eb;border-radius:10px}}table{{width:100%;border-collapse:collapse}}th,td{{padding:14px;border-bottom:1px solid #eaecf0;text-align:left;vertical-align:top}}th{{background:#f8fafc;color:#475467;font-size:12px;text-transform:uppercase;white-space:nowrap}}td div{{margin-top:5px;white-space:nowrap}}a{{color:#4f46e5;text-decoration:none}}.links{{margin-top:8px}}.empty{{color:#667085;text-align:center;padding:28px}}.path-warning,.circuit-warning{{color:#b42318;font-weight:700}}.badge{{display:inline-block;border-radius:999px;padding:4px 10px;background:#e5e7eb;font-size:12px;font-weight:700}}.passed{{background:#d1fadf;color:#05603a}}.failed{{background:#fee4e2;color:#b42318}}.running{{background:#dbeafe;color:#175cd3}}.interrupted{{background:#fef0c7;color:#93370d}}@media(prefers-color-scheme:dark){{body{{background:#111827;color:#f9fafb}}.metric-guide,.table-wrap{{background:#1f2937;border-color:#374151}}th{{background:#111827;color:#d1d5db}}th,td{{border-color:#374151}}}}</style>
</head><body><main><h1>LLM dispatch stress runs</h1><p>Runs are paired by suite ID or by the same timestamped workload definition. This index refreshes every five seconds.</p>
{_metric_guide_html(comparison=True)}
<h2>Runtime mode comparison</h2>
<div class="table-wrap"><table><thead><tr><th>Suite</th><th>Phase</th><th>Queued dispatch</th><th>Direct batch</th><th>Outcome</th><th>Throughput result</th><th>E2E p95 result</th></tr></thead><tbody>{comparison_body}</tbody></table></div>
<h2>All runs</h2>
<div class="table-wrap"><table><thead><tr><th>Suite</th><th>Runtime</th><th>Status</th><th>Raw data</th></tr></thead><tbody>{all_runs_body}</tbody></table></div>
</main></body></html>"""
    (root / "index.html").write_text(document, encoding="utf-8")


def _write_suite_reports(artifact_dir: Path, summary: dict[str, object]) -> Path:
    summary_path = artifact_dir / "comparison.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    rows = summary.get("phases", [])
    assert isinstance(rows, list)
    _write_markdown(
        artifact_dir / "comparison.md",
        str(summary.get("runtime_mode", "unknown")),
        rows,
    )
    write_html_report(artifact_dir / "report.html", summary)
    _write_run_index(artifact_dir.parent)
    return summary_path


def _partial_phase_result(phase: Phase, artifact_dir: Path) -> dict[str, object]:
    final_path = artifact_dir / f"{phase.name}-final.json"
    if final_path.is_file():
        try:
            return _result_row(phase, json.loads(final_path.read_text()))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass

    live_path = artifact_dir / f"{phase.name}-live.json"
    if not live_path.is_file():
        return {}
    try:
        live = json.loads(live_path.read_text())
        counts = live.get("counts", {})
        latency = live.get("latency_stats_ms", {})
        return {
            "completed": counts.get("completed_jobs"),
            "failed": counts.get("failed_jobs"),
            "unresolved": counts.get("open_jobs"),
            "duration_seconds": live.get("elapsed_seconds"),
            "throughput_jobs_per_second": live.get("throughput_jobs_per_second"),
            "queue_wait_p95_ms": latency.get("queue_wait", {}).get("p95"),
            "execution_p95_ms": latency.get("execution", {}).get("p95"),
            "end_to_end_p50_ms": latency.get("end_to_end", {}).get("p50"),
            "end_to_end_p95_ms": latency.get("end_to_end", {}).get("p95"),
        }
    except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
        return {}


def run(args: argparse.Namespace) -> Path:
    repo = Path(__file__).resolve().parents[2]
    suite_id = args.suite_id or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    artifact_dir = args.artifact_root.expanduser() / f"{suite_id}-{args.runtime_mode}"
    artifact_dir.mkdir(parents=True, exist_ok=False)
    manifest = args.input_manifest or write_fixtures(artifact_dir / "fixtures")
    pool_counts = getattr(args, "pool_counts", None)
    provider_concurrency = getattr(args, "provider_concurrency", 4)
    provider_delay_ms = getattr(args, "provider_delay_ms", 100)
    selected = set(args.phases.split(","))
    phases = [
        phase
        for phase in default_phases(
            normal_jobs=args.job_count,
            normal_terminal_timeout=getattr(args, "normal_terminal_timeout", None),
            normal_pool_counts=pool_counts,
        )
        if phase.name in selected
    ]
    if selected != {phase.name for phase in phases}:
        raise ValueError(
            f"Unknown phases: {', '.join(sorted(selected - {p.name for p in phases}))}"
        )

    rows: list[dict[str, object]] = [
        {
            "phase": phase.name,
            "status": "pending",
            "expected_outcome": expected_outcome_for(phase, args.runtime_mode),
            "jobs": phase.jobs,
            "pool_counts": phase.pool_counts or expected_pool_counts(phase.jobs),
            "completed": 0,
            "failed": 0,
        }
        for phase in phases
    ]
    summary: dict[str, object] = {
        "suite_id": suite_id,
        "runtime_mode": args.runtime_mode,
        "status": "running",
        "manifest": str(manifest),
        "pool_counts": pool_counts,
        "provider": {
            "max_concurrent_calls": provider_concurrency,
            "processing_delay_ms": provider_delay_ms,
        },
        "dispatch_preflight": {"status": "pending"},
        "phases": rows,
    }
    summary_path = _write_suite_reports(artifact_dir, summary)
    active_phase: Phase | None = None
    fault_profile_changed = False
    try:
        validate_fixture_manifest(manifest)
        if args.runtime_mode == "queued-dispatch":
            try:
                preflight = preflight_dispatch_runtime(
                    args.config,
                    args.fabric_group_id,
                    required_execution_capacity=provider_concurrency,
                )
            except Exception as exc:
                summary["dispatch_preflight"] = {
                    "status": "failed",
                    "error": str(exc),
                }
                _write_suite_reports(artifact_dir, summary)
                raise
            summary["dispatch_preflight"] = {"status": "passed", **preflight}
        else:
            summary["dispatch_preflight"] = {
                "status": "skipped",
                "reason": "direct-batch does not use the durable dispatch runtime",
            }
        _write_suite_reports(artifact_dir, summary)
        for index, phase in enumerate(phases):
            active_phase = phase
            rows[index]["status"] = "running"
            _write_suite_reports(artifact_dir, summary)
            set_fault(
                args.aimock_admin_url,
                {
                    **phase.fault,
                    "maxConcurrentCalls": provider_concurrency,
                    "processingDelayMs": provider_delay_ms,
                },
            )
            fault_profile_changed = True
            run_id = f"{suite_id}-{args.runtime_mode}-{phase.name}"
            command = build_stresser_command(
                python=args.python,
                config=args.config,
                manifest=manifest,
                request_template=args.request_template,
                artifact_dir=artifact_dir,
                run_id=run_id,
                phase=phase,
                routing_settle_timeout=(
                    30 if args.runtime_mode == "queued-dispatch" else 0
                ),
                debug_sample_interval=getattr(args, "debug_sample_interval", 30),
                max_retained_jobs=getattr(args, "max_retained_jobs", 1000),
                max_metric_samples=getattr(args, "max_metric_samples", 10000),
                runtime_mode=args.runtime_mode,
            )
            try:
                subprocess.run(command, cwd=repo, check=True)
            except subprocess.CalledProcessError as exc:
                final_path = artifact_dir / f"{phase.name}-final.json"
                if final_path.is_file():
                    failed_report = json.loads(final_path.read_text())
                    verification = failed_report.get("verification", {})
                    errors = verification.get("errors", [])
                    if isinstance(errors, list) and errors:
                        details = "; ".join(str(error) for error in errors)
                        raise RuntimeError(
                            f"Phase {phase.name} failed qualification: {details}"
                        ) from exc
                raise
            report = json.loads((artifact_dir / f"{phase.name}-final.json").read_text())
            validate_report(phase, report, runtime_mode=args.runtime_mode)
            provider_state = get_fault_state(args.aimock_admin_url)
            _validate_provider_capacity(
                state=provider_state,
                report=report,
                runtime_mode=args.runtime_mode,
                concurrency=provider_concurrency,
                processing_delay_ms=provider_delay_ms,
            )
            rows[index].update(_result_row(phase, report))
            rows[index].update(
                _provider_metrics(
                    provider_state,
                    duration_seconds=float(report["summary"]["duration_seconds"]),
                )
            )
            rows[index]["status"] = "passed"
            _write_suite_reports(artifact_dir, summary)
        summary["status"] = "passed"
    except (Exception, KeyboardInterrupt) as exc:
        run_status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        summary["status"] = run_status
        summary["error"] = f"{type(exc).__name__}: {exc}"
        if active_phase is not None:
            index = phases.index(active_phase)
            rows[index].update(_partial_phase_result(active_phase, artifact_dir))
            rows[index]["status"] = run_status
        _write_suite_reports(artifact_dir, summary)
        raise
    finally:
        if fault_profile_changed:
            try:
                set_fault(
                    args.aimock_admin_url,
                    {
                        "profile": "normal",
                        "timeoutMs": 180000,
                        "maxConcurrentCalls": 0,
                        "processingDelayMs": 0,
                    },
                )
            except Exception as reset_exc:
                logging.warning("Failed to reset AIMock fault profile: %s", reset_exc)

    _write_suite_reports(artifact_dir, summary)
    return summary_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime-mode",
        choices=("queued-dispatch", "direct-batch"),
        required=True,
        help="Label for the annotator runtime configuration under test",
    )
    parser.add_argument("--suite-id")
    parser.add_argument(
        "--phases",
        default=(
            "normal,unreachable,retryable,delay,hang,error,max-output,repetition,"
            "repetition-terminal,chaos"
        ),
    )
    parser.add_argument(
        "--job-count",
        type=int,
        default=None,
        help="Number of documents in the normal load phase (default: 60)",
    )
    parser.add_argument(
        "--pool-counts",
        type=parse_pool_counts,
        help=(
            "Exact normal-phase partition, for example "
            "document-small=80,document-medium=15,document-large=5"
        ),
    )
    parser.add_argument(
        "--normal-terminal-timeout",
        type=int,
        help=(
            "Seconds to wait after normal-phase submission for terminal events; "
            "defaults to max(1800, job-count * 8)"
        ),
    )
    parser.add_argument(
        "--debug-sample-interval",
        type=int,
        default=30,
        help="Seconds between gateway debug snapshots (default: 30)",
    )
    parser.add_argument("--max-retained-jobs", type=int, default=1000)
    parser.add_argument("--max-metric-samples", type=int, default=10000)
    parser.add_argument(
        "--provider-concurrency",
        type=int,
        default=4,
        help="Maximum concurrent AIMock calls shared by both runtime modes (default: 4)",
    )
    parser.add_argument(
        "--provider-delay-ms",
        type=int,
        default=100,
        help="Deterministic delay for every AIMock call (default: 100 ms)",
    )
    parser.add_argument("--input-manifest", type=Path)
    parser.add_argument(
        "--fabric-group-id",
        default="default",
        help="Runtime Fabric identifier used by the queued-dispatch preflight",
    )
    parser.add_argument(
        "--artifact-root", type=Path, default=Path("~/tmp/llm-dispatch-stress")
    )
    parser.add_argument(
        "--config", type=Path, default=repo / "tools/stress/gateway-e2e.config.json"
    )
    parser.add_argument(
        "--request-template",
        type=Path,
        default=repo / "tools/stress/mock_annotator_llm.invoke.json",
    )
    parser.add_argument("--python", type=Path, default=repo / ".venv/bin/python")
    parser.add_argument("--aimock-admin-url", default="http://127.0.0.1:4011")
    args = parser.parse_args(argv)
    if args.pool_counts is not None:
        configured_total = sum(args.pool_counts.values())
        if args.job_count is not None and args.job_count != configured_total:
            parser.error("--job-count must equal the total in --pool-counts")
        if args.input_manifest is not None:
            parser.error("--pool-counts uses the generated small/medium/large fixtures")
        args.job_count = configured_total
    elif args.job_count is None:
        args.job_count = 60
    if args.job_count < 1:
        parser.error("--job-count must be greater than zero")
    if args.normal_terminal_timeout is not None and args.normal_terminal_timeout < 1:
        parser.error("--normal-terminal-timeout must be greater than zero")
    if args.debug_sample_interval < 1:
        parser.error("--debug-sample-interval must be greater than zero")
    if args.max_retained_jobs < 1:
        parser.error("--max-retained-jobs must be greater than zero")
    if args.max_metric_samples < 1:
        parser.error("--max-metric-samples must be greater than zero")
    if args.provider_concurrency < 1:
        parser.error("--provider-concurrency must be greater than zero")
    if args.provider_delay_ms < 0:
        parser.error("--provider-delay-ms cannot be negative")
    return args


if __name__ == "__main__":
    print(run(parse_args()))
