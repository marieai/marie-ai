from __future__ import annotations

import html
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPORT_FORMAT_CHOICES = ("auto", "json", "html")


def build_latency_stats(
    values: List[float],
    percentile_fn,
) -> Optional[Dict[str, float]]:
    if not values:
        return None
    return {
        "count": float(len(values)),
        "min": min(values),
        "max": max(values),
        "avg": statistics.mean(values),
        "p50": percentile_fn(values, 50),
        "p95": percentile_fn(values, 95),
        "p99": percentile_fn(values, 99),
    }


def resolve_report_format(output_path: str, report_format: str) -> str:
    if report_format != "auto":
        return report_format

    suffix = Path(output_path).suffix.lower()
    if suffix in {".html", ".htm"}:
        return "html"
    return "json"


def write_text_atomically(output_path: str, content: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(content)
    tmp_path.replace(path)


def render_live_report(
    payload: Dict[str, Any], report_format: str, refresh_seconds: float
) -> str:
    if report_format == "html":
        return _render_live_report_html(payload, refresh_seconds)
    return json.dumps(payload, indent=2)


def render_final_report(payload: Dict[str, Any], report_format: str) -> str:
    if report_format == "html":
        return _render_final_report_html(payload)
    return json.dumps(payload, indent=2)


def render_dry_run_report(payload: Dict[str, Any], report_format: str) -> str:
    if report_format == "html":
        return _render_dry_run_report_html(payload)
    return json.dumps(payload, indent=2)


def _html_value(value: Any) -> str:
    if value is None:
        return '<span class="muted">n/a</span>'
    if isinstance(value, float):
        return html.escape(f"{value:.2f}")
    if isinstance(value, bool):
        return "true" if value else "false"
    return html.escape(str(value))


def _html_table(headers: List[str], rows: List[List[Any]]) -> str:
    head = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    body_rows = []
    for row in rows:
        cells = "".join(f"<td>{_html_value(value)}</td>" for value in row)
        body_rows.append(f"<tr>{cells}</tr>")
    body = "".join(body_rows) or (
        f'<tr><td colspan="{len(headers)}"><span class="muted">No rows</span></td></tr>'
    )
    return (
        '<div class="table-wrap"><table>'
        f"<thead><tr>{head}</tr></thead>"
        f"<tbody>{body}</tbody>"
        "</table></div>"
    )


def _html_cards(cards: List[Tuple[str, Any]]) -> str:
    rendered = []
    for label, value in cards:
        rendered.append(
            '<div class="card">'
            f'<div class="card-label">{html.escape(label)}</div>'
            f'<div class="card-value">{_html_value(value)}</div>'
            "</div>"
        )
    return '<div class="card-grid">' + "".join(rendered) + "</div>"


def _html_section(title: str, body: str) -> str:
    return f'<section class="section"><h2>{html.escape(title)}</h2>{body}</section>'


def _render_html_document(
    *,
    title: str,
    subtitle: Optional[str],
    body: str,
    refresh_seconds: Optional[float] = None,
) -> str:
    refresh_tag = ""
    if refresh_seconds is not None and refresh_seconds > 0:
        refresh_interval = max(1, int(math.ceil(refresh_seconds)))
        refresh_tag = f'<meta http-equiv="refresh" content="{refresh_interval}">'

    subtitle_html = (
        f'<p class="subtitle">{html.escape(subtitle)}</p>' if subtitle else ""
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  {refresh_tag}
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #f5f7fb;
      --panel: #ffffff;
      --panel-soft: #eef3ff;
      --border: #d6ddea;
      --text: #172033;
      --muted: #5f6c86;
      --accent: #0f62fe;
      --success: #1a7f37;
      --warning: #b54708;
      --danger: #b42318;
      --shadow: 0 10px 30px rgba(15, 35, 95, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      background: linear-gradient(180deg, #eef4ff 0%, var(--bg) 100%);
      color: var(--text);
    }}
    .page {{
      max-width: 1400px;
      margin: 0 auto;
      padding: 32px 24px 48px;
    }}
    .hero {{
      background: linear-gradient(135deg, #0f62fe 0%, #0530ad 100%);
      color: #fff;
      border-radius: 20px;
      padding: 28px 32px;
      box-shadow: var(--shadow);
      margin-bottom: 24px;
    }}
    .hero h1 {{
      margin: 0;
      font-size: 30px;
      line-height: 1.1;
      letter-spacing: -0.02em;
    }}
    .subtitle {{
      margin: 10px 0 0;
      color: rgba(255, 255, 255, 0.88);
      font-size: 15px;
    }}
    .section {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 20px 22px;
      margin-bottom: 20px;
      box-shadow: var(--shadow);
    }}
    .section h2 {{
      margin: 0 0 16px;
      font-size: 20px;
      letter-spacing: -0.01em;
    }}
    .card-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 14px;
    }}
    .card {{
      background: var(--panel-soft);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 14px 16px;
    }}
    .card-label {{
      color: var(--muted);
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      margin-bottom: 8px;
    }}
    .card-value {{
      font-size: 24px;
      font-weight: 700;
      line-height: 1.1;
    }}
    .table-wrap {{
      overflow-x: auto;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      text-align: left;
      padding: 10px 12px;
      border-bottom: 1px solid var(--border);
      vertical-align: top;
    }}
    th {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      background: #f8faff;
    }}
    .muted {{
      color: var(--muted);
    }}
    pre {{
      margin: 0;
      padding: 14px;
      overflow-x: auto;
      background: #0b1020;
      color: #dce8ff;
      border-radius: 14px;
      font-size: 12px;
      line-height: 1.45;
    }}
    .notice {{
      margin: 0 0 14px;
      color: var(--muted);
      font-size: 14px;
    }}
    @media (max-width: 720px) {{
      .page {{
        padding: 20px 14px 32px;
      }}
      .hero {{
        padding: 22px 20px;
      }}
      .hero h1 {{
        font-size: 24px;
      }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <header class="hero">
      <h1>{html.escape(title)}</h1>
      {subtitle_html}
    </header>
    {body}
  </main>
</body>
</html>
"""


def _render_live_report_html(payload: Dict[str, Any], refresh_seconds: float) -> str:
    counts = payload.get("counts", {})
    throughput = payload.get("throughput_jobs_per_second", {})
    run_health = payload.get("run_health", {})
    latency_stats = payload.get("latency_stats_ms", {})
    sla = payload.get("sla", {})
    recent_jobs = payload.get("recent_jobs", [])
    latest_debug_sample = payload.get("latest_debug_sample")
    verification_errors = payload.get("verification_errors", [])
    debug_sampling = payload.get("debug_sampling", {})
    recent_failures = [
        job
        for job in recent_jobs
        if job.get("terminal_status") in {"failed", "submit_failed"}
        or job.get("failure_reason")
    ]

    cards = _html_cards(
        [
            ("Status", payload.get("status")),
            ("Run Mode", payload.get("run_mode")),
            ("Target Submit TPS", run_health.get("target_submit_rate")),
            ("Created TPS", throughput.get("created")),
            ("Completed TPS", throughput.get("completed")),
            ("Open Jobs", run_health.get("open_jobs")),
            ("Inflight Jobs", run_health.get("inflight_jobs")),
            ("Pending Submit", run_health.get("pending_submit_jobs")),
            ("Terminal Success %", run_health.get("terminal_success_pct")),
            ("Submit Acceptance %", run_health.get("submit_acceptance_pct")),
            ("Completion %", run_health.get("completion_pct")),
            ("Completed", counts.get("completed_jobs")),
            (
                "Failed + Timed Out",
                (counts.get("failed_jobs") or 0)
                + (counts.get("event_timeout_jobs") or 0),
            ),
            ("Updated At", payload.get("updated_at")),
        ]
    )

    flow_metrics = _html_table(
        ["Metric", "Value"],
        [
            ["Created Jobs", counts.get("created_jobs")],
            ["Submitted Jobs", counts.get("submitted_jobs")],
            ["Completed Jobs", counts.get("completed_jobs")],
            ["Failed Jobs", counts.get("failed_jobs")],
            ["Submit Failed Jobs", counts.get("submit_failed_jobs")],
            ["Timed Out Jobs", counts.get("event_timeout_jobs")],
            ["Open Jobs", run_health.get("open_jobs")],
            ["Inflight Jobs", run_health.get("inflight_jobs")],
            ["Pending Submit Jobs", run_health.get("pending_submit_jobs")],
            ["Elapsed Seconds", payload.get("elapsed_seconds")],
        ],
    )

    latency_rows: List[List[Any]] = []
    for name, stats in latency_stats.items():
        if not isinstance(stats, dict):
            continue
        latency_rows.append(
            [
                name,
                int(stats.get("count", 0)),
                stats.get("avg"),
                stats.get("p50"),
                stats.get("p95"),
                stats.get("p99"),
                stats.get("max"),
            ]
        )
    latency_body = _html_table(
        ["Latency", "Count", "Avg", "P50", "P95", "P99", "Max"],
        latency_rows,
    )

    sla_rows: List[List[Any]] = []
    for label in ("soft", "hard"):
        summary = sla.get(label)
        if not isinstance(summary, dict):
            continue
        lateness_stats = summary.get("lateness_stats_ms") or {}
        sla_rows.append(
            [
                label,
                summary.get("configured_seconds"),
                summary.get("step_seconds"),
                summary.get("min_compliance_pct"),
                summary.get("configured_jobs"),
                summary.get("terminal_evaluated_jobs"),
                summary.get("met_jobs"),
                summary.get("missed_jobs"),
                summary.get("pending_jobs"),
                summary.get("overdue_open_jobs"),
                summary.get("terminal_compliance_pct"),
                lateness_stats.get("p95") if isinstance(lateness_stats, dict) else None,
            ]
        )
    sla_body = "<p class=\"notice\">No SLA configured for this run.</p>"
    if sla_rows:
        sla_body = _html_table(
            [
                "SLA",
                "Base Seconds",
                "Step Seconds",
                "Min Compliance %",
                "Configured Jobs",
                "Terminal Jobs",
                "Met",
                "Missed",
                "Pending",
                "Overdue Open",
                "Terminal Compliance %",
                "P95 Lateness ms",
            ],
            sla_rows,
        )

    if not debug_sampling.get("enabled"):
        debug_body = (
            "<p class=\"notice\">Debug sampling disabled. "
            "Set <code>--debug-sample-interval 5</code> or another positive value to enable it.</p>"
        )
    elif latest_debug_sample is None:
        debug_body = (
            "<p class=\"notice\">Waiting for the first debug sample.</p>"
            + _html_table(
                ["Field", "Value"],
                [
                    ["Enabled", debug_sampling.get("enabled")],
                    ["Interval Seconds", debug_sampling.get("interval_seconds")],
                    ["Endpoint", debug_sampling.get("endpoint")],
                ],
            )
        )
    else:
        debug_body = _html_table(
            ["Field", "Value"],
            [
                ["Enabled", debug_sampling.get("enabled")],
                ["Interval Seconds", debug_sampling.get("interval_seconds")],
                ["Endpoint", debug_sampling.get("endpoint")],
                ["Stage", latest_debug_sample.get("stage")],
                ["Captured At", latest_debug_sample.get("captured_at")],
                ["OK", latest_debug_sample.get("ok")],
                ["Status Code", latest_debug_sample.get("status_code")],
                ["Error", latest_debug_sample.get("error")],
                ["Scheduler Running", latest_debug_sample.get("scheduler_running")],
                ["Scheduler Paused", latest_debug_sample.get("scheduler_paused")],
                ["Active DAGs", latest_debug_sample.get("active_dags_count")],
                ["Fetch Counter", latest_debug_sample.get("fetch_counter")],
                ["Event Queue", latest_debug_sample.get("event_queue_size")],
                [
                    "LLM Dispatchers",
                    f"{latest_debug_sample.get('llm_dispatch_running_dispatchers')}/"
                    f"{latest_debug_sample.get('llm_dispatch_registered_dispatchers')}",
                ],
            ],
        )

    errors_body = "<p class=\"notice\">No verification errors.</p>"
    if verification_errors:
        errors_body = _html_table(
            ["Verification Error"],
            [[error] for error in verification_errors],
        )

    failures_body = "<p class=\"notice\">No recent failures.</p>"
    if recent_failures:
        failures_body = _html_table(
            [
                "Job Index",
                "Request ID",
                "Job ID",
                "Status",
                "Failure Reason",
                "Source",
            ],
            [
                [
                    job.get("job_index"),
                    job.get("request_id"),
                    job.get("job_id"),
                    job.get("terminal_status"),
                    job.get("failure_reason"),
                    job.get("source_path"),
                ]
                for job in recent_failures
            ],
        )

    body = "".join(
        [
            _html_section("Run Health", cards),
            _html_section("Throughput and Flow", flow_metrics),
            _html_section("Live SLA Status", sla_body),
            _html_section("Observed Latency", latency_body),
            _html_section("Queue and Dispatcher Signals", debug_body),
            _html_section("Recent Failures", failures_body),
            _html_section("Verification Errors", errors_body),
        ]
    )
    return _render_html_document(
        title="Gateway E2E Live Report",
        subtitle="Auto-refreshing aggregate view of throughput, backlog, and latency.",
        body=body,
        refresh_seconds=refresh_seconds,
    )


def _render_final_report_html(payload: Dict[str, Any]) -> str:
    run_identity = payload.get("run_identity", {})
    summary = payload.get("summary", {})
    latency_stats = payload.get("latency_stats_ms", {})
    failure_reasons = payload.get("failure_reasons", {})
    sla = payload.get("sla", {})
    sla_verification = payload.get("sla_verification", {})
    debug_sampling = payload.get("debug_sampling", {})
    preflight = payload.get("preflight", {})
    reliability = payload.get("reliability", {})
    event_validation = payload.get("event_validation", {})
    correctness = payload.get("correctness_verifier") or {}
    verification = payload.get("verification", {})
    job_record_stream = payload.get("job_record_stream", {})
    jobs = payload.get("jobs", [])

    cards = _html_cards(
        [
            ("Run ID", run_identity.get("run_id")),
            ("Run Mode", summary.get("run_mode")),
            ("Duration Seconds", summary.get("duration_seconds")),
            ("Configured Jobs", summary.get("configured_job_count")),
            ("Created Jobs", summary.get("total_jobs")),
            ("Submitted", summary.get("submitted_jobs")),
            ("Completed", summary.get("completed_jobs")),
            ("Failed", summary.get("failed_jobs")),
            ("Submit Failed", summary.get("submit_failed_jobs")),
            ("Timed Out", summary.get("event_timeout_jobs")),
            ("Throughput", summary.get("throughput")),
            ("Fault Profile", summary.get("fault_profile")),
            ("Mock Process Time", summary.get("mock_process_time")),
            ("Generated At", summary.get("report_generated_at")),
            ("Preflight", (preflight.get("result") or {}).get("passed")),
            ("Reliability", reliability.get("passed")),
            ("Correctness", correctness.get("passed")),
        ]
    )

    preflight_result = preflight.get("result") or {}
    preflight_final = preflight_result.get("final") or {}
    preflight_table = _html_table(
        ["Field", "Value"],
        [
            ["Enabled", preflight_result.get("enabled")],
            ["Passed", preflight_result.get("passed")],
            ["Reason", preflight_result.get("reason")],
            ["Attempts", preflight_result.get("attempts")],
            ["Target Queue", preflight_final.get("target_queue")],
            [
                "Queue Known Before Submission",
                preflight_final.get("queue_known_before_submission"),
            ],
            ["Known Queues Before Submission", preflight_final.get("known_queues")],
            ["Required Executors", preflight_final.get("required_executors")],
            [
                "Executor Readiness Required",
                preflight_final.get("executor_readiness_required"),
            ],
            ["Capacity Observed", preflight_final.get("capacity_observed")],
            ["Matched Slots", preflight_final.get("matched_slots")],
            ["Missing Executors", preflight_final.get("missing_executors")],
            [
                "Zero Capacity Executors",
                preflight_final.get("zero_capacity_executors"),
            ],
            [
                "No Free Slot Executors",
                preflight_final.get("zero_available_executors"),
            ],
            ["Capacity Error", preflight_final.get("capacity_error")],
            ["Error", preflight_result.get("error")],
        ],
    )
    preflight_attempts = preflight.get("attempts") or []
    preflight_attempts_table = _html_table(
        [
            "Attempt",
            "Captured At",
            "Debug Status",
            "Debug Snapshot",
            "Capacity Status",
            "Capacity Snapshot",
            "Capacity Error",
            "Ready",
            "Reason",
        ],
        [
            [
                attempt.get("attempt"),
                attempt.get("captured_at"),
                (attempt.get("debug") or {}).get("status_code"),
                (attempt.get("debug") or {}).get("payload"),
                (attempt.get("capacity") or {}).get("status_code"),
                (attempt.get("capacity") or {}).get("payload"),
                (attempt.get("capacity") or {}).get("error"),
                (attempt.get("interpretation") or {}).get("ready"),
                (attempt.get("interpretation") or {}).get("reason"),
            ]
            for attempt in preflight_attempts
        ],
    )

    reliability_table = _html_table(
        ["Field", "Value"],
        [
            ["Passed", reliability.get("passed")],
            *[
                [name, value]
                for name, value in (reliability.get("observed") or {}).items()
            ],
            *[
                [f"gate: {name}", value]
                for name, value in (reliability.get("gates") or {}).items()
            ],
            *[["Error", error] for error in reliability.get("errors") or []],
        ],
    )

    event_table = _html_table(
        ["Validation Counter", "Count"],
        [[name, value] for name, value in event_validation.items()],
    )

    verifier_table = _html_table(
        ["Field", "Value"],
        [
            ["Overall Verification Passed", verification.get("passed")],
            ["Correctness Enabled", correctness.get("enabled")],
            ["Correctness Passed", correctness.get("passed")],
            ["Correctness Status", correctness.get("status")],
            ["Correctness Exit Code", correctness.get("exit_code")],
            ["Correctness Error", correctness.get("error")],
            ["Trace Mode", payload.get("trace_mode")],
            ["Query Budget Deltas", payload.get("query_budget_deltas")],
            ["Post-Drain Capacity", payload.get("post_drain_capacity")],
            ["Job JSONL", job_record_stream.get("path")],
            ["Streamed Records", job_record_stream.get("records_written")],
            ["Retained Jobs", job_record_stream.get("retained_jobs")],
            ["Retention Truncated", job_record_stream.get("truncated")],
            *[["Error", error] for error in verification.get("errors") or []],
        ],
    )

    latency_rows: List[List[Any]] = []
    for name, stats in latency_stats.items():
        if not isinstance(stats, dict):
            continue
        latency_rows.append(
            [
                name,
                int(stats.get("count", 0)),
                stats.get("min"),
                stats.get("max"),
                stats.get("avg"),
                stats.get("p50"),
                stats.get("p95"),
                stats.get("p99"),
            ]
        )
    latency_table = _html_table(
        ["Latency", "Count", "Min", "Max", "Avg", "P50", "P95", "P99"],
        latency_rows,
    )

    failure_table = _html_table(
        ["Reason", "Count"],
        [[reason, count] for reason, count in sorted(failure_reasons.items())],
    )

    sla_table = _html_table(
        ["Field", "Value"],
        [
            ["Soft SLA Seconds", sla_verification.get("soft_sla_seconds")],
            ["Hard SLA Seconds", sla_verification.get("hard_sla_seconds")],
            ["Soft SLA Step Seconds", sla_verification.get("soft_sla_step_seconds")],
            ["Hard SLA Step Seconds", sla_verification.get("hard_sla_step_seconds")],
            ["SLA Step Every Jobs", sla_verification.get("sla_step_every_jobs")],
            ["SLA Step Cycle", sla_verification.get("sla_step_cycle")],
            [
                "Min Soft SLA Compliance %",
                sla_verification.get("min_soft_sla_compliance_pct"),
            ],
            [
                "Min Hard SLA Compliance %",
                sla_verification.get("min_hard_sla_compliance_pct"),
            ],
            ["Verification Passed", sla_verification.get("passed")],
        ],
    )

    sla_summary_rows: List[List[Any]] = []
    for label in ("soft", "hard"):
        summary_block = sla.get(label)
        if not isinstance(summary_block, dict):
            continue
        lateness_stats = summary_block.get("lateness_stats_ms") or {}
        sla_summary_rows.append(
            [
                label,
                summary_block.get("configured_seconds"),
                summary_block.get("step_seconds"),
                summary_block.get("min_compliance_pct"),
                summary_block.get("configured_jobs"),
                summary_block.get("terminal_evaluated_jobs"),
                summary_block.get("met_jobs"),
                summary_block.get("missed_jobs"),
                summary_block.get("failed_jobs"),
                summary_block.get("pending_jobs"),
                summary_block.get("overdue_open_jobs"),
                summary_block.get("terminal_compliance_pct"),
                lateness_stats.get("p95") if isinstance(lateness_stats, dict) else None,
                lateness_stats.get("max") if isinstance(lateness_stats, dict) else None,
            ]
        )
    sla_summary_body = "<p class=\"notice\">No SLA configured for this run.</p>"
    if sla_summary_rows:
        sla_summary_body = _html_table(
            [
                "SLA",
                "Base Seconds",
                "Step Seconds",
                "Min Compliance %",
                "Configured Jobs",
                "Terminal Jobs",
                "Met",
                "Missed",
                "Terminal Failed",
                "Pending",
                "Overdue Open",
                "Terminal Compliance %",
                "P95 Lateness ms",
                "Max Lateness ms",
            ],
            sla_summary_rows,
        )

    sla_miss_jobs = [
        job
        for job in jobs
        if job.get("soft_sla_status") == "deadline_missed"
        or job.get("hard_sla_status") == "deadline_missed"
        or job.get("soft_sla_status") == "terminal_failed"
        or job.get("hard_sla_status") == "terminal_failed"
    ]
    sla_miss_jobs.sort(
        key=lambda job: max(
            float(job.get("hard_sla_lateness_ms") or 0.0),
            float(job.get("soft_sla_lateness_ms") or 0.0),
        ),
        reverse=True,
    )
    top_sla_misses = sla_miss_jobs[:25]
    sla_miss_body = "<p class=\"notice\">No SLA misses recorded.</p>"
    if top_sla_misses:
        sla_miss_body = _html_table(
            [
                "Job Index",
                "Request ID",
                "Job ID",
                "Status",
                "Soft SLA",
                "Soft Lateness ms",
                "Hard SLA",
                "Hard Lateness ms",
                "End-to-End ms",
                "Failure",
            ],
            [
                [
                    job.get("job_index"),
                    job.get("request_id"),
                    job.get("job_id"),
                    job.get("terminal_status"),
                    job.get("soft_sla_status"),
                    job.get("soft_sla_lateness_ms"),
                    job.get("hard_sla_status"),
                    job.get("hard_sla_lateness_ms"),
                    job.get("end_to_end_ms"),
                    job.get("failure_reason"),
                ]
                for job in top_sla_misses
            ],
        )

    errors = sla_verification.get("errors") or []
    errors_body = "<p class=\"notice\">No SLA verification errors.</p>"
    if errors:
        errors_body = _html_table(["Verification Error"], [[error] for error in errors])

    debug_samples = debug_sampling.get("samples") or []
    debug_body = _html_table(
        [
            "Stage",
            "Captured At",
            "OK",
            "Status Code",
            "Active DAGs",
            "Event Queue",
            "Dispatchers",
            "Error",
        ],
        [
            [
                sample.get("stage"),
                sample.get("captured_at"),
                sample.get("ok"),
                sample.get("status_code"),
                sample.get("active_dags_count"),
                sample.get("event_queue_size"),
                (
                    f"{sample.get('llm_dispatch_running_dispatchers')}/"
                    f"{sample.get('llm_dispatch_registered_dispatchers')}"
                ),
                sample.get("error"),
            ]
            for sample in debug_samples[-25:]
        ],
    )

    visible_jobs = jobs[-200:]
    jobs_body = (
        '<p class="notice">'
        f"Showing the most recent {len(visible_jobs)} of {len(jobs)} jobs."
        "</p>"
        + _html_table(
            [
                "Job Index",
                "Request ID",
                "Job ID",
                "Status",
                "Source",
                "S3 URI",
                "Scheduling ms",
                "Queue Wait ms",
                "Execution ms",
                "End-to-End ms",
                "Soft SLA",
                "Hard SLA",
                "Failure",
            ],
            [
                [
                    job.get("job_index"),
                    job.get("request_id"),
                    job.get("job_id"),
                    job.get("terminal_status"),
                    job.get("source_path"),
                    job.get("s3_uri"),
                    job.get("scheduling_ms"),
                    job.get("queue_wait_ms"),
                    job.get("execution_ms"),
                    job.get("end_to_end_ms"),
                    job.get("soft_sla_status"),
                    job.get("hard_sla_status"),
                    job.get("failure_reason"),
                ]
                for job in visible_jobs
            ],
        )
    )

    body = "".join(
        [
            _html_section("Run Summary", cards),
            _html_section(
                "Dispatch Preflight", preflight_table + preflight_attempts_table
            ),
            _html_section("Reliability Gates", reliability_table),
            _html_section("Event Validation", event_table),
            _html_section("Correctness and Instrumentation", verifier_table),
            _html_section("Latency Breakdown", latency_table),
            _html_section("Failure Reasons", failure_table),
            _html_section("SLA Outcome", sla_summary_body),
            _html_section("SLA Verification", sla_table + errors_body),
            _html_section("Worst SLA Misses", sla_miss_body),
            _html_section("Debug Sampling", debug_body),
            _html_section("Recent Jobs", jobs_body),
        ]
    )
    return _render_html_document(
        title="Gateway E2E Stress Report",
        subtitle="Human-readable summary of scheduler, queue, and LLM execution behavior.",
        body=body,
    )


def _render_dry_run_report_html(payload: Dict[str, Any]) -> str:
    submissions = payload.get("submissions", [])
    cards = _html_cards(
        [
            ("Run Mode", payload.get("run_mode")),
            ("Job Count", payload.get("job_count")),
            ("Run Time Seconds", payload.get("run_time_seconds")),
            ("Estimated Jobs", payload.get("estimated_job_count")),
            ("Preview Jobs", payload.get("preview_job_count")),
            ("Resolved Inputs", payload.get("input_assets_resolved")),
            ("Planner", payload.get("planner")),
            ("Job Name", payload.get("job_name")),
            ("Fault Profile", payload.get("fault_profile")),
            ("Generated At", payload.get("generated_at")),
        ]
    )
    submission_table = _html_table(
        [
            "Job Index",
            "Request ID",
            "Source",
            "Input Mode",
            "S3 URI",
            "Planner",
            "Upload Planned",
            "Soft SLA Offset",
            "Hard SLA Offset",
        ],
        [
            [
                item.get("job_index"),
                item.get("request_id"),
                item.get("source_path"),
                item.get("input_mode"),
                item.get("s3_uri"),
                item.get("planner"),
                item.get("upload_planned"),
                item.get("soft_sla_offset_seconds"),
                item.get("hard_sla_offset_seconds"),
            ]
            for item in submissions
        ],
    )
    body = "".join(
        [
            _html_section("Dry-Run Summary", cards),
            _html_section("Preview Submissions", submission_table),
            _html_section(
                "Raw Payload",
                "<pre>" + html.escape(json.dumps(payload, indent=2)) + "</pre>",
            ),
        ]
    )
    return _render_html_document(
        title="Gateway E2E Dry-Run Report",
        subtitle="Preview of the resolved submissions without upload or gateway traffic.",
        body=body,
    )
