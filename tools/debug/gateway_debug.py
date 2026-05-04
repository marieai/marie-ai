#!/usr/bin/env python3
"""
Gateway scheduler debug utility.

Collects gateway debug/status snapshots, scheduler database diagnostics, and
optional gateway logs into a single JSON report.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from collections import deque
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable
from urllib import error as urllib_error
from urllib import request as urllib_request
from uuid import UUID

try:
    import requests as requests_lib
except ImportError:  # pragma: no cover - requests is optional
    requests_lib = None

try:  # pragma: no cover - environment-dependent import
    import psycopg  # type: ignore[import-not-found]

    PSYCOPG_DRIVER = "psycopg"
except ImportError:  # pragma: no cover - environment-dependent import
    psycopg = None
    PSYCOPG_DRIVER = None

if psycopg is None:  # pragma: no cover - environment-dependent import
    try:
        import psycopg2  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover - environment-dependent import
        psycopg2 = None


TOOL_VERSION = "0.1.0"
SCHEMA_PATTERN = re.compile(r"^[a-z_][a-z0-9_]*$")
ANSI_ESCAPE_PATTERN = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
ACTIVE_JOB_STATES = {"active", "retry"}
TERMINAL_JOB_STATES = {"completed", "skipped", "expired", "cancelled", "failed"}
TERMINAL_DAG_STATES = TERMINAL_JOB_STATES
STATUS_ENDPOINT_CANDIDATES = ("/status", "/health/status")
FINDING_HINTS: dict[str, list[str]] = {
    "admission_starvation": [
        "Check whether the active DAGs are making progress or are stuck in terminal-zombie or stale-active states.",
        "Inspect the oldest ready jobs and confirm the gateway still has free workers or executors available to admit them.",
        "If slots are occupied by terminal or wedged DAGs, capture a report first and then clear or restart the affected scheduler process.",
    ],
    "terminal_zombie_dags": [
        "Compare gateway in-memory active DAGs with database DAG state to confirm the admission slots are stale.",
        "Check gateway logs for missing DAG cleanup or exceptions after job completion.",
        "If the scheduler is wedged on stale DAG state, capture evidence and restart the gateway or clear the bad DAGs through the normal operator path.",
    ],
    "stuck_active_jobs": [
        "Inspect the affected job IDs in executor and gateway logs to confirm whether the worker crashed or stopped heartbeating.",
        "Check lease_owner and lease_expires_at fields to see whether another worker still owns the job.",
        "If the executor is gone and the lease will not recover, use the normal operator workflow to reset or fail the job so the DAG can move again.",
    ],
    "ready_backlog_aging": [
        "Check whether the frontier has ready work but no DAG slots or workers available to pick it up.",
        "Review queue balance and worker capacity for the affected planner or queue before increasing thresholds.",
        "If the backlog is expected for this workload, adjust the long-running or ready-age thresholds rather than treating it as an outage.",
    ],
    "scheduler_not_polling": [
        "Check gateway logs for fetch-loop or scheduler exceptions around the current time window.",
        "Confirm the scheduler thread/process is running and can still reach PostgreSQL and any required broker dependencies.",
        "If the process is up but the fetch counter stays at zero, capture the report and restart the gateway scheduler component.",
    ],
    "unresolved_terminal_dags": [
        "Inspect DAG finalization code paths to see why the DAG state was not updated after all jobs finished.",
        "Check for database write failures or transaction rollbacks near the end of the DAG lifecycle.",
        "Use the normal operator cleanup path to reconcile the DAG state once you confirm the jobs are truly terminal.",
    ],
    "hydrated_created_dags": [
        "Confirm the DAG was admitted into gateway memory but never advanced into active execution.",
        "Check gateway scheduling decisions and worker availability for the planner that owns the DAG.",
        "Look for dependency or frontier issues that keep the DAG loaded but unable to start any jobs.",
    ],
    "gateway_db_divergence": [
        "Compare gateway /api/debug state with direct database counts to confirm whether the divergence is persistent.",
        "Check for recent gateway restarts, failed writes, or cleanup exceptions that could desynchronize memory from PostgreSQL.",
        "If divergence persists, restart the gateway after capturing the report so in-memory state is rebuilt from the database.",
    ],
    "submission_workers_idle": [
        "Check submission worker logs for startup failures or blocked queue consumption.",
        "Confirm the pending request queue is draining and that workers still have connectivity to the scheduler database.",
        "If workers are not recovering on their own, restart the submission worker process after collecting evidence.",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect Marie gateway scheduler diagnostics."
    )
    parser.add_argument(
        "--gateway-url",
        default=os.getenv("MARIE_GATEWAY_URL", "http://localhost:51000"),
        help="Gateway base URL",
    )
    parser.add_argument(
        "--no-gateway", action="store_true", help="Skip gateway HTTP collection"
    )
    parser.add_argument(
        "--gateway-timeout",
        type=float,
        default=10.0,
        help="Gateway HTTP timeout in seconds",
    )
    parser.add_argument("--db-host", default=os.getenv("MARIE_DB_HOST", "localhost"))
    parser.add_argument(
        "--db-port", type=int, default=int(os.getenv("MARIE_DB_PORT", "5432"))
    )
    parser.add_argument("--db-user", default=os.getenv("MARIE_DB_USER", "marie"))
    parser.add_argument("--db-password", default=os.getenv("MARIE_DB_PASSWORD"))
    parser.add_argument("--db-name", default=os.getenv("MARIE_DB_NAME", "marie"))
    parser.add_argument("--db-schema", default="marie_scheduler")
    parser.add_argument(
        "--no-db", action="store_true", help="Skip scheduler database diagnostics"
    )
    parser.add_argument("--job-id", help="Inspect a specific job UUID")
    parser.add_argument("--dag-id", help="Inspect a specific DAG UUID")
    parser.add_argument(
        "--long-running-threshold",
        type=int,
        default=15,
        help="Long-running threshold in minutes",
    )
    parser.add_argument(
        "--container-name", default="marieai-gateway", help="Gateway container name"
    )
    parser.add_argument(
        "--log-file",
        default=os.getenv("MARIE_GATEWAY_LOG_FILE"),
        help="Read gateway logs from a file instead of docker",
    )
    parser.add_argument(
        "--log-tail", type=int, default=200, help="Number of log lines to tail"
    )
    parser.add_argument(
        "--include-log-lines",
        action="store_true",
        help="Include raw tailed log lines in the JSON output",
    )
    parser.add_argument("--no-logs", action="store_true", help="Skip log collection")
    parser.add_argument(
        "--pretty", action="store_true", help="Pretty-print JSON output"
    )
    args = parser.parse_args()

    if not SCHEMA_PATTERN.match(args.db_schema):
        raise ValueError(f"Invalid db schema: {args.db_schema!r}")

    if args.log_tail <= 0:
        raise ValueError("--log-tail must be greater than zero")

    if args.long_running_threshold <= 0:
        raise ValueError("--long-running-threshold must be greater than zero")

    if args.job_id:
        UUID(args.job_id)
    if args.dag_id:
        UUID(args.dag_id)

    if args.no_gateway and args.no_db and args.no_logs:
        raise ValueError("At least one source must be enabled (gateway, db, or logs)")

    if args.log_file:
        path = Path(args.log_file).expanduser()
        args.log_file = str(path)

    return args


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def normalize_state(value: Any) -> str:
    return str(value or "").strip().lower()


def timedelta_seconds(value: timedelta | None) -> float | None:
    if value is None:
        return None
    return max(0.0, value.total_seconds())


def serialize_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.astimezone(UTC).isoformat()
    if isinstance(value, timedelta):
        return max(0.0, value.total_seconds())
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, Decimal):
        return int(value) if value == int(value) else float(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, dict):
        return {str(key): serialize_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize_value(item) for item in value]
    return value


def humanize_seconds(seconds: float | None) -> str | None:
    if seconds is None:
        return None
    return str(timedelta(seconds=int(seconds)))


def strip_ansi(text: str) -> str:
    return ANSI_ESCAPE_PATTERN.sub("", text)


def coerce_json_response(payload: Any) -> Any:
    if (
        isinstance(payload, dict)
        and "result" in payload
        and payload.get("status") in {"OK", "error"}
    ):
        return payload.get("result")
    return payload


def http_get_json(url: str, timeout: float) -> dict[str, Any]:
    if requests_lib is not None:
        response = requests_lib.get(url, timeout=timeout)
        response.raise_for_status()
        return {
            "status_code": response.status_code,
            "body": response.json(),
        }

    req = urllib_request.Request(url, headers={"Accept": "application/json"})
    with urllib_request.urlopen(req, timeout=timeout) as resp:  # nosec B310
        body = resp.read().decode("utf-8")
        return {
            "status_code": getattr(resp, "status", 200),
            "body": json.loads(body),
        }


def collect_gateway(url: str, timeout: float) -> dict[str, Any]:
    base_url = url.rstrip("/")
    gateway: dict[str, Any] = {
        "reachable": False,
        "status": None,
        "status_endpoint": None,
        "debug": None,
        "errors": {},
    }

    for status_path in STATUS_ENDPOINT_CANDIDATES:
        status_url = f"{base_url}{status_path}"
        try:
            result = http_get_json(status_url, timeout)
            gateway["reachable"] = True
            gateway["status_endpoint"] = status_path
            gateway["status"] = coerce_json_response(result["body"])
            break
        except Exception as exc:  # noqa: BLE001
            gateway["errors"][status_path] = format_source_error(exc)

    debug_url = f"{base_url}/api/debug"
    try:
        result = http_get_json(debug_url, timeout)
        gateway["reachable"] = True
        gateway["debug"] = coerce_json_response(result["body"])
    except json.JSONDecodeError as exc:
        gateway["errors"]["/api/debug"] = {
            "kind": "invalid_json",
            "message": str(exc),
        }
    except Exception as exc:  # noqa: BLE001
        gateway["errors"]["/api/debug"] = format_source_error(exc)

    return gateway


def format_source_error(exc: Exception) -> dict[str, Any]:
    if requests_lib is not None and isinstance(
        exc, requests_lib.exceptions.RequestException
    ):
        return {"kind": exc.__class__.__name__, "message": str(exc)}
    if isinstance(exc, urllib_error.HTTPError):
        return {"kind": "http_error", "message": str(exc), "status_code": exc.code}
    if isinstance(exc, urllib_error.URLError):
        return {"kind": "url_error", "message": str(exc.reason)}
    return {"kind": exc.__class__.__name__, "message": str(exc)}


def connect_db(args: argparse.Namespace):
    kwargs = {
        "host": args.db_host,
        "port": args.db_port,
        "user": args.db_user,
        "dbname": args.db_name,
        "connect_timeout": 10,
    }
    if args.db_password:
        kwargs["password"] = args.db_password

    if PSYCOPG_DRIVER == "psycopg" and psycopg is not None:  # pragma: no branch
        conn = psycopg.connect(**kwargs)
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("SET default_transaction_read_only = on")
        return conn

    if psycopg2 is not None:  # pragma: no branch
        conn = psycopg2.connect(**kwargs)
        conn.set_session(readonly=True, autocommit=True)
        return conn

    raise RuntimeError("No PostgreSQL driver available (psycopg or psycopg2 required)")


def run_query(
    conn: Any, sql: str, params: Iterable[Any] | None = None
) -> list[dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(sql, tuple(params or ()))
        if cur.description is None:
            return []
        columns = [getattr(desc, "name", desc[0]) for desc in cur.description]
        rows = cur.fetchall()
        return [dict(zip(columns, row, strict=False)) for row in rows]


def with_human_duration(row: dict[str, Any], field: str) -> dict[str, Any]:
    value = row.get(field)
    seconds = None
    if isinstance(value, timedelta):
        seconds = timedelta_seconds(value)
    elif isinstance(value, (int, float)):
        seconds = float(value)
    normalized = dict(row)
    normalized[f"{field}_seconds"] = seconds
    normalized[f"{field}_human"] = humanize_seconds(seconds)
    if field in normalized:
        normalized.pop(field, None)
    return normalized


def normalize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [serialize_value(row) for row in rows]


def get_active_dag_ids(gateway_debug: Any) -> list[str]:
    if not isinstance(gateway_debug, dict):
        return []

    active = gateway_debug.get("active_dags")
    raw_ids: list[str] = []
    if isinstance(active, dict):
        raw_ids = [str(key) for key in active.keys()]
    elif isinstance(active, list):
        for item in active:
            if isinstance(item, str):
                raw_ids.append(item)
            elif isinstance(item, dict):
                dag_id = item.get("dag_id") or item.get("id")
                if dag_id:
                    raw_ids.append(str(dag_id))

    valid_ids: list[str] = []
    for dag_id in raw_ids:
        try:
            valid_ids.append(str(UUID(dag_id)))
        except (ValueError, TypeError):
            continue
    return valid_ids


def query_dag_classification_rows(
    conn: Any,
    schema: str,
    dag_ids: list[str] | None,
) -> list[dict[str, Any]]:
    select_sql = f"""
        SELECT
          d.id, d.name, d.state, d.planner, d.started_on, d.completed_on,
          COUNT(*) FILTER (WHERE j.state = 'created')   AS created_jobs,
          COUNT(*) FILTER (WHERE j.state = 'retry')     AS retry_jobs,
          COUNT(*) FILTER (WHERE j.state = 'active')    AS active_jobs,
          COUNT(*) FILTER (WHERE j.state = 'completed') AS completed_jobs,
          COUNT(*) FILTER (WHERE j.state = 'failed')    AS failed_jobs,
          COUNT(*) FILTER (WHERE j.state = 'skipped')   AS skipped_jobs,
          COUNT(*) FILTER (WHERE j.state = 'expired')   AS expired_jobs,
          COUNT(*) FILTER (WHERE j.state = 'cancelled') AS cancelled_jobs
        FROM {schema}.dag d
        LEFT JOIN {schema}.job j ON j.dag_id = d.id
    """

    params: list[Any] = []
    if dag_ids:
        placeholders = ", ".join(["%s"] * len(dag_ids))
        where_sql = f"WHERE d.id IN ({placeholders})"
        params.extend(dag_ids)
    else:
        where_sql = "WHERE d.state IN ('active', 'created')"

    sql = f"""
        {select_sql}
        {where_sql}
        GROUP BY d.id, d.name, d.state, d.planner, d.started_on, d.completed_on
        ORDER BY d.state, d.completed_on NULLS FIRST
    """
    return run_query(conn, sql, params)


def classify_dag_row(
    row: dict[str, Any],
    in_memory_ids: set[str],
    stuck_dag_ids: set[str],
) -> dict[str, Any]:
    terminal_jobs = (
        int(row.get("completed_jobs") or 0)
        + int(row.get("failed_jobs") or 0)
        + int(row.get("skipped_jobs") or 0)
        + int(row.get("expired_jobs") or 0)
        + int(row.get("cancelled_jobs") or 0)
    )
    created_jobs = int(row.get("created_jobs") or 0)
    retry_jobs = int(row.get("retry_jobs") or 0)
    active_jobs = int(row.get("active_jobs") or 0)
    total_jobs = terminal_jobs + created_jobs + retry_jobs + active_jobs
    dag_state = normalize_state(row.get("state"))
    dag_id = str(row.get("id"))

    classification = "truly_running"
    if dag_id in in_memory_ids and dag_state in TERMINAL_DAG_STATES:
        classification = "terminal_zombie"
    elif (
        total_jobs > 0
        and terminal_jobs == total_jobs
        and dag_state not in TERMINAL_DAG_STATES
    ):
        classification = "all_jobs_terminal"
    elif dag_id in stuck_dag_ids:
        classification = "stuck_active"
    elif dag_state == "created" and active_jobs == 0:
        classification = "hydrated_not_running"
    elif active_jobs > 0:
        classification = "truly_running"

    normalized = dict(row)
    normalized["classification"] = classification
    return normalized


def collect_database(
    args: argparse.Namespace, gateway_data: dict[str, Any]
) -> dict[str, Any]:
    database: dict[str, Any] = {
        "reachable": False,
        "dag_state_distribution": [],
        "job_state_distribution": [],
        "dag_classification": [],
        "stuck_active_jobs": [],
        "stale_active_by_queue": [],
        "long_completed_jobs": [],
        "job_detail": None,
        "dag_jobs": None,
        "error": None,
    }

    try:
        conn = connect_db(args)
    except Exception as exc:  # noqa: BLE001
        database["error"] = {"stage": "connect", **format_source_error(exc)}
        return database

    schema = args.db_schema
    threshold_minutes = args.long_running_threshold

    try:
        database["reachable"] = True
        database["dag_state_distribution"] = normalize_rows(
            run_query(
                conn,
                f"SELECT state, COUNT(1) AS total_count FROM {schema}.dag GROUP BY state ORDER BY total_count DESC",
            )
        )
        database["job_state_distribution"] = normalize_rows(
            run_query(
                conn,
                f"SELECT state, COUNT(1) AS total_count FROM {schema}.job GROUP BY state ORDER BY total_count DESC",
            )
        )

        stuck_rows = run_query(
            conn,
            f"""
            SELECT dag_id::text, id::text, name, state, started_on,
                   now() - started_on AS run_time, job_level, priority,
                   run_owner, run_lease_expires_at, lease_owner, lease_expires_at
            FROM {schema}.job
            WHERE state = 'active' AND started_on IS NOT NULL
              AND now() - started_on > interval %s
            ORDER BY (now() - started_on) DESC
            LIMIT 50
            """,
            [f"{threshold_minutes} minutes"],
        )
        stuck_rows = [with_human_duration(row, "run_time") for row in stuck_rows]
        database["stuck_active_jobs"] = normalize_rows(stuck_rows)
        stuck_dag_ids = {str(row["dag_id"]) for row in stuck_rows if row.get("dag_id")}

        stale_rows = run_query(
            conn,
            f"""
            SELECT name AS queue, COUNT(*) AS active_count, MIN(started_on) AS oldest_started
            FROM {schema}.job
            WHERE state = 'active'
            GROUP BY name
            ORDER BY active_count DESC
            """,
        )
        database["stale_active_by_queue"] = normalize_rows(stale_rows)

        long_completed_rows = run_query(
            conn,
            f"""
            SELECT dag_id::text, id::text, state, created_on, started_on, completed_on,
                   completed_on - started_on AS run_time, job_level, priority
            FROM {schema}.job
            WHERE started_on IS NOT NULL AND completed_on IS NOT NULL
              AND completed_on - started_on > interval %s
            ORDER BY (completed_on - started_on) DESC
            LIMIT 50
            """,
            [f"{threshold_minutes} minutes"],
        )
        long_completed_rows = [
            with_human_duration(row, "run_time") for row in long_completed_rows
        ]
        database["long_completed_jobs"] = normalize_rows(long_completed_rows)

        if args.job_id:
            details = run_query(
                conn,
                f"SELECT * FROM {schema}.job WHERE id = %s ORDER BY created_on DESC LIMIT 1",
                [args.job_id],
            )
            database["job_detail"] = serialize_value(details[0]) if details else None

        if args.dag_id:
            dag_jobs = run_query(
                conn,
                f"SELECT * FROM {schema}.job WHERE dag_id = %s ORDER BY job_level, created_on",
                [args.dag_id],
            )
            database["dag_jobs"] = normalize_rows(dag_jobs)

        gateway_ids = get_active_dag_ids(gateway_data.get("debug"))
        if args.dag_id:
            target_dag_ids = [args.dag_id]
        elif gateway_ids:
            target_dag_ids = gateway_ids
        else:
            target_dag_ids = None

        dag_rows = query_dag_classification_rows(conn, schema, target_dag_ids)
        in_memory_ids = set(gateway_ids)
        classified_rows = [
            classify_dag_row(
                row, in_memory_ids=in_memory_ids, stuck_dag_ids=stuck_dag_ids
            )
            for row in dag_rows
        ]
        database["dag_classification"] = normalize_rows(classified_rows)
    except Exception as exc:  # noqa: BLE001
        database["error"] = {"stage": "query", **format_source_error(exc)}
    finally:
        conn.close()

    return database


def format_log_payload(
    *,
    available: bool,
    source: str | None,
    exit_code: int | None,
    error: dict[str, Any] | None,
    lines: list[str],
    include_lines: bool,
    skipped: bool = False,
) -> dict[str, Any]:
    sanitized_lines = [strip_ansi(line) for line in lines]
    payload = {
        "available": available,
        "source": source,
        "exit_code": exit_code,
        "error": error,
        "skipped": skipped,
        "line_count": len(sanitized_lines),
        "lines_included": include_lines,
        "lines": sanitized_lines if include_lines else [],
    }
    return payload


def collect_container_logs(name: str, tail: int, include_lines: bool) -> dict[str, Any]:
    try:
        result = subprocess.run(
            ["docker", "logs", "--tail", str(tail), name],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        return format_log_payload(
            available=False,
            source="docker",
            exit_code=None,
            error={"kind": "docker_unavailable", "message": str(exc)},
            lines=[],
            include_lines=include_lines,
        )
    except PermissionError as exc:
        return format_log_payload(
            available=False,
            source="docker",
            exit_code=None,
            error={"kind": "permission_denied", "message": str(exc)},
            lines=[],
            include_lines=include_lines,
        )

    output = "\n".join(
        part for part in [result.stdout, result.stderr] if part
    ).splitlines()
    if result.returncode != 0:
        kind = "container_missing"
        stderr = (result.stderr or result.stdout or "").lower()
        if "permission denied" in stderr:
            kind = "permission_denied"
        return format_log_payload(
            available=False,
            source="docker",
            exit_code=result.returncode,
            error={
                "kind": kind,
                "message": (result.stderr or result.stdout).strip(),
            },
            lines=output[-tail:],
            include_lines=include_lines,
        )

    return format_log_payload(
        available=True,
        source="docker",
        exit_code=result.returncode,
        error=None,
        lines=output[-tail:],
        include_lines=include_lines,
    )


def collect_file_logs(path: str, tail: int, include_lines: bool) -> dict[str, Any]:
    log_path = Path(path)
    if not log_path.exists():
        return format_log_payload(
            available=False,
            source="file",
            exit_code=None,
            error={"kind": "file_missing", "message": f"Log file not found: {path}"},
            lines=[],
            include_lines=include_lines,
        )
    if not log_path.is_file():
        return format_log_payload(
            available=False,
            source="file",
            exit_code=None,
            error={
                "kind": "invalid_log_source",
                "message": f"Not a regular file: {path}",
            },
            lines=[],
            include_lines=include_lines,
        )

    try:
        with log_path.open("r", encoding="utf-8", errors="replace") as handle:
            lines = list(deque((line.rstrip("\n") for line in handle), maxlen=tail))
    except OSError as exc:
        return format_log_payload(
            available=False,
            source="file",
            exit_code=None,
            error={"kind": "file_unreadable", "message": str(exc)},
            lines=[],
            include_lines=include_lines,
        )

    return format_log_payload(
        available=True,
        source="file",
        exit_code=0,
        error=None,
        lines=lines,
        include_lines=include_lines,
    )


def collect_logs(args: argparse.Namespace) -> dict[str, Any]:
    if args.no_logs:
        return format_log_payload(
            available=False,
            source=None,
            exit_code=None,
            error=None,
            lines=[],
            include_lines=args.include_log_lines,
            skipped=True,
        )
    if args.log_file:
        return collect_file_logs(args.log_file, args.log_tail, args.include_log_lines)
    return collect_container_logs(
        args.container_name, args.log_tail, args.include_log_lines
    )


def nested_get(payload: Any, path: Iterable[str], default: Any = None) -> Any:
    current = payload
    for key in path:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default
    return current


def distribution_to_map(rows: list[dict[str, Any]]) -> dict[str, int]:
    output: dict[str, int] = {}
    for row in rows:
        state = normalize_state(row.get("state"))
        count = int(row.get("total_count") or 0)
        output[state] = count
    return output


def build_finding(
    issue: str, severity: str, detail: str, **extra: Any
) -> dict[str, Any]:
    finding = {
        "issue": issue,
        "severity": severity,
        "detail": detail,
        "hints": FINDING_HINTS.get(issue, []),
    }
    finding.update(extra)
    return finding


def analyze(
    gateway_data: dict[str, Any], db_data: dict[str, Any], threshold_minutes: int
) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    gateway_debug = gateway_data.get("debug") if gateway_data.get("reachable") else None
    gateway_metrics_available = isinstance(gateway_debug, dict)
    scheduler_info = nested_get(gateway_debug, ["scheduler_info"], {}) or {}
    counters = nested_get(gateway_debug, ["counters"], {}) or {}
    queue_status = nested_get(gateway_debug, ["queue_status"], {}) or {}
    frontier_summary = (
        nested_get(gateway_debug, ["frontier_summary"], {})
        or nested_get(gateway_debug, ["frontier", "summary"], {})
        or {}
    )

    active_dags_count = (
        int(scheduler_info.get("active_dags_count") or 0)
        if gateway_metrics_available
        else None
    )
    max_concurrent_dags = (
        int(scheduler_info.get("max_concurrent_dags") or 0)
        if gateway_metrics_available
        else None
    )
    ready_count = (
        int(nested_get(frontier_summary, ["totals", "ready"], 0) or 0)
        if gateway_metrics_available
        else None
    )
    ready_age_p90 = (
        float(nested_get(frontier_summary, ["ready_age_seconds", "p90"], 0.0) or 0.0)
        if gateway_metrics_available
        else None
    )
    fetch_counter = (
        int(counters.get("fetch_counter") or 0) if gateway_metrics_available else None
    )
    pending_requests = (
        int(counters.get("pending_requests") or 0)
        if gateway_metrics_available
        else None
    )
    active_workers = (
        int(nested_get(queue_status, ["workers", "active"], 0) or 0)
        if gateway_metrics_available
        else None
    )

    dag_classification = db_data.get("dag_classification") or []
    stuck_active_jobs = db_data.get("stuck_active_jobs") or []
    in_memory_dag_ids = get_active_dag_ids(gateway_debug)
    state_counts = distribution_to_map(db_data.get("dag_state_distribution") or [])

    if (
        max_concurrent_dags is not None
        and active_dags_count is not None
        and ready_count is not None
        and max_concurrent_dags > 0
        and active_dags_count >= max_concurrent_dags
        and ready_count > 0
    ):
        findings.append(
            build_finding(
                "admission_starvation",
                "critical",
                f"All {max_concurrent_dags} DAG slots full, {ready_count} ready jobs waiting",
            )
        )

    zombies = [
        row
        for row in dag_classification
        if row.get("classification") == "terminal_zombie"
    ]
    if zombies:
        findings.append(
            build_finding(
                "terminal_zombie_dags",
                "critical",
                "DAGs with terminal DB state still consuming admission slots",
                count=len(zombies),
                dag_ids=[row["id"] for row in zombies[:25]],
            )
        )

    if stuck_active_jobs:
        findings.append(
            build_finding(
                "stuck_active_jobs",
                "critical",
                "Jobs in ACTIVE state past threshold, likely executor crash or wedge",
                count=len(stuck_active_jobs),
                oldest_run_time_seconds=stuck_active_jobs[0].get("run_time_seconds"),
                oldest_run_time_human=stuck_active_jobs[0].get("run_time_human"),
            )
        )

    if ready_age_p90 is not None and ready_age_p90 > (threshold_minutes * 60):
        findings.append(
            build_finding(
                "ready_backlog_aging",
                "warning",
                "Ready work has been waiting longer than the configured threshold",
                p90_age_seconds=ready_age_p90,
            )
        )

    if fetch_counter is not None and fetch_counter == 0:
        findings.append(
            build_finding(
                "scheduler_not_polling",
                "critical",
                "Scheduler fetch counter is zero",
            )
        )

    unresolved = [
        row
        for row in dag_classification
        if row.get("classification") == "all_jobs_terminal"
    ]
    if unresolved:
        findings.append(
            build_finding(
                "unresolved_terminal_dags",
                "warning",
                "All jobs are terminal but DAG state is not resolved",
                count=len(unresolved),
                dag_ids=[row["id"] for row in unresolved[:25]],
            )
        )

    hydrated = [
        row
        for row in dag_classification
        if row.get("classification") == "hydrated_not_running"
    ]
    if hydrated:
        findings.append(
            build_finding(
                "hydrated_created_dags",
                "warning",
                "Created DAGs are loaded in gateway memory but not actively executing",
                count=len(hydrated),
                dag_ids=[row["id"] for row in hydrated[:25]],
            )
        )

    if gateway_data.get("reachable") and db_data.get("reachable"):
        if in_memory_dag_ids and not zombies and state_counts.get("active", 0) == 0:
            findings.append(
                build_finding(
                    "gateway_db_divergence",
                    "warning",
                    "Gateway reports in-memory DAGs while DB shows no active DAGs",
                )
            )

    if (
        active_workers is not None
        and pending_requests is not None
        and active_workers == 0
        and pending_requests > 0
    ):
        findings.append(
            build_finding(
                "submission_workers_idle",
                "warning",
                f"No active submission workers but {pending_requests} pending requests are queued",
            )
        )

    severity_rank = {"critical": 0, "warning": 1, "info": 2}
    findings.sort(
        key=lambda item: (
            severity_rank.get(item.get("severity", "info"), 9),
            item.get("issue", ""),
        )
    )
    return {"findings": findings}


def determine_exit_code(report: dict[str, Any]) -> int:
    gateway_ok = bool(report.get("gateway", {}).get("reachable"))
    db_ok = bool(report.get("database", {}).get("reachable"))
    logs_ok = bool(report.get("container_logs", {}).get("available"))

    if not gateway_ok and not db_ok and not logs_ok:
        return 3

    findings = report.get("analysis", {}).get("findings", [])
    if any(finding.get("severity") == "critical" for finding in findings):
        return 1

    return 0


def build_report(
    args: argparse.Namespace,
    gateway_data: dict[str, Any],
    db_data: dict[str, Any],
    logs_data: dict[str, Any],
) -> dict[str, Any]:
    report = {
        "meta": {
            "generated_at": utc_now_iso(),
            "tool_version": TOOL_VERSION,
            "gateway_url": None if args.no_gateway else args.gateway_url,
            "db_host": None if args.no_db else f"{args.db_host}:{args.db_port}",
            "filters": {
                "job_id": args.job_id,
                "dag_id": args.dag_id,
                "long_running_threshold_min": args.long_running_threshold,
            },
        },
        "gateway": gateway_data,
        "database": db_data,
        "container_logs": logs_data,
    }
    report["analysis"] = analyze(gateway_data, db_data, args.long_running_threshold)
    return report


def main() -> int:
    try:
        args = parse_args()
    except Exception as exc:  # noqa: BLE001
        report = {
            "meta": {
                "generated_at": utc_now_iso(),
                "tool_version": TOOL_VERSION,
                "gateway_url": None,
                "db_host": None,
                "filters": {
                    "job_id": None,
                    "dag_id": None,
                    "long_running_threshold_min": None,
                },
            },
            "gateway": {
                "reachable": False,
                "status": None,
                "debug": None,
                "errors": {},
            },
            "database": {"reachable": False, "error": None},
            "container_logs": {
                "available": False,
                "source": None,
                "exit_code": None,
                "error": None,
                "skipped": False,
                "line_count": 0,
                "lines_included": False,
                "lines": [],
            },
            "analysis": {"findings": []},
            "error": {"kind": exc.__class__.__name__, "message": str(exc)},
        }
        print(json.dumps(report, indent=2))
        return 2

    gateway_data = (
        {
            "reachable": False,
            "status": None,
            "debug": None,
            "errors": {},
            "skipped": True,
        }
        if args.no_gateway
        else collect_gateway(args.gateway_url, args.gateway_timeout)
    )
    db_data = (
        {
            "reachable": False,
            "dag_state_distribution": [],
            "job_state_distribution": [],
            "dag_classification": [],
            "stuck_active_jobs": [],
            "stale_active_by_queue": [],
            "long_completed_jobs": [],
            "job_detail": None,
            "dag_jobs": None,
            "error": None,
            "skipped": True,
        }
        if args.no_db
        else collect_database(args, gateway_data)
    )
    logs_data = collect_logs(args)
    report = build_report(args, gateway_data, db_data, logs_data)
    print(json.dumps(report, indent=2 if args.pretty else None, sort_keys=False))
    return determine_exit_code(report)


if __name__ == "__main__":
    raise SystemExit(main())
