#!/usr/bin/env python3
"""
End-to-end gateway stress tool for scheduler and LLM failure testing.

This harness exercises the full job path:

1. Discover local files or existing S3 assets
2. Optionally upload local files to S3/MinIO using the Marie storage layer
3. Submit planner-aware jobs through the gateway
4. Consume scheduler lifecycle events from RabbitMQ
5. Report submit, queue, execution, end-to-end, and failure metrics

The tool is intentionally separate from `gateway_stresser.py`. The existing
stresser remains a lightweight request benchmark; this script owns the
scheduler/S3/RabbitMQ concerns needed for full end-to-end validation.
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import logging
import math
import os
import re
import statistics
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from tools.stress.gateway_e2e_reporting import (
    REPORT_FORMAT_CHOICES,
    build_latency_stats,
    render_dry_run_report,
    render_final_report,
    render_live_report,
    resolve_report_format,
    write_text_atomically,
)

IMPORT_ERROR: Optional[ImportError] = None
try:
    import aiohttp
    import grpc
    import pika

    from marie import Client, Document, DocumentArray
    from marie.storage import S3StorageHandler, StorageManager
    from marie.utils.asset_util import s3_asset_path
except ImportError as exc:
    IMPORT_ERROR = exc


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("GatewayE2EStresser")


DEFAULT_EXTENSIONS = (
    ".tif",
    ".tiff",
    ".png",
    ".jpg",
    ".jpeg",
    ".pdf",
)

MOCK_FAILURE_MODES = ("exception", "timeout", "random")
REDACTED_SECRET = "<redacted>"


def _parse_duration_seconds(duration_str: str) -> float:
    """Parse duration strings like '30s', '2m', or '1h' into seconds."""
    raw = duration_str.strip().lower()
    if not raw:
        raise ValueError("Duration cannot be empty")

    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([hms])?", raw)
    if not match:
        raise ValueError("Invalid duration. Use values like '30s', '2m', or '1h'")

    value = float(match.group(1))
    unit = match.group(2) or "s"
    if unit == "h":
        return value * 3600.0
    if unit == "m":
        return value * 60.0
    return value


def _parse_csv_values(raw: Optional[str]) -> List[str]:
    if raw is None:
        return []
    values = [item.strip() for item in raw.split(",")]
    return [item for item in values if item]


@dataclass(frozen=True)
class InputAsset:
    source_name: str
    source_path: str
    local_path: Optional[Path] = None
    existing_s3_uri: Optional[str] = None


def _render_template_value(value: Any, template_vars: Dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {
            key: _render_template_value(item, template_vars)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_render_template_value(item, template_vars) for item in value]
    if isinstance(value, str):
        rendered = value
        for key, template_value in template_vars.items():
            rendered = rendered.replace(f"{{{{{key}}}}}", template_value)
        return rendered
    return value


def _redact_secret_value(value: Any, secret: str) -> Any:
    if not secret:
        return value
    if isinstance(value, dict):
        return {key: _redact_secret_value(item, secret) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_secret_value(item, secret) for item in value]
    if isinstance(value, str):
        return value.replace(secret, REDACTED_SECRET)
    return value


def _inject_purge_annotators_feature(
    metadata: Dict[str, Any], purge_annotators: List[str]
) -> None:
    if not purge_annotators:
        return

    features = metadata.get("features")
    if features is None:
        features = []
    if not isinstance(features, list):
        raise ValueError("metadata.features must be a list when provided")

    for feature in features:
        if (
            isinstance(feature, dict)
            and feature.get("type") == "pipeline"
            and "purge_annotators" in feature
        ):
            existing = feature.get("purge_annotators")
            if not isinstance(existing, list):
                raise ValueError("pipeline purge_annotators must be a list")
            merged = list(existing)
            for annotator_name in purge_annotators:
                if annotator_name not in merged:
                    merged.append(annotator_name)
            feature["purge_annotators"] = merged
            metadata["features"] = features
            return

    features.append(
        {
            "type": "pipeline",
            "name": "stress-purge-annotators",
            "purge_annotators": list(purge_annotators),
        }
    )
    metadata["features"] = features


def _extract_template(
    template_payload: Dict[str, Any],
) -> Tuple[Optional[str], Dict[str, Any]]:
    if not isinstance(template_payload, dict):
        raise ValueError("Template payload must be a JSON object")

    invoke_action = template_payload.get("invoke_action")
    if isinstance(invoke_action, dict):
        metadata = invoke_action.get("metadata", {})
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            raise ValueError("Template invoke_action.metadata must be a JSON object")
        name = invoke_action.get("name")
        return (str(name) if name else None, metadata)

    return None, template_payload


def _resolve_inputs(
    *,
    input_glob: Optional[str],
    input_dir: Optional[str],
    input_manifest: Optional[str],
    extensions: Tuple[str, ...] = DEFAULT_EXTENSIONS,
) -> List[Path]:
    paths: List[Path] = []

    if input_manifest:
        manifest_path = Path(input_manifest).expanduser()
        for raw_line in manifest_path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            paths.append(Path(line).expanduser())
    elif input_glob:
        paths.extend(
            sorted(Path(match).expanduser() for match in glob.glob(input_glob))
        )
    elif input_dir:
        base = Path(input_dir).expanduser()
        for ext in extensions:
            paths.extend(sorted(base.rglob(f"*{ext}")))
    else:
        raise ValueError(
            "One of --input-glob, --input-dir, or --input-manifest is required"
        )

    normalized: List[Path] = []
    seen: set[str] = set()
    for path in paths:
        resolved = path.resolve()
        if not resolved.is_file():
            continue
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        normalized.append(resolved)

    return normalized


def _resolve_s3_inputs(
    *,
    s3_uri: Optional[str],
    s3_uri_manifest: Optional[str],
) -> List[str]:
    uris: List[str] = []

    if s3_uri_manifest:
        manifest_path = Path(s3_uri_manifest).expanduser()
        for raw_line in manifest_path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            uris.append(line)
    elif s3_uri:
        uris.append(s3_uri.strip())

    normalized: List[str] = []
    seen: set[str] = set()
    for value in uris:
        if not value.startswith("s3://"):
            raise ValueError(f"Expected s3:// URI, got: {value}")
        if value in seen:
            continue
        seen.add(value)
        normalized.append(value)

    return normalized


def _build_input_assets(
    *,
    input_glob: Optional[str],
    input_dir: Optional[str],
    input_manifest: Optional[str],
    s3_uri: Optional[str],
    s3_uri_manifest: Optional[str],
) -> List[InputAsset]:
    local_inputs = (
        _resolve_inputs(
            input_glob=input_glob,
            input_dir=input_dir,
            input_manifest=input_manifest,
        )
        if any((input_glob, input_dir, input_manifest))
        else []
    )
    remote_inputs = (
        _resolve_s3_inputs(
            s3_uri=s3_uri,
            s3_uri_manifest=s3_uri_manifest,
        )
        if any((s3_uri, s3_uri_manifest))
        else []
    )

    assets: List[InputAsset] = []
    for path in local_inputs:
        assets.append(
            InputAsset(
                source_name=path.name,
                source_path=str(path),
                local_path=path,
            )
        )
    for uri in remote_inputs:
        parsed = urlparse(uri)
        source_name = Path(parsed.path).name or parsed.netloc or "remote-object"
        assets.append(
            InputAsset(
                source_name=source_name,
                source_path=uri,
                existing_s3_uri=uri,
            )
        )

    return assets


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    idx = int(math.ceil((pct / 100.0) * len(sorted_values))) - 1
    idx = min(max(idx, 0), len(sorted_values) - 1)
    return sorted_values[idx]


def _extract_ref_id_from_event(message: Dict[str, Any]) -> Optional[str]:
    payload = message.get("payload")
    if payload is None:
        return None

    try:
        payload_dict = json.loads(payload) if isinstance(payload, str) else payload
    except Exception:
        return None

    if not isinstance(payload_dict, dict):
        return None

    metadata = payload_dict.get("metadata")
    if isinstance(metadata, dict):
        ref_id = metadata.get("ref_id")
        if isinstance(ref_id, str) and ref_id:
            return ref_id

    ref_id = payload_dict.get("ref_id")
    if isinstance(ref_id, str) and ref_id:
        return ref_id
    return None


def _extract_failure_error(message: Any) -> str:
    if not isinstance(message, dict):
        return "Queue reported failed event"

    for key in ("error", "message", "reason"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    payload = message.get("payload")
    if payload:
        try:
            payload_dict = json.loads(payload) if isinstance(payload, str) else payload
            if isinstance(payload_dict, dict):
                for key in ("error", "message", "reason", "status"):
                    value = payload_dict.get(key)
                    if isinstance(value, str) and value.strip():
                        return value.strip()
        except Exception:
            pass

    event_name = message.get("event")
    if isinstance(event_name, str) and event_name.strip():
        return f"Queue event {event_name}"
    return "Queue reported failed event"


def _now() -> float:
    return time.time()


def _normalize_epoch_seconds(raw: float) -> float:
    value = float(raw)
    abs_value = abs(value)
    if abs_value >= 1e17:
        return value / 1_000_000_000.0
    if abs_value >= 1e14:
        return value / 1_000_000.0
    if abs_value >= 1e11:
        return value / 1_000.0
    return value


def _parse_optional_epoch_seconds(raw: Any) -> Optional[float]:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return _normalize_epoch_seconds(float(raw))
    if isinstance(raw, str):
        value = raw.strip()
        if not value:
            return None
        try:
            return _normalize_epoch_seconds(float(value))
        except ValueError:
            normalized = value.replace("Z", "+00:00")
            dt = datetime.fromisoformat(normalized)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            return dt.timestamp()
    raise ValueError(f"Unsupported timestamp value: {raw!r}")


def _format_epoch_seconds(epoch_seconds: float) -> str:
    return datetime.fromtimestamp(epoch_seconds, tz=timezone.utc).isoformat()


def _extract_event_timestamp(message: Dict[str, Any]) -> float:
    try:
        event_timestamp = _parse_optional_epoch_seconds(message.get("timestamp"))
    except ValueError:
        event_timestamp = None
    return event_timestamp if event_timestamp is not None else _now()


@dataclass
class SubmitResult:
    request_id: str
    success: bool
    latency_ms: float
    job_id: Optional[str] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class JobRun:
    request_id: str
    job_index: int
    source_path: str
    source_name: str
    input_mode: str
    ref_id: str
    ref_type: str
    s3_uri: str
    planner: str
    job_name: str
    fault_profile: str
    llm_pool_id: Optional[str] = None
    mock_failure_rate: Optional[float] = None
    mock_failure_mode: Optional[str] = None
    force_fail: bool = False
    soft_sla_at: Optional[float] = None
    hard_sla_at: Optional[float] = None
    sla_bucket_index: int = 0
    soft_sla_offset_seconds: Optional[float] = None
    hard_sla_offset_seconds: Optional[float] = None
    created_at: float = field(default_factory=_now)
    upload_started_at: Optional[float] = None
    upload_finished_at: Optional[float] = None
    submit_started_at: Optional[float] = None
    submit_finished_at: Optional[float] = None
    submit_latency_ms: Optional[float] = None
    job_id: Optional[str] = None
    submit_error_type: Optional[str] = None
    submit_error_message: Optional[str] = None
    scheduled_at: Optional[float] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    failed_at: Optional[float] = None
    failure_reason: Optional[str] = None
    raw_events: List[str] = field(default_factory=list)

    @property
    def terminal_status(self) -> Optional[str]:
        if self.submit_error_type:
            return "submit_failed"
        if self.completed_at is not None:
            return "completed"
        if self.failed_at is not None:
            return "failed"
        return None

    @property
    def terminal_at(self) -> Optional[float]:
        if self.completed_at is not None:
            return self.completed_at
        if self.failed_at is not None:
            return self.failed_at
        return None

    @property
    def queue_wait_ms(self) -> Optional[float]:
        if self.submit_finished_at is None:
            return None
        if self.started_at is not None:
            return max(0.0, (self.started_at - self.submit_finished_at) * 1000.0)
        if self.scheduled_at is not None:
            return max(0.0, (self.scheduled_at - self.submit_finished_at) * 1000.0)
        return None

    @property
    def scheduling_ms(self) -> Optional[float]:
        if self.submit_finished_at is None or self.scheduled_at is None:
            return None
        return max(0.0, (self.scheduled_at - self.submit_finished_at) * 1000.0)

    @property
    def execution_ms(self) -> Optional[float]:
        if self.started_at is None or self.terminal_at is None:
            return None
        return (self.terminal_at - self.started_at) * 1000.0

    @property
    def end_to_end_ms(self) -> Optional[float]:
        if self.submit_started_at is None or self.terminal_at is None:
            return None
        return (self.terminal_at - self.submit_started_at) * 1000.0

    def _sla_status(self, deadline_at: Optional[float]) -> Optional[str]:
        if deadline_at is None:
            return None

        status = self.terminal_status
        if status is None:
            return "pending"
        if status == "submit_failed":
            return "submit_failed"
        if status != "completed":
            if self.terminal_at is not None and self.terminal_at > deadline_at:
                return "terminal_failed_after_deadline"
            return "terminal_failed"
        if self.terminal_at is not None and self.terminal_at <= deadline_at:
            return "met"
        return "deadline_missed"

    def _sla_met(self, deadline_at: Optional[float]) -> Optional[bool]:
        status = self._sla_status(deadline_at)
        if status is None or status == "pending":
            return None
        return status == "met"

    def _sla_lateness_ms(self, deadline_at: Optional[float]) -> Optional[float]:
        if deadline_at is None or self.terminal_at is None:
            return None
        return max(0.0, (self.terminal_at - deadline_at) * 1000.0)

    @property
    def soft_sla_status(self) -> Optional[str]:
        return self._sla_status(self.soft_sla_at)

    @property
    def hard_sla_status(self) -> Optional[str]:
        return self._sla_status(self.hard_sla_at)

    @property
    def soft_sla_met(self) -> Optional[bool]:
        return self._sla_met(self.soft_sla_at)

    @property
    def hard_sla_met(self) -> Optional[bool]:
        return self._sla_met(self.hard_sla_at)

    @property
    def soft_sla_lateness_ms(self) -> Optional[float]:
        return self._sla_lateness_ms(self.soft_sla_at)

    @property
    def hard_sla_lateness_ms(self) -> Optional[float]:
        return self._sla_lateness_ms(self.hard_sla_at)


@dataclass
class E2EMetrics:
    total_jobs: int = 0
    submitted_jobs: int = 0
    completed_jobs: int = 0
    failed_jobs: int = 0
    submit_failed_jobs: int = 0
    event_timeout_jobs: int = 0
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    submit_latencies_ms: List[float] = field(default_factory=list)
    scheduling_latencies_ms: List[float] = field(default_factory=list)
    queue_wait_ms: List[float] = field(default_factory=list)
    execution_latencies_ms: List[float] = field(default_factory=list)
    end_to_end_latencies_ms: List[float] = field(default_factory=list)
    failure_reasons: Dict[str, int] = field(default_factory=dict)
    soft_sla_configured_jobs: int = 0
    soft_sla_met_jobs: int = 0
    soft_sla_missed_jobs: int = 0
    hard_sla_configured_jobs: int = 0
    hard_sla_met_jobs: int = 0
    hard_sla_missed_jobs: int = 0
    soft_sla_lateness_ms: List[float] = field(default_factory=list)
    hard_sla_lateness_ms: List[float] = field(default_factory=list)

    @property
    def throughput(self) -> float:
        if self.start_time is None or self.end_time is None:
            return 0.0
        duration = self.end_time - self.start_time
        if duration <= 0:
            return 0.0
        return self.total_jobs / duration

    @property
    def soft_sla_compliance_pct(self) -> float:
        if self.soft_sla_configured_jobs <= 0:
            return 0.0
        return (self.soft_sla_met_jobs / self.soft_sla_configured_jobs) * 100.0

    @property
    def hard_sla_compliance_pct(self) -> float:
        if self.hard_sla_configured_jobs <= 0:
            return 0.0
        return (self.hard_sla_met_jobs / self.hard_sla_configured_jobs) * 100.0


@dataclass
class DebugSnapshot:
    stage: str
    captured_at: float
    ok: bool
    status_code: Optional[int] = None
    error: Optional[str] = None
    scheduler_running: Optional[bool] = None
    scheduler_paused: Optional[bool] = None
    active_dags_count: Optional[int] = None
    max_concurrent_dags: Optional[int] = None
    fetch_counter: Optional[int] = None
    pending_requests: Optional[int] = None
    request_queue_size: Optional[int] = None
    event_queue_size: Optional[int] = None
    queue_status: Optional[Dict[str, Any]] = None
    queue_status_error: Optional[str] = None
    llm_dispatch: Optional[Dict[str, Any]] = None
    llm_dispatch_registered_dispatchers: Optional[int] = None
    llm_dispatch_running_dispatchers: Optional[int] = None


def _coerce_gateway_debug_payload(payload: Any) -> Any:
    if (
        isinstance(payload, dict)
        and payload.get("status") in {"OK", "error"}
        and "result" in payload
    ):
        return payload.get("result")
    return payload


def _build_debug_snapshot(
    *,
    stage: str,
    payload: Optional[Dict[str, Any]] = None,
    captured_at: Optional[float] = None,
    status_code: Optional[int] = None,
    error: Optional[str] = None,
) -> DebugSnapshot:
    captured = captured_at if captured_at is not None else _now()
    normalized_payload = payload if isinstance(payload, dict) else {}
    scheduler_info = normalized_payload.get("scheduler_info")
    counters = normalized_payload.get("counters")
    queues = normalized_payload.get("queues")
    queue_status = normalized_payload.get("queue_status")
    llm_dispatch = normalized_payload.get("llm_dispatch")

    if not isinstance(scheduler_info, dict):
        scheduler_info = {}
    if not isinstance(counters, dict):
        counters = {}
    if not isinstance(queues, dict):
        queues = {}
    if not isinstance(queue_status, dict):
        queue_status = None
    if not isinstance(llm_dispatch, dict):
        llm_dispatch = None

    return DebugSnapshot(
        stage=stage,
        captured_at=captured,
        ok=error is None and bool(normalized_payload),
        status_code=status_code,
        error=error,
        scheduler_running=(
            bool(scheduler_info.get("running")) if "running" in scheduler_info else None
        ),
        scheduler_paused=(
            bool(scheduler_info.get("paused")) if "paused" in scheduler_info else None
        ),
        active_dags_count=(
            int(scheduler_info.get("active_dags_count"))
            if scheduler_info.get("active_dags_count") is not None
            else None
        ),
        max_concurrent_dags=(
            int(scheduler_info.get("max_concurrent_dags"))
            if scheduler_info.get("max_concurrent_dags") is not None
            else None
        ),
        fetch_counter=(
            int(counters.get("fetch_counter"))
            if counters.get("fetch_counter") is not None
            else None
        ),
        pending_requests=(
            int(counters.get("pending_requests"))
            if counters.get("pending_requests") is not None
            else None
        ),
        request_queue_size=(
            int(queues.get("request_queue_size"))
            if queues.get("request_queue_size") is not None
            else None
        ),
        event_queue_size=(
            int(queues.get("event_queue_size"))
            if queues.get("event_queue_size") is not None
            else None
        ),
        queue_status=queue_status,
        queue_status_error=(
            str(normalized_payload.get("queue_status_error"))
            if normalized_payload.get("queue_status_error") is not None
            else None
        ),
        llm_dispatch=llm_dispatch,
        llm_dispatch_registered_dispatchers=(
            int(llm_dispatch.get("registered_dispatchers"))
            if llm_dispatch and llm_dispatch.get("registered_dispatchers") is not None
            else None
        ),
        llm_dispatch_running_dispatchers=(
            int(llm_dispatch.get("running_dispatchers"))
            if llm_dispatch and llm_dispatch.get("running_dispatchers") is not None
            else None
        ),
    )


class SchedulerEventConsumer:
    def __init__(
        self,
        *,
        connection_config: Dict[str, Any],
        api_key: str,
        callback,
        logger: logging.Logger,
    ) -> None:
        self.connection_config = connection_config
        self.api_key = api_key
        self.callback = callback
        self.logger = logger
        self.stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self.stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name=f"gateway-e2e-events-{self.api_key[:8]}",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self.stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)

    def _run(self) -> None:
        retry_count = 0
        backoff = 1.0
        max_backoff = 30.0

        while not self.stop_event.is_set():
            connection = None
            channel = None
            try:
                hostname = self.connection_config.get("hostname", "localhost")
                port = int(self.connection_config.get("port", 5672))
                username = self.connection_config.get("username", "guest")
                password = self.connection_config.get("password", "guest")
                tls_enabled = bool(self.connection_config.get("tls", False))

                scheme = "amqps" if tls_enabled else "amqp"
                url = f"{scheme}://{username}:{password}@{hostname}:{port}?connection_attempts=1&heartbeat=30"
                parameters = pika.URLParameters(url)
                parameters.heartbeat = 30
                parameters.socket_timeout = 15
                parameters.blocked_connection_timeout = 300
                parameters.client_properties = {
                    "connection_name": f"{self.api_key}-gateway-e2e-stresser"
                }

                connection = pika.BlockingConnection(parameters)
                channel = connection.channel()
                channel.basic_qos(prefetch_count=50)

                exchange = f"{self.api_key}.events"
                channel.exchange_declare(exchange, durable=True, exchange_type="topic")
                queue_result = channel.queue_declare(
                    queue="", exclusive=True, auto_delete=True
                )
                queue_name = queue_result.method.queue
                channel.queue_bind(queue=queue_name, exchange=exchange, routing_key="#")

                retry_count = 0
                backoff = 1.0
                self.logger.info("Connected to scheduler events exchange %s", exchange)

                def on_message(ch, method_frame, _header_frame, body):
                    try:
                        payload = json.loads(body.decode("utf-8"))
                        self.callback(payload)
                    except Exception as exc:  # pragma: no cover - log-only path
                        self.logger.error("Failed to process scheduler event: %r", exc)
                    finally:
                        ch.basic_ack(delivery_tag=method_frame.delivery_tag)

                channel.basic_consume(queue_name, on_message)

                while not self.stop_event.is_set():
                    connection.process_data_events(time_limit=1)
                try:
                    channel.stop_consuming()
                except Exception:
                    pass
                break
            except (
                pika.exceptions.AMQPConnectionError,
                ConnectionResetError,
                OSError,
            ) as exc:
                retry_count += 1
                self.logger.warning(
                    "Scheduler event connection error (attempt %s): %s",
                    retry_count,
                    exc,
                )
                time.sleep(min(max_backoff, backoff))
                backoff = min(max_backoff, backoff * 2)
            except Exception as exc:  # pragma: no cover - log-only path
                self.logger.exception(
                    "Unexpected scheduler event consumer failure: %r", exc
                )
                time.sleep(min(max_backoff, backoff))
                backoff = min(max_backoff, backoff * 2)
            finally:
                try:
                    if channel and getattr(channel, "is_open", False):
                        channel.close()
                except Exception:
                    pass
                try:
                    if connection and getattr(connection, "is_open", False):
                        connection.close()
                except Exception:
                    pass


class GatewayE2EStresser:
    def __init__(
        self,
        *,
        gateway_host: str,
        gateway_port: int,
        http_port: Optional[int],
        protocol: str,
        endpoint: str,
        api_key: str,
        queue_name: str,
        planner: str,
        input_assets: List[InputAsset],
        job_count: Optional[int],
        run_time_seconds: Optional[float],
        submit_concurrency: int,
        submit_rate: float,
        timeout: float,
        terminal_timeout: float,
        s3_config: Optional[Dict[str, Any]],
        queue_config: Optional[Dict[str, Any]],
        metadata_template: Optional[Dict[str, Any]],
        template_job_name: Optional[str],
        fault_profile: str,
        aimock_admin_url: Optional[str] = None,
        soft_sla_seconds: Optional[float] = None,
        hard_sla_seconds: Optional[float] = None,
        soft_sla_step_seconds: Optional[float] = None,
        hard_sla_step_seconds: Optional[float] = None,
        sla_step_every_jobs: int = 1,
        sla_step_cycle: Optional[int] = None,
        min_soft_sla_compliance_pct: Optional[float] = None,
        min_hard_sla_compliance_pct: Optional[float] = None,
        ref_type: Optional[str] = None,
        policy: str = "allow_all",
        project_id: Optional[str] = None,
        llm_pool_id: Optional[str] = None,
        llm_pool_cycle: Optional[List[str]] = None,
        purge_annotators: Optional[List[str]] = None,
        mock_failure_rate: Optional[float] = None,
        mock_failure_mode: str = "exception",
        force_failure_every: Optional[int] = None,
        upload_companion_meta: bool = True,
        batch_size: int = 1,
        debug_sample_interval: float = 0.0,
        dry_run_preview_count: int = 3,
        progress_interval: float = 5.0,
        live_report_path: Optional[str] = None,
        live_report_format: str = "auto",
    ) -> None:
        self.gateway_host = gateway_host
        self.gateway_port = gateway_port
        self.http_port = (
            http_port
            if http_port is not None
            else (gateway_port if protocol == "http" else None)
        )
        self.protocol = protocol
        self.endpoint = endpoint
        self.api_key = api_key
        self.queue_name = queue_name or template_job_name or planner
        self.planner = planner
        self.input_assets = input_assets
        self.job_count = job_count if job_count and job_count > 0 else None
        self.run_time_seconds = (
            float(run_time_seconds)
            if run_time_seconds is not None and run_time_seconds > 0
            else None
        )
        self.submit_concurrency = submit_concurrency
        self.submit_rate = submit_rate
        self.timeout = timeout
        self.terminal_timeout = terminal_timeout
        self.s3_config = s3_config or {}
        self.queue_config = queue_config
        self.metadata_template = metadata_template or {}
        self.fault_profile = fault_profile
        self.aimock_admin_url = (
            aimock_admin_url.rstrip("/") if aimock_admin_url else None
        )
        self.soft_sla_seconds = soft_sla_seconds
        self.hard_sla_seconds = hard_sla_seconds
        self.soft_sla_step_seconds = soft_sla_step_seconds
        self.hard_sla_step_seconds = hard_sla_step_seconds
        self.sla_step_every_jobs = max(1, sla_step_every_jobs)
        self.sla_step_cycle = sla_step_cycle
        self.min_soft_sla_compliance_pct = min_soft_sla_compliance_pct
        self.min_hard_sla_compliance_pct = min_hard_sla_compliance_pct
        self.ref_type = ref_type or self.queue_name
        self.policy = policy
        self.project_id = project_id or api_key
        self.llm_pool_id = llm_pool_id.strip() if llm_pool_id else None
        self.llm_pool_cycle = [
            item.strip() for item in llm_pool_cycle or [] if item.strip()
        ]
        self.purge_annotators = []
        seen_purge_annotators: set[str] = set()
        for item in purge_annotators or []:
            annotator_name = item.strip()
            if annotator_name and annotator_name not in seen_purge_annotators:
                self.purge_annotators.append(annotator_name)
                seen_purge_annotators.add(annotator_name)
        self.mock_failure_rate = mock_failure_rate
        self.mock_failure_mode = mock_failure_mode
        self.force_failure_every = force_failure_every
        self.upload_companion_meta = upload_companion_meta
        self.batch_size = batch_size
        self.debug_sample_interval = max(0.0, debug_sample_interval)
        self.dry_run_preview_count = max(1, dry_run_preview_count)
        self.progress_interval = progress_interval
        self.live_report_path = live_report_path
        self.live_report_format = live_report_format
        self.requires_upload = any(
            asset.local_path is not None for asset in input_assets
        )
        if self.job_count is None and self.run_time_seconds is None:
            raise ValueError("Either job_count or run_time_seconds must be provided")
        if self.job_count is not None and self.run_time_seconds is not None:
            raise ValueError("job_count and run_time_seconds are mutually exclusive")
        if self.submit_rate <= 0:
            raise ValueError("--submit-rate must be greater than zero")
        if (
            self.soft_sla_seconds is not None
            and self.hard_sla_seconds is not None
            and self.soft_sla_seconds > self.hard_sla_seconds
        ):
            raise ValueError(
                "--soft-sla-seconds must be less than or equal to --hard-sla-seconds"
            )
        if self.soft_sla_step_seconds is not None and self.soft_sla_seconds is None:
            raise ValueError("--soft-sla-step-seconds requires --soft-sla-seconds")
        if self.hard_sla_step_seconds is not None and self.hard_sla_seconds is None:
            raise ValueError("--hard-sla-step-seconds requires --hard-sla-seconds")
        if self.sla_step_cycle is not None and self.sla_step_cycle <= 0:
            raise ValueError("--sla-step-cycle must be greater than zero")
        if self.mock_failure_rate is not None and not (
            0.0 <= self.mock_failure_rate <= 1.0
        ):
            raise ValueError("--mock-failure-rate must be between 0 and 1 inclusive")
        if self.mock_failure_mode not in MOCK_FAILURE_MODES:
            raise ValueError(
                f"--mock-failure-mode must be one of {', '.join(MOCK_FAILURE_MODES)}"
            )
        if self.force_failure_every is not None and self.force_failure_every <= 0:
            raise ValueError("--force-failure-every must be greater than zero")
        if self.llm_pool_id and self.llm_pool_cycle:
            raise ValueError(
                "--llm-pool-id and --llm-pool-cycle are mutually exclusive"
            )
        if llm_pool_cycle is not None and not self.llm_pool_cycle:
            raise ValueError("--llm-pool-cycle must contain at least one pool ID")

        self.metrics = E2EMetrics(total_jobs=self.job_count or 0)
        self._logger = logging.getLogger(self.__class__.__name__)
        self._runs_by_request_id: Dict[str, JobRun] = {}
        self._runs_by_job_id: Dict[str, JobRun] = {}
        self._runs_by_ref_id: Dict[str, JobRun] = {}
        self._debug_samples: List[DebugSnapshot] = []
        self._verification_errors: List[str] = []
        self._state_lock = threading.Lock()
        self._http_session: Optional[aiohttp.ClientSession] = None
        self._event_consumer: Optional[SchedulerEventConsumer] = None

    @property
    def verification_errors(self) -> List[str]:
        return list(self._verification_errors)

    @property
    def run_mode(self) -> str:
        return "fixed-count" if self.job_count is not None else "duration"

    @property
    def estimated_job_count(self) -> Optional[int]:
        if self.job_count is not None:
            return self.job_count
        if self.run_time_seconds is None:
            return None
        return max(1, int(math.ceil(self.run_time_seconds * self.submit_rate)))

    def _dry_run_submission_count(self) -> int:
        if self.job_count is not None:
            return self.job_count
        estimated = self.estimated_job_count or 0
        return min(self.dry_run_preview_count, max(1, estimated))

    def _build_sla_summary(
        self,
        runs: List[JobRun],
        *,
        now: float,
        deadline_attr: str,
        status_attr: str,
        met_attr: str,
        lateness_attr: str,
        configured_seconds: Optional[float],
        step_seconds: Optional[float],
        min_compliance_pct: Optional[float],
    ) -> Optional[Dict[str, Any]]:
        configured_jobs = 0
        terminal_evaluated_jobs = 0
        met_jobs = 0
        missed_jobs = 0
        failed_jobs = 0
        pending_jobs = 0
        overdue_open_jobs = 0
        lateness_values: List[float] = []

        for run in runs:
            deadline_at = getattr(run, deadline_attr)
            if deadline_at is None:
                continue
            configured_jobs += 1
            status_value = getattr(run, status_attr)
            met_value = getattr(run, met_attr)

            if status_value == "pending":
                pending_jobs += 1
                if now > deadline_at:
                    overdue_open_jobs += 1
                continue

            terminal_evaluated_jobs += 1
            if met_value is True:
                met_jobs += 1
            else:
                if status_value == "terminal_failed":
                    failed_jobs += 1
                else:
                    missed_jobs += 1
                lateness_ms = getattr(run, lateness_attr)
                if lateness_ms is not None and lateness_ms > 0:
                    lateness_values.append(lateness_ms)

        if configured_jobs == 0 and configured_seconds is None:
            return None

        terminal_compliance_pct = (
            (met_jobs / terminal_evaluated_jobs) * 100.0
            if terminal_evaluated_jobs > 0
            else None
        )

        return {
            "configured_seconds": configured_seconds,
            "step_seconds": step_seconds,
            "step_every_jobs": self.sla_step_every_jobs,
            "step_cycle": self.sla_step_cycle,
            "min_compliance_pct": min_compliance_pct,
            "configured_jobs": configured_jobs,
            "terminal_evaluated_jobs": terminal_evaluated_jobs,
            "met_jobs": met_jobs,
            "missed_jobs": missed_jobs,
            "failed_jobs": failed_jobs,
            "pending_jobs": pending_jobs,
            "overdue_open_jobs": overdue_open_jobs,
            "terminal_compliance_pct": terminal_compliance_pct,
            "lateness_stats_ms": build_latency_stats(lateness_values, _percentile),
        }

    def _build_sla_payload(
        self,
        runs: List[JobRun],
        *,
        now: float,
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        return {
            "soft": self._build_sla_summary(
                runs,
                now=now,
                deadline_attr="soft_sla_at",
                status_attr="soft_sla_status",
                met_attr="soft_sla_met",
                lateness_attr="soft_sla_lateness_ms",
                configured_seconds=self.soft_sla_seconds,
                step_seconds=self.soft_sla_step_seconds,
                min_compliance_pct=self.min_soft_sla_compliance_pct,
            ),
            "hard": self._build_sla_summary(
                runs,
                now=now,
                deadline_attr="hard_sla_at",
                status_attr="hard_sla_status",
                met_attr="hard_sla_met",
                lateness_attr="hard_sla_lateness_ms",
                configured_seconds=self.hard_sla_seconds,
                step_seconds=self.hard_sla_step_seconds,
                min_compliance_pct=self.min_hard_sla_compliance_pct,
            ),
        }

    def _build_live_status_payload(self, status: str = "running") -> Dict[str, Any]:
        now = _now()
        with self._state_lock:
            runs = list(self._runs_by_request_id.values())
            latest_debug_sample = (
                self._debug_samples[-1] if self._debug_samples else None
            )

        created = len(runs)
        submitted = 0
        completed = 0
        failed = 0
        submit_failed = 0
        event_timeout = 0
        pending_submit = 0

        for run in runs:
            if run.job_id:
                submitted += 1
            if run.terminal_status == "completed":
                completed += 1
            elif run.terminal_status == "failed":
                if run.failure_reason == "terminal event timeout":
                    event_timeout += 1
                else:
                    failed += 1
            elif run.terminal_status == "submit_failed":
                submit_failed += 1
            elif run.job_id is None and run.submit_error_type is None:
                pending_submit += 1

        inflight_jobs = max(submitted - completed - failed - event_timeout, 0)
        terminal_jobs = completed + failed + submit_failed + event_timeout
        open_jobs = max(created - terminal_jobs, 0)
        submit_acceptance_pct = (submitted / created) * 100.0 if created > 0 else None
        terminal_success_pct = (
            (completed / terminal_jobs) * 100.0 if terminal_jobs > 0 else None
        )
        completion_pct = ((completed / created) * 100.0) if created > 0 else None

        submit_latency_ms = [
            run.submit_latency_ms for run in runs if run.submit_latency_ms is not None
        ]
        scheduling_ms = [
            run.scheduling_ms for run in runs if run.scheduling_ms is not None
        ]
        queue_wait_ms = [
            run.queue_wait_ms for run in runs if run.queue_wait_ms is not None
        ]
        execution_ms = [
            run.execution_ms for run in runs if run.execution_ms is not None
        ]
        end_to_end_ms = [
            run.end_to_end_ms for run in runs if run.end_to_end_ms is not None
        ]

        elapsed_seconds = None
        if self.metrics.start_time is not None:
            finished_at = (
                self.metrics.end_time if self.metrics.end_time is not None else now
            )
            elapsed_seconds = max(0.0, finished_at - self.metrics.start_time)

        throughput_created = None
        throughput_completed = None
        if elapsed_seconds and elapsed_seconds > 0:
            throughput_created = created / elapsed_seconds
            throughput_completed = completed / elapsed_seconds

        payload: Dict[str, Any] = {
            "status": status,
            "updated_at": _format_epoch_seconds(now),
            "run_mode": self.run_mode,
            "configured_job_count": self.job_count,
            "configured_run_time_seconds": self.run_time_seconds,
            "estimated_job_count": self.estimated_job_count,
            "submit_rate": self.submit_rate,
            "submit_concurrency": self.submit_concurrency,
            "failure_simulation": {
                "mock_failure_rate": self.mock_failure_rate,
                "mock_failure_mode": self.mock_failure_mode,
                "force_failure_every": self.force_failure_every,
            },
            "progress_interval_seconds": self.progress_interval,
            "live_report_path": self.live_report_path,
            "live_report_format": (
                resolve_report_format(self.live_report_path, self.live_report_format)
                if self.live_report_path
                else None
            ),
            "elapsed_seconds": elapsed_seconds,
            "throughput_jobs_per_second": {
                "created": throughput_created,
                "completed": throughput_completed,
            },
            "debug_sampling": {
                "enabled": self.debug_sample_interval > 0,
                "interval_seconds": self.debug_sample_interval,
                "endpoint": (
                    f"http://{self.gateway_host}:{self.http_port}/api/debug"
                    if self.http_port is not None
                    else None
                ),
            },
            "run_health": {
                "target_submit_rate": self.submit_rate,
                "submit_acceptance_pct": submit_acceptance_pct,
                "terminal_success_pct": terminal_success_pct,
                "completion_pct": completion_pct,
                "open_jobs": open_jobs,
                "inflight_jobs": inflight_jobs,
                "pending_submit_jobs": pending_submit,
                "terminal_jobs": terminal_jobs,
            },
            "counts": {
                "created_jobs": created,
                "submitted_jobs": submitted,
                "completed_jobs": completed,
                "failed_jobs": failed,
                "submit_failed_jobs": submit_failed,
                "event_timeout_jobs": event_timeout,
            },
            "latency_stats_ms": {
                "submit": build_latency_stats(submit_latency_ms, _percentile),
                "scheduling": build_latency_stats(scheduling_ms, _percentile),
                "queue_wait": build_latency_stats(queue_wait_ms, _percentile),
                "execution": build_latency_stats(execution_ms, _percentile),
                "end_to_end": build_latency_stats(end_to_end_ms, _percentile),
            },
            "sla": self._build_sla_payload(runs, now=now),
            "latest_debug_sample": (
                {
                    "stage": latest_debug_sample.stage,
                    "captured_at": latest_debug_sample.captured_at,
                    "ok": latest_debug_sample.ok,
                    "status_code": latest_debug_sample.status_code,
                    "error": latest_debug_sample.error,
                    "scheduler_running": latest_debug_sample.scheduler_running,
                    "scheduler_paused": latest_debug_sample.scheduler_paused,
                    "active_dags_count": latest_debug_sample.active_dags_count,
                    "fetch_counter": latest_debug_sample.fetch_counter,
                    "request_queue_size": latest_debug_sample.request_queue_size,
                    "event_queue_size": latest_debug_sample.event_queue_size,
                    "llm_dispatch_registered_dispatchers": latest_debug_sample.llm_dispatch_registered_dispatchers,
                    "llm_dispatch_running_dispatchers": latest_debug_sample.llm_dispatch_running_dispatchers,
                }
                if latest_debug_sample is not None
                else None
            ),
            "recent_jobs": [
                {
                    "job_index": run.job_index,
                    "request_id": run.request_id,
                    "job_id": run.job_id,
                    "terminal_status": run.terminal_status,
                    "source_path": run.source_path,
                    "s3_uri": run.s3_uri,
                    "failure_reason": run.failure_reason,
                    "force_fail": run.force_fail,
                }
                for run in sorted(runs, key=lambda item: item.job_index)[-10:]
            ],
            "verification_errors": list(self._verification_errors),
        }
        if self.job_count is not None and self.job_count > 0:
            payload["progress_pct"] = {
                "created": min(100.0, (created / self.job_count) * 100.0),
                "submitted": min(100.0, (submitted / self.job_count) * 100.0),
                "completed": min(100.0, (completed / self.job_count) * 100.0),
            }
        return payload

    def _log_progress_payload(self, payload: Dict[str, Any]) -> None:
        counts = payload["counts"]
        if self.job_count is not None:
            self._logger.info(
                "Progress: submitted=%s/%s completed=%s failed=%s submit_failed=%s",
                counts["submitted_jobs"],
                self.job_count,
                counts["completed_jobs"],
                counts["failed_jobs"],
                counts["submit_failed_jobs"],
            )
            return

        self._logger.info(
            "Progress: created=%s submitted=%s completed=%s failed=%s submit_failed=%s",
            counts["created_jobs"],
            counts["submitted_jobs"],
            counts["completed_jobs"],
            counts["failed_jobs"],
            counts["submit_failed_jobs"],
        )

    def _write_live_report_payload(self, payload: Dict[str, Any]) -> None:
        if not self.live_report_path:
            return
        try:
            live_format = resolve_report_format(
                self.live_report_path, self.live_report_format
            )
            rendered = render_live_report(payload, live_format, self.progress_interval)
            write_text_atomically(self.live_report_path, rendered)
        except Exception as exc:
            self._logger.warning(
                "Failed to write live report snapshot to %s: %s",
                self.live_report_path,
                exc,
            )

    def _write_live_json_payload(self, payload: Dict[str, Any]) -> None:
        self._write_live_report_payload(payload)

    def _prepare_storage_env(self) -> None:
        bucket = self.s3_config.get("S3_STORAGE_BUCKET_NAME")
        if bucket:
            os.environ["MARIE_S3_BUCKET"] = str(bucket)

    def _setup_storage(self) -> None:
        self._prepare_storage_env()
        if not self.requires_upload:
            return
        handler = S3StorageHandler(config=self.s3_config)
        StorageManager.register_handler(handler=handler)

    def _build_submit_request(
        self, run: JobRun, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {
            "data": [
                {"id": f"{run.request_id}-{i}", "text": f"stress-{run.request_id}-{i}"}
                for i in range(self.batch_size)
            ],
            "parameters": {
                "invoke_action": {
                    "action_type": "command",
                    "command": "job",
                    "action": "submit",
                    "name": self.queue_name,
                    "api_key": self.api_key,
                    "metadata": metadata,
                }
            },
            "header": {
                "requestId": run.request_id,
                "targetExecutor": "",
            },
        }

    def build_dry_run_plan(self) -> Dict[str, Any]:
        self._prepare_storage_env()
        generated_at = _now()
        submissions: List[Dict[str, Any]] = []
        preview_job_count = self._dry_run_submission_count()
        for job_index in range(preview_job_count):
            asset = self.input_assets[job_index % len(self.input_assets)]
            run = self._build_run(asset, job_index)
            run.upload_started_at = generated_at
            run.upload_finished_at = generated_at
            run.submit_started_at = generated_at
            metadata = self._build_metadata(run, sla_anchor_at=generated_at)
            request_payload = self._build_submit_request(run, metadata)
            transport: Dict[str, Any] = {
                "protocol": self.protocol,
                "endpoint": self.endpoint,
            }
            if self.protocol == "http":
                transport["url"] = (
                    f"http://{self.gateway_host}:{self.http_port}{self.endpoint}"
                )
                transport["headers"] = {
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                }
            else:
                transport["gateway_host"] = self.gateway_host
                transport["gateway_port"] = self.gateway_port
                transport["request_size"] = self.batch_size

            submissions.append(
                {
                    "job_index": run.job_index,
                    "request_id": run.request_id,
                    "source_path": run.source_path,
                    "source_name": run.source_name,
                    "input_mode": run.input_mode,
                    "upload_planned": run.input_mode == "upload",
                    "upload_companion_meta_planned": (
                        run.input_mode == "upload"
                        and self.upload_companion_meta
                        and Path(run.source_path)
                        .with_name(f"{Path(run.source_path).name}.meta.json")
                        .exists()
                    ),
                    "s3_uri": run.s3_uri,
                    "planner": run.planner,
                    "job_name": run.job_name,
                    "fault_profile": run.fault_profile,
                    "llm_pool_id": run.llm_pool_id,
                    "mock_failure_rate": run.mock_failure_rate,
                    "mock_failure_mode": run.mock_failure_mode,
                    "force_fail": run.force_fail,
                    "sla_bucket_index": run.sla_bucket_index,
                    "soft_sla_offset_seconds": run.soft_sla_offset_seconds,
                    "hard_sla_offset_seconds": run.hard_sla_offset_seconds,
                    "metadata": _redact_secret_value(metadata, self.api_key),
                    "request_payload": _redact_secret_value(
                        request_payload, self.api_key
                    ),
                    "transport": _redact_secret_value(transport, self.api_key),
                }
            )

        return {
            "dry_run": True,
            "generated_at": _format_epoch_seconds(generated_at),
            "run_mode": self.run_mode,
            "job_count": self.job_count,
            "run_time_seconds": self.run_time_seconds,
            "estimated_job_count": self.estimated_job_count,
            "preview_job_count": preview_job_count,
            "input_assets_resolved": len(self.input_assets),
            "protocol": self.protocol,
            "endpoint": self.endpoint,
            "planner": self.planner,
            "job_name": self.queue_name,
            "fault_profile": self.fault_profile,
            "llm_pool_id": self.llm_pool_id,
            "llm_pool_cycle": list(self.llm_pool_cycle),
            "mock_failure_rate": self.mock_failure_rate,
            "mock_failure_mode": self.mock_failure_mode,
            "force_failure_every": self.force_failure_every,
            "submit_rate": self.submit_rate,
            "submit_concurrency": self.submit_concurrency,
            "batch_size": self.batch_size,
            "submissions": submissions,
        }

    def _start_event_consumer(self) -> None:
        if not self.queue_config:
            return
        self._event_consumer = SchedulerEventConsumer(
            connection_config=self.queue_config,
            api_key=self.api_key,
            callback=self._handle_event_message,
            logger=self._logger,
        )
        self._event_consumer.start()

    def _stop_event_consumer(self) -> None:
        consumer = self._event_consumer
        if consumer is not None:
            consumer.stop()
            self._event_consumer = None

    def _append_debug_snapshot(self, snapshot: DebugSnapshot) -> None:
        with self._state_lock:
            self._debug_samples.append(snapshot)

    async def _capture_debug_snapshot(self, stage: str) -> None:
        if self.http_port is None:
            self._append_debug_snapshot(
                _build_debug_snapshot(
                    stage=stage,
                    error="Gateway HTTP port unavailable for /api/debug sampling",
                )
            )
            return

        owns_session = False
        session = self._http_session
        if session is None:
            session = aiohttp.ClientSession()
            owns_session = True

        try:
            url = f"http://{self.gateway_host}:{self.http_port}/api/debug"
            async with session.get(
                url,
                headers={"Accept": "application/json"},
                timeout=aiohttp.ClientTimeout(total=min(self.timeout, 10.0)),
            ) as response:
                response_text = await response.text()
                if not 200 <= response.status < 300:
                    self._append_debug_snapshot(
                        _build_debug_snapshot(
                            stage=stage,
                            status_code=response.status,
                            error=f"HTTP {response.status}: {response_text[:200]}",
                        )
                    )
                    return

                try:
                    response_json = json.loads(response_text)
                except json.JSONDecodeError as exc:
                    self._append_debug_snapshot(
                        _build_debug_snapshot(
                            stage=stage,
                            status_code=response.status,
                            error=f"Invalid JSON from /api/debug: {exc}",
                        )
                    )
                    return

                payload = _coerce_gateway_debug_payload(response_json)
                self._append_debug_snapshot(
                    _build_debug_snapshot(
                        stage=stage,
                        status_code=response.status,
                        payload=payload if isinstance(payload, dict) else None,
                    )
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._append_debug_snapshot(
                _build_debug_snapshot(stage=stage, error=str(exc))
            )
        finally:
            if owns_session:
                await session.close()

    async def _debug_sampler(self) -> None:
        while self.metrics.end_time is None:
            await asyncio.sleep(self.debug_sample_interval)
            if self.metrics.end_time is not None:
                break
            await self._capture_debug_snapshot("periodic")

    def _register_run(self, run: JobRun) -> None:
        with self._state_lock:
            self._runs_by_request_id[run.request_id] = run
            self._runs_by_ref_id[run.ref_id] = run

    def _mark_submit_result(self, run: JobRun, result: SubmitResult) -> None:
        with self._state_lock:
            run.submit_finished_at = _now()
            run.submit_latency_ms = result.latency_ms
            if result.success:
                run.job_id = result.job_id
                if result.job_id:
                    self._runs_by_job_id[result.job_id] = run
            else:
                run.submit_error_type = result.error_type
                run.submit_error_message = result.error_message

    def _log_submit_summary(self, run: JobRun, result: SubmitResult) -> None:
        if result.success:
            self._logger.info(
                "Submitted job_index=%s request_id=%s job_id=%s planner=%s input_mode=%s source=%s s3_uri=%s",
                run.job_index,
                run.request_id,
                result.job_id,
                run.planner,
                run.input_mode,
                run.source_path,
                run.s3_uri,
            )
            return

        self._logger.warning(
            "Submit failed job_index=%s request_id=%s planner=%s input_mode=%s source=%s s3_uri=%s error_type=%s error=%s",
            run.job_index,
            run.request_id,
            run.planner,
            run.input_mode,
            run.source_path,
            run.s3_uri,
            result.error_type,
            result.error_message,
        )

    def _handle_event_message(self, message: Dict[str, Any]) -> None:
        event_name = message.get("event")
        job_id = message.get("jobid")
        ref_id = _extract_ref_id_from_event(message)
        timestamp = _extract_event_timestamp(message)

        if not isinstance(event_name, str):
            return

        with self._state_lock:
            run = self._runs_by_job_id.get(job_id) if job_id else None
            if run is None and ref_id:
                run = self._runs_by_ref_id.get(ref_id)
                if run is not None and job_id:
                    run.job_id = job_id
                    self._runs_by_job_id[job_id] = run

            if run is None:
                return

            run.raw_events.append(event_name)
            if event_name.endswith(".scheduled") and run.scheduled_at is None:
                run.scheduled_at = timestamp
            elif event_name.endswith(".started") and run.started_at is None:
                run.started_at = timestamp
            elif event_name.endswith(".completed") and run.completed_at is None:
                run.completed_at = timestamp
            elif event_name.endswith(".failed") and run.failed_at is None:
                run.failed_at = timestamp
                run.failure_reason = _extract_failure_error(message)

    def _build_run(self, asset: InputAsset, job_index: int) -> JobRun:
        request_id = f"job-{job_index}-{uuid.uuid4().hex[:10]}"
        ref_id = asset.source_name
        s3_uri = asset.existing_s3_uri or s3_asset_path(
            ref_id=ref_id,
            ref_type=self.ref_type,
            include_filename=True,
        )
        return JobRun(
            request_id=request_id,
            job_index=job_index,
            source_path=asset.source_path,
            source_name=asset.source_name,
            input_mode="upload" if asset.local_path is not None else "existing_s3",
            ref_id=ref_id,
            ref_type=self.ref_type,
            s3_uri=s3_uri,
            planner=self.planner,
            job_name=self.queue_name,
            fault_profile=self.fault_profile,
            llm_pool_id=self._resolve_llm_pool_id(job_index),
        )

    def _resolve_llm_pool_id(self, job_index: int) -> Optional[str]:
        if self.llm_pool_cycle:
            return self.llm_pool_cycle[job_index % len(self.llm_pool_cycle)]
        return self.llm_pool_id

    def _resolve_sla_offsets(
        self, run: JobRun
    ) -> Tuple[Optional[float], Optional[float]]:
        bucket_index = run.job_index // self.sla_step_every_jobs
        if self.sla_step_cycle is not None:
            bucket_index %= self.sla_step_cycle
        run.sla_bucket_index = bucket_index

        soft_offset = self.soft_sla_seconds
        hard_offset = self.hard_sla_seconds
        if soft_offset is not None and self.soft_sla_step_seconds is not None:
            soft_offset += bucket_index * self.soft_sla_step_seconds
        if hard_offset is not None and self.hard_sla_step_seconds is not None:
            hard_offset += bucket_index * self.hard_sla_step_seconds

        run.soft_sla_offset_seconds = soft_offset
        run.hard_sla_offset_seconds = hard_offset
        if soft_offset is not None and hard_offset is not None:
            if soft_offset > hard_offset:
                raise ValueError(
                    f"Resolved soft SLA offset {soft_offset} exceeds hard SLA offset "
                    f"{hard_offset} for job_index={run.job_index}"
                )
        return soft_offset, hard_offset

    def _build_failure_metadata(self, run: JobRun) -> Dict[str, Any]:
        force_fail = (
            self.force_failure_every is not None
            and (run.job_index + 1) % self.force_failure_every == 0
        )
        if self.mock_failure_rate is None and not force_fail:
            run.mock_failure_rate = None
            run.mock_failure_mode = None
            run.force_fail = False
            return {}

        run.mock_failure_rate = self.mock_failure_rate
        run.mock_failure_mode = self.mock_failure_mode
        run.force_fail = force_fail

        failure_metadata: Dict[str, Any] = {
            "failure_mode": self.mock_failure_mode,
            "stress_failure_simulation": {
                "source": "gateway_e2e_stresser",
                "mock_failure_rate": self.mock_failure_rate,
                "mock_failure_mode": self.mock_failure_mode,
                "force_failure_every": self.force_failure_every,
                "force_fail": force_fail,
            },
        }
        if self.mock_failure_rate is not None:
            failure_metadata["failure_rate"] = self.mock_failure_rate
        if force_fail:
            failure_metadata["force_fail"] = True
        return failure_metadata

    def _build_metadata(self, run: JobRun, *, sla_anchor_at: float) -> Dict[str, Any]:
        template_vars = {
            "request_id": run.request_id,
            "job_index": str(run.job_index),
            "timestamp": str(int(sla_anchor_at)),
            "timestamp_ms": str(int(sla_anchor_at * 1000)),
            "uuid": uuid.uuid4().hex,
            "api_key": self.api_key,
            "job_name": self.queue_name,
            "planner": self.planner,
            "ref_id": run.ref_id,
            "ref_type": run.ref_type,
            "s3_uri": run.s3_uri,
            "filename": run.source_name,
            "source_path": run.source_path,
        }

        metadata = _render_template_value(self.metadata_template, template_vars)
        if not isinstance(metadata, dict):
            raise ValueError("Rendered metadata template must be a JSON object")

        if run.llm_pool_id:
            metadata["pool_id"] = run.llm_pool_id

        _inject_purge_annotators_feature(metadata, self.purge_annotators)
        metadata["planner"] = self.planner
        metadata["project_id"] = self.project_id
        metadata["ref_id"] = run.ref_id
        metadata["ref_type"] = run.ref_type
        metadata["policy"] = metadata.get("policy", self.policy)
        metadata["stress_fault_profile"] = self.fault_profile
        metadata["uri"] = run.s3_uri
        metadata.update(self._build_failure_metadata(run))
        soft_offset, hard_offset = self._resolve_sla_offsets(run)
        if soft_offset is not None:
            metadata["soft_sla"] = _format_epoch_seconds(sla_anchor_at + soft_offset)
        if hard_offset is not None:
            metadata["hard_sla"] = _format_epoch_seconds(sla_anchor_at + hard_offset)

        run.soft_sla_at = _parse_optional_epoch_seconds(metadata.get("soft_sla"))
        run.hard_sla_at = _parse_optional_epoch_seconds(metadata.get("hard_sla"))
        if run.soft_sla_at is not None and run.hard_sla_at is not None:
            if run.soft_sla_at > run.hard_sla_at:
                raise ValueError("soft_sla must be less than or equal to hard_sla")
        return metadata

    def _upload_to_s3(self, run: JobRun) -> None:
        if run.input_mode != "upload":
            run.upload_started_at = _now()
            run.upload_finished_at = run.upload_started_at
            return

        run.upload_started_at = _now()
        source_path = Path(run.source_path)
        s3_root = s3_asset_path(ref_id=run.ref_id, ref_type=run.ref_type)

        status = StorageManager.write(str(source_path), run.s3_uri, overwrite=True)
        self._logger.debug("Uploaded %s to %s: %s", source_path, run.s3_uri, status)

        if self.upload_companion_meta:
            companion_meta_path = source_path.with_name(f"{source_path.name}.meta.json")
            if companion_meta_path.exists():
                try:
                    upload_meta = json.loads(companion_meta_path.read_text())
                    if isinstance(upload_meta, dict):
                        upload_meta["ref_id"] = run.ref_id
                        upload_meta["ref_type"] = run.ref_type
                        upload_meta["planner"] = self.planner
                        upload_meta["project_id"] = self.project_id
                        upload_meta["uri"] = run.s3_uri
                        meta_s3_path = f"{s3_root}/{run.ref_id}.meta.json"
                        tmp_path = (
                            source_path.parent / f".{run.ref_id}.stress.meta.json"
                        )
                        tmp_path.write_text(json.dumps(upload_meta, indent=2))
                        try:
                            StorageManager.write(
                                str(tmp_path), meta_s3_path, overwrite=True
                            )
                        finally:
                            tmp_path.unlink(missing_ok=True)
                except Exception as exc:
                    self._logger.warning(
                        "Failed to upload companion metadata for %s: %s",
                        source_path,
                        exc,
                    )

        run.upload_finished_at = _now()

    async def _submit_http(self, run: JobRun, metadata: Dict[str, Any]) -> SubmitResult:
        start_time = _now()
        payload = self._build_submit_request(run, metadata)
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        url = f"http://{self.gateway_host}:{self.http_port}{self.endpoint}"

        try:
            assert self._http_session is not None
            async with self._http_session.post(
                url,
                json=payload,
                headers=headers,
                timeout=timeout,
            ) as response:
                response_text = await response.text()
                latency_ms = (_now() - start_time) * 1000.0
                try:
                    response_json = json.loads(response_text)
                except json.JSONDecodeError:
                    response_json = None

                if not 200 <= response.status < 300:
                    return SubmitResult(
                        request_id=run.request_id,
                        success=False,
                        latency_ms=latency_ms,
                        error_type=f"HTTP_{response.status}",
                        error_message=response_text[:200],
                    )

                parameters = (
                    response_json.get("parameters")
                    if isinstance(response_json, dict)
                    else None
                )
                if not isinstance(parameters, dict):
                    return SubmitResult(
                        request_id=run.request_id,
                        success=False,
                        latency_ms=latency_ms,
                        error_type="BAD_RESPONSE",
                        error_message=response_text[:200],
                    )

                if parameters.get("status") != "ok":
                    return SubmitResult(
                        request_id=run.request_id,
                        success=False,
                        latency_ms=latency_ms,
                        error_type="APP_ERROR",
                        error_message=str(parameters.get("msg") or parameters)[:200],
                    )

                job_id = parameters.get("job_id")
                if not isinstance(job_id, str) or not job_id:
                    return SubmitResult(
                        request_id=run.request_id,
                        success=False,
                        latency_ms=latency_ms,
                        error_type="APP_ERROR",
                        error_message="Submit response missing job_id",
                    )

                return SubmitResult(
                    request_id=run.request_id,
                    success=True,
                    latency_ms=latency_ms,
                    job_id=job_id,
                )
        except asyncio.TimeoutError:
            return SubmitResult(
                request_id=run.request_id,
                success=False,
                latency_ms=(_now() - start_time) * 1000.0,
                error_type="TIMEOUT",
                error_message="HTTP request timed out",
            )
        except aiohttp.ClientError as exc:
            return SubmitResult(
                request_id=run.request_id,
                success=False,
                latency_ms=(_now() - start_time) * 1000.0,
                error_type=type(exc).__name__,
                error_message=str(exc)[:200],
            )
        except Exception as exc:
            return SubmitResult(
                request_id=run.request_id,
                success=False,
                latency_ms=(_now() - start_time) * 1000.0,
                error_type=type(exc).__name__,
                error_message=str(exc)[:200],
            )

    async def _submit_grpc(self, run: JobRun, metadata: Dict[str, Any]) -> SubmitResult:
        start_time = _now()
        try:
            client = Client(
                host=self.gateway_host,
                port=self.gateway_port,
                protocol=self.protocol,
                request_size=-1,
                asyncio=True,
            )
            request_payload = self._build_submit_request(run, metadata)
            docs = DocumentArray(
                [Document(text=item["text"]) for item in request_payload["data"]]
            )
            response_docs = None
            async for response in client.post(
                self.endpoint,
                inputs=docs,
                parameters=request_payload["parameters"],
                request_size=self.batch_size,
                timeout=self.timeout,
            ):
                response_docs = response

            latency_ms = (_now() - start_time) * 1000.0
            parameters = (
                dict(getattr(response_docs, "parameters", {}) or {})
                if response_docs is not None
                else {}
            )
            if parameters.get("status") != "ok":
                return SubmitResult(
                    request_id=run.request_id,
                    success=False,
                    latency_ms=latency_ms,
                    error_type="APP_ERROR",
                    error_message=str(parameters.get("msg") or parameters)[:200],
                )

            job_id = parameters.get("job_id")
            if not isinstance(job_id, str) or not job_id:
                return SubmitResult(
                    request_id=run.request_id,
                    success=False,
                    latency_ms=latency_ms,
                    error_type="APP_ERROR",
                    error_message="Submit response missing job_id",
                )

            return SubmitResult(
                request_id=run.request_id,
                success=True,
                latency_ms=latency_ms,
                job_id=job_id,
            )
        except asyncio.TimeoutError:
            return SubmitResult(
                request_id=run.request_id,
                success=False,
                latency_ms=(_now() - start_time) * 1000.0,
                error_type="TIMEOUT",
                error_message="gRPC request timed out",
            )
        except grpc.RpcError as exc:
            error_code = exc.code().name if hasattr(exc, "code") else "UNKNOWN"
            error_details = exc.details() if hasattr(exc, "details") else str(exc)
            return SubmitResult(
                request_id=run.request_id,
                success=False,
                latency_ms=(_now() - start_time) * 1000.0,
                error_type=f"GRPC_{error_code}",
                error_message=str(error_details)[:200],
            )
        except Exception as exc:
            return SubmitResult(
                request_id=run.request_id,
                success=False,
                latency_ms=(_now() - start_time) * 1000.0,
                error_type=type(exc).__name__,
                error_message=str(exc)[:200],
            )

    async def _submit_run(self, run: JobRun, semaphore: asyncio.Semaphore) -> None:
        async with semaphore:
            await asyncio.to_thread(self._upload_to_s3, run)
            run.submit_started_at = _now()
            metadata = self._build_metadata(run, sla_anchor_at=run.submit_started_at)
            if self.protocol == "http":
                result = await self._submit_http(run, metadata)
            else:
                result = await self._submit_grpc(run, metadata)
            self._mark_submit_result(run, result)
            self._log_submit_summary(run, result)

    async def _apply_aimock_fault_profile(self) -> None:
        if not self.aimock_admin_url:
            return

        owns_session = False
        if self._http_session is None:
            connector = aiohttp.TCPConnector(
                limit=4, limit_per_host=4, enable_cleanup_closed=True
            )
            self._http_session = aiohttp.ClientSession(connector=connector)
            owns_session = True

        try:
            assert self._http_session is not None
            async with self._http_session.post(
                f"{self.aimock_admin_url}/fault-profile",
                json={"profile": self.fault_profile},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                response_text = await response.text()
                if not 200 <= response.status < 300:
                    raise RuntimeError(
                        f"AIMock admin returned HTTP {response.status}: {response_text[:200]}"
                    )
                self._logger.info(
                    "Configured AIMock fault profile=%s via %s",
                    self.fault_profile,
                    self.aimock_admin_url,
                )
        finally:
            if owns_session and self._http_session is not None:
                await self._http_session.close()
                self._http_session = None

    async def _progress_reporter(self) -> None:
        while self.metrics.end_time is None:
            await asyncio.sleep(self.progress_interval)
            payload = self._build_live_status_payload("running")
            self._log_progress_payload(payload)
            if self.live_report_path:
                await asyncio.to_thread(self._write_live_report_payload, payload)

    async def _wait_for_terminal_states(self) -> None:
        deadline = _now() + self.terminal_timeout
        while _now() < deadline:
            with self._state_lock:
                if all(
                    run.terminal_status is not None
                    for run in self._runs_by_request_id.values()
                ):
                    return
            await asyncio.sleep(0.25)

        with self._state_lock:
            for run in self._runs_by_request_id.values():
                if run.terminal_status is None:
                    run.failed_at = run.failed_at or _now()
                    if run.failure_reason is None:
                        run.failure_reason = "terminal event timeout"

    def _finalize_metrics(self) -> None:
        with self._state_lock:
            runs = list(self._runs_by_request_id.values())

        metrics = E2EMetrics(total_jobs=len(runs))
        metrics.start_time = self.metrics.start_time
        metrics.end_time = self.metrics.end_time

        for run in runs:
            if run.job_id:
                metrics.submitted_jobs += 1
            if run.submit_latency_ms is not None:
                metrics.submit_latencies_ms.append(run.submit_latency_ms)
            if run.scheduling_ms is not None:
                metrics.scheduling_latencies_ms.append(run.scheduling_ms)
            if run.queue_wait_ms is not None:
                metrics.queue_wait_ms.append(run.queue_wait_ms)
            if run.execution_ms is not None:
                metrics.execution_latencies_ms.append(run.execution_ms)
            if run.end_to_end_ms is not None:
                metrics.end_to_end_latencies_ms.append(run.end_to_end_ms)

            status = run.terminal_status
            if status == "completed":
                metrics.completed_jobs += 1
            elif status == "failed":
                if run.failure_reason == "terminal event timeout":
                    metrics.event_timeout_jobs += 1
                else:
                    metrics.failed_jobs += 1
                reason = run.failure_reason or "unknown failure"
                metrics.failure_reasons[reason] = (
                    metrics.failure_reasons.get(reason, 0) + 1
                )
            elif status == "submit_failed":
                metrics.submit_failed_jobs += 1
                reason = run.submit_error_type or "submit failed"
                metrics.failure_reasons[reason] = (
                    metrics.failure_reasons.get(reason, 0) + 1
                )

            if run.job_id and run.soft_sla_at is not None:
                metrics.soft_sla_configured_jobs += 1
                if run.soft_sla_met is True:
                    metrics.soft_sla_met_jobs += 1
                else:
                    metrics.soft_sla_missed_jobs += 1
                lateness_ms = run.soft_sla_lateness_ms
                if lateness_ms is not None and lateness_ms > 0:
                    metrics.soft_sla_lateness_ms.append(lateness_ms)

            if run.job_id and run.hard_sla_at is not None:
                metrics.hard_sla_configured_jobs += 1
                if run.hard_sla_met is True:
                    metrics.hard_sla_met_jobs += 1
                else:
                    metrics.hard_sla_missed_jobs += 1
                lateness_ms = run.hard_sla_lateness_ms
                if lateness_ms is not None and lateness_ms > 0:
                    metrics.hard_sla_lateness_ms.append(lateness_ms)

        self.metrics = metrics
        self._verification_errors = self._evaluate_sla_verification_errors()

    def _evaluate_sla_verification_errors(self) -> List[str]:
        errors: List[str] = []
        if self.min_soft_sla_compliance_pct is not None:
            if self.metrics.soft_sla_configured_jobs <= 0:
                errors.append(
                    "Soft SLA verification requested but no submitted jobs carried a soft_sla"
                )
            elif (
                self.metrics.soft_sla_compliance_pct < self.min_soft_sla_compliance_pct
            ):
                errors.append(
                    "Soft SLA compliance "
                    f"{self.metrics.soft_sla_compliance_pct:.2f}% is below the required "
                    f"{self.min_soft_sla_compliance_pct:.2f}%"
                )
        if self.min_hard_sla_compliance_pct is not None:
            if self.metrics.hard_sla_configured_jobs <= 0:
                errors.append(
                    "Hard SLA verification requested but no submitted jobs carried a hard_sla"
                )
            elif (
                self.metrics.hard_sla_compliance_pct < self.min_hard_sla_compliance_pct
            ):
                errors.append(
                    "Hard SLA compliance "
                    f"{self.metrics.hard_sla_compliance_pct:.2f}% is below the required "
                    f"{self.min_hard_sla_compliance_pct:.2f}%"
                )
        return errors

    async def run(self) -> E2EMetrics:
        self._setup_storage()
        self._start_event_consumer()
        await self._apply_aimock_fault_profile()

        connector = None
        if self.protocol == "http" or self.debug_sample_interval > 0:
            connector = aiohttp.TCPConnector(
                limit=self.submit_concurrency * 2,
                limit_per_host=self.submit_concurrency * 2,
                enable_cleanup_closed=True,
            )
            self._http_session = aiohttp.ClientSession(connector=connector)

        semaphore = asyncio.Semaphore(self.submit_concurrency)
        reporter = asyncio.create_task(self._progress_reporter())
        debug_sampler = None
        self.metrics.start_time = _now()
        if self.live_report_path:
            await asyncio.to_thread(
                self._write_live_report_payload,
                self._build_live_status_payload("running"),
            )

        try:
            if self.debug_sample_interval > 0:
                await self._capture_debug_snapshot("start")
                debug_sampler = asyncio.create_task(self._debug_sampler())

            tasks = []
            interval = 1.0 / self.submit_rate
            next_submit_at = time.monotonic()
            deadline = (
                next_submit_at + self.run_time_seconds
                if self.run_time_seconds is not None
                else None
            )
            job_index = 0
            while True:
                if self.job_count is not None and job_index >= self.job_count:
                    break
                now_monotonic = time.monotonic()
                if deadline is not None and now_monotonic >= deadline:
                    break
                asset = self.input_assets[job_index % len(self.input_assets)]
                run = self._build_run(asset, job_index)
                self._register_run(run)
                tasks.append(asyncio.create_task(self._submit_run(run, semaphore)))
                job_index += 1
                next_submit_at += interval
                sleep_for = next_submit_at - time.monotonic()
                if sleep_for > 0:
                    await asyncio.sleep(sleep_for)

            await asyncio.gather(*tasks)
            await self._wait_for_terminal_states()
        finally:
            self.metrics.end_time = _now()
            reporter.cancel()
            await asyncio.gather(reporter, return_exceptions=True)
            if debug_sampler is not None:
                debug_sampler.cancel()
                await asyncio.gather(debug_sampler, return_exceptions=True)
            if self.debug_sample_interval > 0:
                await self._capture_debug_snapshot("end")
            if self._http_session is not None:
                await self._http_session.close()
                self._http_session = None
            self._stop_event_consumer()

        self._finalize_metrics()
        if self.live_report_path:
            await asyncio.to_thread(
                self._write_live_report_payload,
                self._build_live_status_payload("completed"),
            )
        return self.metrics

    def print_report(self) -> None:
        m = self.metrics
        duration = (
            (m.end_time - m.start_time)
            if m.start_time is not None and m.end_time is not None
            else 0.0
        )
        print("\n" + "=" * 78)
        print("GATEWAY END-TO-END STRESS REPORT")
        print("=" * 78)
        print(f"Duration: {duration:.2f}s")
        print(f"Run mode: {self.run_mode}")
        if self.job_count is not None:
            print(f"Jobs requested: {self.job_count}")
        else:
            print(f"Run time target: {self.run_time_seconds:.2f}s")
            if self.estimated_job_count is not None:
                print(f"Estimated jobs at submit rate: {self.estimated_job_count}")
            print(f"Jobs created: {m.total_jobs}")
        print(f"Fault profile: {self.fault_profile}")
        if self.mock_failure_rate is not None or self.force_failure_every is not None:
            print(
                "Mock failure simulation: "
                f"rate={self.mock_failure_rate if self.mock_failure_rate is not None else 'default'} "
                f"mode={self.mock_failure_mode} "
                f"force_every={self.force_failure_every if self.force_failure_every is not None else 'none'}"
            )
        if self.llm_pool_id:
            print(f"LLM dispatch pool: {self.llm_pool_id}")
        if self.llm_pool_cycle:
            print(f"LLM dispatch pool cycle: {', '.join(self.llm_pool_cycle)}")
        print(f"Jobs submitted: {m.submitted_jobs}")
        print(f"Completed: {m.completed_jobs}")
        print(f"Failed: {m.failed_jobs}")
        print(f"Submit failed: {m.submit_failed_jobs}")
        print(f"Timed out waiting for events: {m.event_timeout_jobs}")
        print(f"Throughput: {m.throughput:.2f} jobs/s")
        if self.soft_sla_seconds is not None:
            print(f"Soft SLA target: +{self.soft_sla_seconds:.2f}s from submit start")
        if self.hard_sla_seconds is not None:
            print(f"Hard SLA target: +{self.hard_sla_seconds:.2f}s from submit start")
        if self.soft_sla_step_seconds is not None:
            print(
                "Soft SLA increment: "
                f"{self.soft_sla_step_seconds:+.2f}s every {self.sla_step_every_jobs} jobs"
            )
        if self.hard_sla_step_seconds is not None:
            print(
                "Hard SLA increment: "
                f"{self.hard_sla_step_seconds:+.2f}s every {self.sla_step_every_jobs} jobs"
            )
        if self.sla_step_cycle is not None:
            print(f"SLA step cycle: {self.sla_step_cycle} buckets")

        def print_latency_block(title: str, values: List[float]) -> None:
            if not values:
                return
            print(f"\n--- {title} (ms) ---")
            print(f"Count: {len(values)}")
            print(f"Min: {min(values):.2f}")
            print(f"Max: {max(values):.2f}")
            print(f"Avg: {statistics.mean(values):.2f}")
            print(f"P50: {_percentile(values, 50):.2f}")
            print(f"P95: {_percentile(values, 95):.2f}")
            print(f"P99: {_percentile(values, 99):.2f}")

        print_latency_block("Submit Latency", m.submit_latencies_ms)
        print_latency_block("Scheduling Latency", m.scheduling_latencies_ms)
        print_latency_block("Queue Wait", m.queue_wait_ms)
        print_latency_block("Execution Latency", m.execution_latencies_ms)
        print_latency_block("End-to-End Latency", m.end_to_end_latencies_ms)
        print_latency_block("Soft SLA Lateness", m.soft_sla_lateness_ms)
        print_latency_block("Hard SLA Lateness", m.hard_sla_lateness_ms)

        if m.soft_sla_configured_jobs or m.hard_sla_configured_jobs:
            print("\n--- SLA Compliance ---")
            if m.soft_sla_configured_jobs:
                print(
                    f"Soft SLA: {m.soft_sla_met_jobs}/{m.soft_sla_configured_jobs} "
                    f"met ({m.soft_sla_compliance_pct:.2f}%)"
                )
            if m.hard_sla_configured_jobs:
                print(
                    f"Hard SLA: {m.hard_sla_met_jobs}/{m.hard_sla_configured_jobs} "
                    f"met ({m.hard_sla_compliance_pct:.2f}%)"
                )

        if m.failure_reasons:
            print("\n--- Failure Reasons ---")
            for reason, count in sorted(
                m.failure_reasons.items(), key=lambda item: (-item[1], item[0])
            ):
                print(f"{count:5d}  {reason}")

        with self._state_lock:
            debug_samples = list(self._debug_samples)
        if debug_samples:
            ok_count = sum(1 for sample in debug_samples if sample.ok)
            error_count = len(debug_samples) - ok_count
            last_ok = next(
                (sample for sample in reversed(debug_samples) if sample.ok), None
            )

            print("\n--- Gateway Debug Sampling ---")
            print(f"Samples: {len(debug_samples)}")
            print(f"Successful samples: {ok_count}")
            print(f"Failed samples: {error_count}")
            if last_ok is not None:
                print(f"Last active DAG count: {last_ok.active_dags_count}")
                print(f"Last request queue size: {last_ok.request_queue_size}")
                print(f"Last fetch counter: {last_ok.fetch_counter}")
                print(
                    "Last LLM dispatchers: "
                    f"{last_ok.llm_dispatch_running_dispatchers}/"
                    f"{last_ok.llm_dispatch_registered_dispatchers}"
                )

        if (
            self.min_soft_sla_compliance_pct is not None
            or self.min_hard_sla_compliance_pct is not None
        ):
            print("\n--- SLA Verification ---")
            if self._verification_errors:
                print("FAILED")
                for error in self._verification_errors:
                    print(f"- {error}")
            else:
                print("PASSED")

        print("=" * 78 + "\n")

    def build_report_payload(self) -> Dict[str, Any]:
        with self._state_lock:
            runs = list(self._runs_by_request_id.values())
            debug_samples = list(self._debug_samples)
        report_now = self.metrics.end_time or _now()

        debug_sample_payload = [
            {
                "stage": sample.stage,
                "captured_at": sample.captured_at,
                "ok": sample.ok,
                "status_code": sample.status_code,
                "error": sample.error,
                "scheduler_running": sample.scheduler_running,
                "scheduler_paused": sample.scheduler_paused,
                "active_dags_count": sample.active_dags_count,
                "max_concurrent_dags": sample.max_concurrent_dags,
                "fetch_counter": sample.fetch_counter,
                "pending_requests": sample.pending_requests,
                "request_queue_size": sample.request_queue_size,
                "event_queue_size": sample.event_queue_size,
                "queue_status": sample.queue_status,
                "queue_status_error": sample.queue_status_error,
                "llm_dispatch": sample.llm_dispatch,
                "llm_dispatch_registered_dispatchers": sample.llm_dispatch_registered_dispatchers,
                "llm_dispatch_running_dispatchers": sample.llm_dispatch_running_dispatchers,
            }
            for sample in debug_samples
        ]

        duration = (
            (self.metrics.end_time - self.metrics.start_time)
            if self.metrics.start_time is not None and self.metrics.end_time is not None
            else None
        )

        return {
            "summary": {
                "report_generated_at": _format_epoch_seconds(_now()),
                "start_time": (
                    _format_epoch_seconds(self.metrics.start_time)
                    if self.metrics.start_time is not None
                    else None
                ),
                "end_time": (
                    _format_epoch_seconds(self.metrics.end_time)
                    if self.metrics.end_time is not None
                    else None
                ),
                "duration_seconds": duration,
                "run_mode": self.run_mode,
                "configured_job_count": self.job_count,
                "configured_run_time_seconds": self.run_time_seconds,
                "estimated_job_count": self.estimated_job_count,
                "progress_interval_seconds": self.progress_interval,
                "live_report_path": self.live_report_path,
                "live_report_format": (
                    resolve_report_format(
                        self.live_report_path, self.live_report_format
                    )
                    if self.live_report_path
                    else None
                ),
                "total_jobs": self.metrics.total_jobs,
                "fault_profile": self.fault_profile,
                "mock_failure_rate": self.mock_failure_rate,
                "mock_failure_mode": self.mock_failure_mode,
                "force_failure_every": self.force_failure_every,
                "submitted_jobs": self.metrics.submitted_jobs,
                "completed_jobs": self.metrics.completed_jobs,
                "failed_jobs": self.metrics.failed_jobs,
                "submit_failed_jobs": self.metrics.submit_failed_jobs,
                "event_timeout_jobs": self.metrics.event_timeout_jobs,
                "throughput": self.metrics.throughput,
                "soft_sla_configured_jobs": self.metrics.soft_sla_configured_jobs,
                "soft_sla_met_jobs": self.metrics.soft_sla_met_jobs,
                "soft_sla_missed_jobs": self.metrics.soft_sla_missed_jobs,
                "soft_sla_compliance_pct": self.metrics.soft_sla_compliance_pct,
                "hard_sla_configured_jobs": self.metrics.hard_sla_configured_jobs,
                "hard_sla_met_jobs": self.metrics.hard_sla_met_jobs,
                "hard_sla_missed_jobs": self.metrics.hard_sla_missed_jobs,
                "hard_sla_compliance_pct": self.metrics.hard_sla_compliance_pct,
                "sla_step_every_jobs": self.sla_step_every_jobs,
                "sla_step_cycle": self.sla_step_cycle,
                "debug_sample_count": len(debug_sample_payload),
            },
            "latencies_ms": {
                "submit": self.metrics.submit_latencies_ms,
                "scheduling": self.metrics.scheduling_latencies_ms,
                "queue_wait": self.metrics.queue_wait_ms,
                "execution": self.metrics.execution_latencies_ms,
                "end_to_end": self.metrics.end_to_end_latencies_ms,
                "soft_sla_lateness": self.metrics.soft_sla_lateness_ms,
                "hard_sla_lateness": self.metrics.hard_sla_lateness_ms,
            },
            "latency_stats_ms": {
                "submit": build_latency_stats(
                    self.metrics.submit_latencies_ms, _percentile
                ),
                "scheduling": build_latency_stats(
                    self.metrics.scheduling_latencies_ms, _percentile
                ),
                "queue_wait": build_latency_stats(
                    self.metrics.queue_wait_ms, _percentile
                ),
                "execution": build_latency_stats(
                    self.metrics.execution_latencies_ms, _percentile
                ),
                "end_to_end": build_latency_stats(
                    self.metrics.end_to_end_latencies_ms, _percentile
                ),
                "soft_sla_lateness": build_latency_stats(
                    self.metrics.soft_sla_lateness_ms, _percentile
                ),
                "hard_sla_lateness": build_latency_stats(
                    self.metrics.hard_sla_lateness_ms, _percentile
                ),
            },
            "failure_reasons": self.metrics.failure_reasons,
            "sla_verification": {
                "soft_sla_seconds": self.soft_sla_seconds,
                "hard_sla_seconds": self.hard_sla_seconds,
                "soft_sla_step_seconds": self.soft_sla_step_seconds,
                "hard_sla_step_seconds": self.hard_sla_step_seconds,
                "sla_step_every_jobs": self.sla_step_every_jobs,
                "sla_step_cycle": self.sla_step_cycle,
                "min_soft_sla_compliance_pct": self.min_soft_sla_compliance_pct,
                "min_hard_sla_compliance_pct": self.min_hard_sla_compliance_pct,
                "passed": not self._verification_errors,
                "errors": self.verification_errors,
            },
            "sla": self._build_sla_payload(runs, now=report_now),
            "debug_sampling": {
                "enabled": self.debug_sample_interval > 0,
                "sample_interval_seconds": self.debug_sample_interval,
                "samples": debug_sample_payload,
            },
            "jobs": [
                {
                    "request_id": run.request_id,
                    "job_index": run.job_index,
                    "job_id": run.job_id,
                    "source_path": run.source_path,
                    "input_mode": run.input_mode,
                    "s3_uri": run.s3_uri,
                    "planner": run.planner,
                    "job_name": run.job_name,
                    "fault_profile": run.fault_profile,
                    "mock_failure_rate": run.mock_failure_rate,
                    "mock_failure_mode": run.mock_failure_mode,
                    "force_fail": run.force_fail,
                    "terminal_status": run.terminal_status,
                    "sla_bucket_index": run.sla_bucket_index,
                    "soft_sla_offset_seconds": run.soft_sla_offset_seconds,
                    "hard_sla_offset_seconds": run.hard_sla_offset_seconds,
                    "soft_sla": (
                        _format_epoch_seconds(run.soft_sla_at)
                        if run.soft_sla_at is not None
                        else None
                    ),
                    "hard_sla": (
                        _format_epoch_seconds(run.hard_sla_at)
                        if run.hard_sla_at is not None
                        else None
                    ),
                    "soft_sla_status": run.soft_sla_status,
                    "hard_sla_status": run.hard_sla_status,
                    "soft_sla_met": run.soft_sla_met,
                    "hard_sla_met": run.hard_sla_met,
                    "soft_sla_lateness_ms": run.soft_sla_lateness_ms,
                    "hard_sla_lateness_ms": run.hard_sla_lateness_ms,
                    "submit_latency_ms": run.submit_latency_ms,
                    "queue_wait_ms": run.queue_wait_ms,
                    "scheduling_ms": run.scheduling_ms,
                    "execution_ms": run.execution_ms,
                    "end_to_end_ms": run.end_to_end_ms,
                    "failure_reason": run.failure_reason,
                    "submit_error_type": run.submit_error_type,
                    "submit_error_message": run.submit_error_message,
                    "raw_events": run.raw_events,
                }
                for run in runs
            ],
        }

    def write_report(self, output_path: str, report_format: str = "auto") -> None:
        payload = self.build_report_payload()
        resolved_format = resolve_report_format(output_path, report_format)
        rendered = render_final_report(payload, resolved_format)
        write_text_atomically(output_path, rendered)

    def write_json_report(self, output_path: str) -> None:
        self.write_report(output_path, "json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gateway end-to-end stress tester for scheduler + LLM pipelines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/stress/gateway_e2e_stresser.py \\
      --config tools/stress/gateway-e2e.config.example.json \\
      --s3-uri-manifest tools/stress/gateway-e2e.s3-uri-manifest.example.txt \\
      --job-count 50 \\
      --job-name gen5_extract \\
      --planner extract \\
      --soft-sla-seconds 30 \\
      --hard-sla-seconds 90 \\
      --soft-sla-step-seconds 10 \\
      --hard-sla-step-seconds 20 \\
      --sla-step-every-jobs 25 \\
      --sla-step-cycle 4 \\
      --min-hard-sla-compliance-pct 99 \\
      --submit-rate 4 \\
      --report /tmp/gateway-e2e-report.html

  python tools/stress/gateway_e2e_stresser.py \\
      --config tools/stress/gateway-e2e.config.example.json \\
      --s3-uri-manifest tools/stress/gateway-e2e.s3-uri-manifest.example.txt \\
      --run-time 1h \\
      --job-name gen5_extract \\
      --planner extract \\
      --submit-rate 10 \\
      --live-report /tmp/gateway-e2e-live.html

  python tools/stress/gateway_e2e_stresser.py \\
      --config tools/stress/gateway-e2e.config.example.json \\
      --s3-uri s3://marie/gen5_extract/sample-001.tif \\
      --job-count 10 \\
      --job-name gen5_extract \\
      --planner extract \\
      --fault-profile chaos \\
      --aimock-admin-url http://localhost:4011

  python tools/stress/gateway_e2e_stresser.py \\
      --config tools/stress/gateway-e2e.config.example.json \\
      --s3-uri-manifest tools/stress/gateway-e2e.s3-uri-manifest.example.txt \\
      --job-count 50 \\
      --job-name mock_parallel_subgraphs \\
      --planner mock_parallel_subgraphs \\
      --mock-failure-rate 0.10 \\
      --mock-failure-mode exception \\
      --force-failure-every 5

  python tools/stress/gateway_e2e_stresser.py \\
      --config tools/stress/gateway-e2e.config.example.json \\
      --input-dir /mnt/data/marie-ai/generators \\
      --job-count 25 \\
      --job-name gen5_extract \\
      --planner extract
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to gateway E2E stress config JSON containing api_base_url, api_key, storage, and queue",
    )
    parser.add_argument("--gateway-host", default=None, help="Gateway host override")
    parser.add_argument(
        "--gateway-port", type=int, default=None, help="Gateway gRPC port override"
    )
    parser.add_argument(
        "--http-port", type=int, default=None, help="Gateway HTTP port override"
    )
    parser.add_argument(
        "--protocol",
        choices=["grpc", "http"],
        default=None,
        help="Gateway protocol override",
    )
    parser.add_argument(
        "--endpoint",
        default="/api/v1/invoke",
        help="Gateway endpoint (default: /api/v1/invoke)",
    )
    parser.add_argument("--api-key", default=None, help="Gateway API key override")

    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument(
        "--input-dir", type=str, help="Directory of source files to cycle through"
    )
    input_group.add_argument(
        "--input-glob", type=str, help="Glob pattern of source files"
    )
    input_group.add_argument(
        "--input-manifest", type=str, help="Text file with one source path per line"
    )
    input_group.add_argument(
        "--s3-uri", type=str, help="Existing s3:// URI to submit without uploading"
    )
    input_group.add_argument(
        "--s3-uri-manifest",
        type=str,
        help="Text file with one existing s3:// URI per line",
    )

    workload_group = parser.add_mutually_exclusive_group(required=True)
    workload_group.add_argument(
        "--job-count",
        type=int,
        help="Total jobs to submit",
    )
    workload_group.add_argument(
        "--run-time",
        "--duration",
        dest="run_time",
        type=str,
        help="Run duration for rate-controlled submission, for example 30s, 2m, or 1h",
    )
    parser.add_argument(
        "--job-name",
        type=str,
        required=False,
        default=None,
        help="Scheduler submit name / queue name",
    )
    parser.add_argument(
        "--planner", type=str, required=True, help="Planner to place in metadata"
    )
    parser.add_argument(
        "--fault-profile",
        choices=["normal", "timeout", "error", "chaos"],
        default="normal",
        help="Logical fault profile label for this run; combine with AIMock config/admin control when using mock backends",
    )
    parser.add_argument(
        "--mock-failure-rate",
        type=float,
        default=None,
        help="Per-request mock executor failure_rate override to place in job metadata (0.0 to 1.0)",
    )
    parser.add_argument(
        "--mock-failure-mode",
        choices=MOCK_FAILURE_MODES,
        default="exception",
        help="Per-request mock executor failure_mode override used with --mock-failure-rate or --force-failure-every",
    )
    parser.add_argument(
        "--force-failure-every",
        type=int,
        default=None,
        help="Force every Nth generated job to fail by setting force_fail=true in job metadata",
    )
    parser.add_argument(
        "--aimock-admin-url",
        type=str,
        default=None,
        help="Optional AIMock admin base URL used to set the active fault profile before the run, for example http://localhost:4011",
    )
    parser.add_argument(
        "--soft-sla-seconds",
        type=float,
        default=None,
        help="Optional soft SLA offset in seconds from submit start",
    )
    parser.add_argument(
        "--hard-sla-seconds",
        type=float,
        default=None,
        help="Optional hard SLA offset in seconds from submit start",
    )
    parser.add_argument(
        "--soft-sla-step-seconds",
        type=float,
        default=None,
        help="Optional soft SLA increment applied per SLA bucket; can be negative",
    )
    parser.add_argument(
        "--hard-sla-step-seconds",
        type=float,
        default=None,
        help="Optional hard SLA increment applied per SLA bucket; can be negative",
    )
    parser.add_argument(
        "--sla-step-every-jobs",
        type=int,
        default=1,
        help="How many jobs share the same SLA bucket before the step increment advances",
    )
    parser.add_argument(
        "--sla-step-cycle",
        type=int,
        default=None,
        help="Optional number of SLA buckets before the incremental deadline pattern wraps",
    )
    parser.add_argument(
        "--min-soft-sla-compliance-pct",
        type=float,
        default=None,
        help="Optional verification threshold for soft SLA compliance percentage",
    )
    parser.add_argument(
        "--min-hard-sla-compliance-pct",
        type=float,
        default=None,
        help="Optional verification threshold for hard SLA compliance percentage",
    )
    parser.add_argument(
        "--ref-type",
        type=str,
        default=None,
        help="ref_type override (default: job-name)",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="allow_all",
        help="Submission policy metadata value",
    )
    parser.add_argument(
        "--project-id",
        type=str,
        default=None,
        help="project_id metadata override (default: api_key)",
    )
    llm_pool_group = parser.add_mutually_exclusive_group(required=False)
    llm_pool_group.add_argument(
        "--llm-pool-id",
        type=str,
        default=None,
        help="Fixed LLM dispatch pool ID to place in metadata.pool_id, for example document-small",
    )
    llm_pool_group.add_argument(
        "--llm-pool-cycle",
        type=str,
        default=None,
        help="Comma-separated LLM dispatch pool IDs to cycle through metadata.pool_id by generated job index",
    )
    parser.add_argument(
        "--purge-annotators",
        type=str,
        default=None,
        help="Comma-separated annotator names to purge before annotation, for example mock-llm",
    )

    parser.add_argument(
        "--submit-concurrency",
        type=int,
        default=8,
        help="Concurrent upload+submit workers",
    )
    parser.add_argument(
        "--submit-rate", type=float, default=2.0, help="Target submit rate in jobs/sec"
    )
    parser.add_argument(
        "--timeout", type=float, default=30.0, help="Submit request timeout in seconds"
    )
    parser.add_argument(
        "--terminal-timeout",
        type=float,
        default=900.0,
        help="How long to wait for completion/failed events after submissions finish",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Dummy documents per submit request"
    )
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=5.0,
        help="Console progress and live report refresh interval in seconds",
    )
    parser.add_argument(
        "--live-report",
        type=str,
        default=None,
        help="Optional path to a live status report rewritten during the run",
    )
    parser.add_argument(
        "--live-report-format",
        choices=REPORT_FORMAT_CHOICES,
        default="auto",
        help="Live report format override; default auto infers from the file extension",
    )
    parser.add_argument(
        "--live-json",
        dest="live_json_compat",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--dry-run-preview-count",
        type=int,
        default=3,
        help="When using --dry-run with --run-time, preview this many would-be submissions",
    )
    parser.add_argument(
        "--debug-sample-interval",
        type=float,
        default=0.0,
        help="Optional /api/debug sampling interval in seconds (0 disables sampling)",
    )
    parser.add_argument(
        "--request-template",
        type=str,
        default=None,
        help="JSON file containing either metadata or a full invoke_action template",
    )
    parser.add_argument(
        "--skip-companion-meta-upload",
        action="store_true",
        help="Do not upload <file>.meta.json sidecars when present",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="Optional final report output path; use .json or .html or override with --report-format",
    )
    parser.add_argument(
        "--report-format",
        choices=REPORT_FORMAT_CHOICES,
        default="auto",
        help="Final report format override; default auto infers from the file extension",
    )
    parser.add_argument(
        "--report-json",
        dest="report_json_compat",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the fully resolved submit plan and payload(s) as JSON without uploading or submitting",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging"
    )
    args = parser.parse_args()
    if not any(
        (
            args.input_dir,
            args.input_glob,
            args.input_manifest,
            args.s3_uri,
            args.s3_uri_manifest,
        )
    ):
        parser.error(
            "one of the arguments --input-dir --input-glob --input-manifest --s3-uri --s3-uri-manifest is required"
        )
    if args.live_report and args.live_json_compat:
        parser.error("Use only one of --live-report or --live-json")
    if args.report and args.report_json_compat:
        parser.error("Use only one of --report or --report-json")
    if args.debug_sample_interval < 0:
        parser.error("--debug-sample-interval must be greater than or equal to zero")
    if args.job_count is not None and args.job_count <= 0:
        parser.error("--job-count must be greater than zero")
    if args.run_time is not None:
        try:
            parsed_run_time = _parse_duration_seconds(args.run_time)
        except ValueError as exc:
            parser.error(str(exc))
        if parsed_run_time <= 0:
            parser.error("--run-time must be greater than zero")
    if args.progress_interval <= 0:
        parser.error("--progress-interval must be greater than zero")
    if args.dry_run_preview_count <= 0:
        parser.error("--dry-run-preview-count must be greater than zero")
    if args.mock_failure_rate is not None and not (
        0.0 <= args.mock_failure_rate <= 1.0
    ):
        parser.error("--mock-failure-rate must be between 0 and 1 inclusive")
    if args.force_failure_every is not None and args.force_failure_every <= 0:
        parser.error("--force-failure-every must be greater than zero")
    if args.soft_sla_seconds is not None and args.soft_sla_seconds < 0:
        parser.error("--soft-sla-seconds must be greater than or equal to zero")
    if args.hard_sla_seconds is not None and args.hard_sla_seconds < 0:
        parser.error("--hard-sla-seconds must be greater than or equal to zero")
    if args.sla_step_every_jobs <= 0:
        parser.error("--sla-step-every-jobs must be greater than zero")
    if args.sla_step_cycle is not None and args.sla_step_cycle <= 0:
        parser.error("--sla-step-cycle must be greater than zero")
    if (
        args.soft_sla_seconds is not None
        and args.hard_sla_seconds is not None
        and args.soft_sla_seconds > args.hard_sla_seconds
    ):
        parser.error(
            "--soft-sla-seconds must be less than or equal to --hard-sla-seconds"
        )
    if args.soft_sla_step_seconds is not None and args.soft_sla_seconds is None:
        parser.error("--soft-sla-step-seconds requires --soft-sla-seconds")
    if args.hard_sla_step_seconds is not None and args.hard_sla_seconds is None:
        parser.error("--hard-sla-step-seconds requires --hard-sla-seconds")
    llm_pool_cycle = _parse_csv_values(args.llm_pool_cycle)
    purge_annotators = _parse_csv_values(args.purge_annotators)
    if args.llm_pool_cycle is not None and not llm_pool_cycle:
        parser.error("--llm-pool-cycle must contain at least one pool ID")
    if args.llm_pool_id is not None and not args.llm_pool_id.strip():
        parser.error("--llm-pool-id cannot be empty")
    if args.purge_annotators is not None and not purge_annotators:
        parser.error("--purge-annotators must contain at least one annotator name")
    if args.min_soft_sla_compliance_pct is not None and not (
        0.0 <= args.min_soft_sla_compliance_pct <= 100.0
    ):
        parser.error(
            "--min-soft-sla-compliance-pct must be between 0 and 100 inclusive"
        )
    if args.min_hard_sla_compliance_pct is not None and not (
        0.0 <= args.min_hard_sla_compliance_pct <= 100.0
    ):
        parser.error(
            "--min-hard-sla-compliance-pct must be between 0 and 100 inclusive"
        )
    args.live_report = args.live_report or args.live_json_compat
    args.report = args.report or args.report_json_compat
    args.llm_pool_cycle_values = (
        llm_pool_cycle if args.llm_pool_cycle is not None else None
    )
    args.purge_annotators_values = (
        purge_annotators if args.purge_annotators is not None else None
    )
    return args


def _load_json(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())


def _resolve_runtime_config(
    args: argparse.Namespace,
) -> Tuple[
    str,
    int,
    Optional[int],
    str,
    str,
    Optional[Dict[str, Any]],
    Optional[Dict[str, Any]],
]:
    config_payload: Dict[str, Any] = _load_json(args.config) if args.config else {}

    api_base_url = config_payload.get("api_base_url")
    parsed = urlparse(api_base_url) if api_base_url else None
    default_protocol = parsed.scheme if parsed and parsed.scheme else "http"
    default_host = parsed.hostname if parsed and parsed.hostname else "localhost"
    default_port = (
        parsed.port
        if parsed and parsed.port
        else (51000 if default_protocol == "http" else 52000)
    )

    protocol = args.protocol or default_protocol
    gateway_host = args.gateway_host or default_host
    if args.gateway_port is not None:
        gateway_port = args.gateway_port
    elif protocol == default_protocol:
        gateway_port = default_port
    else:
        gateway_port = 52000 if protocol == "grpc" else 51000

    if args.http_port is not None:
        http_port = args.http_port
    elif protocol == "http":
        http_port = default_port if default_protocol == "http" else 51000
    else:
        http_port = None

    api_key = (
        args.api_key if args.api_key is not None else config_payload.get("api_key")
    )
    if not api_key:
        raise ValueError("API key is required. Provide --api-key or config.api_key")

    s3_config = config_payload.get("storage") or None

    queue_config = config_payload.get("queue")
    return (
        gateway_host,
        gateway_port,
        http_port,
        protocol,
        api_key,
        s3_config,
        queue_config,
    )


async def main() -> None:
    if IMPORT_ERROR is not None:
        raise SystemExit(
            f"Error importing dependencies: {IMPORT_ERROR}\n"
            "Make sure marie, aiohttp, grpcio, and pika are installed."
        )

    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    run_time_seconds = (
        _parse_duration_seconds(args.run_time) if args.run_time is not None else None
    )

    (
        gateway_host,
        gateway_port,
        http_port,
        protocol,
        api_key,
        s3_config,
        queue_config,
    ) = _resolve_runtime_config(args)

    template_job_name = None
    template_metadata = None
    if args.request_template:
        template_job_name, template_metadata = _extract_template(
            _load_json(args.request_template)
        )

    job_name = args.job_name or template_job_name
    if not job_name:
        raise ValueError(
            "--job-name is required unless the request template contains invoke_action.name"
        )

    input_assets = _build_input_assets(
        input_glob=args.input_glob,
        input_dir=args.input_dir,
        input_manifest=args.input_manifest,
        s3_uri=args.s3_uri,
        s3_uri_manifest=args.s3_uri_manifest,
    )
    if not input_assets:
        raise ValueError("No inputs resolved for stress run")
    if (
        any(asset.local_path is not None for asset in input_assets)
        and not s3_config
        and not args.dry_run
    ):
        raise ValueError(
            "S3 storage config is required for local upload mode under config.storage"
        )

    if queue_config is None:
        logger.warning(
            "No queue configuration found in config. End-to-end event metrics will be incomplete."
        )

    stresser = GatewayE2EStresser(
        gateway_host=gateway_host,
        gateway_port=gateway_port,
        http_port=http_port,
        protocol=protocol,
        endpoint=args.endpoint,
        api_key=api_key,
        queue_name=job_name,
        planner=args.planner,
        input_assets=input_assets,
        job_count=args.job_count,
        run_time_seconds=run_time_seconds,
        submit_concurrency=args.submit_concurrency,
        submit_rate=args.submit_rate,
        timeout=args.timeout,
        terminal_timeout=args.terminal_timeout,
        s3_config=s3_config,
        queue_config=queue_config,
        metadata_template=template_metadata,
        template_job_name=template_job_name,
        fault_profile=args.fault_profile,
        aimock_admin_url=args.aimock_admin_url,
        soft_sla_seconds=args.soft_sla_seconds,
        hard_sla_seconds=args.hard_sla_seconds,
        soft_sla_step_seconds=args.soft_sla_step_seconds,
        hard_sla_step_seconds=args.hard_sla_step_seconds,
        sla_step_every_jobs=args.sla_step_every_jobs,
        sla_step_cycle=args.sla_step_cycle,
        min_soft_sla_compliance_pct=args.min_soft_sla_compliance_pct,
        min_hard_sla_compliance_pct=args.min_hard_sla_compliance_pct,
        ref_type=args.ref_type,
        policy=args.policy,
        project_id=args.project_id,
        llm_pool_id=args.llm_pool_id,
        llm_pool_cycle=args.llm_pool_cycle_values,
        purge_annotators=args.purge_annotators_values,
        mock_failure_rate=args.mock_failure_rate,
        mock_failure_mode=args.mock_failure_mode,
        force_failure_every=args.force_failure_every,
        upload_companion_meta=not args.skip_companion_meta_upload,
        batch_size=args.batch_size,
        progress_interval=args.progress_interval,
        live_report_path=args.live_report,
        live_report_format=args.live_report_format,
        debug_sample_interval=args.debug_sample_interval,
        dry_run_preview_count=args.dry_run_preview_count,
    )

    if args.dry_run:
        dry_run_payload = stresser.build_dry_run_plan()
        rendered = json.dumps(dry_run_payload, indent=2)
        print(rendered)
        if args.report:
            resolved_format = resolve_report_format(args.report, args.report_format)
            rendered_report = render_dry_run_report(dry_run_payload, resolved_format)
            write_text_atomically(args.report, rendered_report)
            logger.info("Wrote dry-run %s report to %s", resolved_format, args.report)
        return

    await stresser.run()
    stresser.print_report()
    if args.report:
        resolved_format = resolve_report_format(args.report, args.report_format)
        stresser.write_report(args.report, args.report_format)
        logger.info("Wrote %s report to %s", resolved_format, args.report)
    if stresser.verification_errors:
        raise SystemExit(2)


if __name__ == "__main__":
    asyncio.run(main())
