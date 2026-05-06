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
import statistics
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

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
            return (self.started_at - self.submit_finished_at) * 1000.0
        if self.scheduled_at is not None:
            return (self.scheduled_at - self.submit_finished_at) * 1000.0
        return None

    @property
    def scheduling_ms(self) -> Optional[float]:
        if self.submit_finished_at is None or self.scheduled_at is None:
            return None
        return (self.scheduled_at - self.submit_finished_at) * 1000.0

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

    @property
    def throughput(self) -> float:
        if self.start_time is None or self.end_time is None:
            return 0.0
        duration = self.end_time - self.start_time
        if duration <= 0:
            return 0.0
        return self.total_jobs / duration


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
        job_count: int,
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
        ref_type: Optional[str] = None,
        policy: str = "allow_all",
        project_id: Optional[str] = None,
        upload_companion_meta: bool = True,
        batch_size: int = 1,
        debug_sample_interval: float = 0.0,
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
        self.job_count = job_count
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
        self.ref_type = ref_type or self.queue_name
        self.policy = policy
        self.project_id = project_id or api_key
        self.upload_companion_meta = upload_companion_meta
        self.batch_size = batch_size
        self.debug_sample_interval = max(0.0, debug_sample_interval)
        self.requires_upload = any(
            asset.local_path is not None for asset in input_assets
        )

        self.metrics = E2EMetrics(total_jobs=job_count)
        self._logger = logging.getLogger(self.__class__.__name__)
        self._runs_by_request_id: Dict[str, JobRun] = {}
        self._runs_by_job_id: Dict[str, JobRun] = {}
        self._runs_by_ref_id: Dict[str, JobRun] = {}
        self._debug_samples: List[DebugSnapshot] = []
        self._state_lock = threading.Lock()
        self._http_session: Optional[aiohttp.ClientSession] = None
        self._event_consumer: Optional[SchedulerEventConsumer] = None

    def _setup_storage(self) -> None:
        if not self.requires_upload:
            return
        bucket = self.s3_config.get("S3_STORAGE_BUCKET_NAME", "marie")
        os.environ["MARIE_S3_BUCKET"] = str(bucket)
        handler = S3StorageHandler(config=self.s3_config)
        StorageManager.register_handler(handler=handler)
        StorageManager.ensure_connection("s3://")
        StorageManager.mkdir(f"s3://{bucket}")

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

    def _handle_event_message(self, message: Dict[str, Any]) -> None:
        event_name = message.get("event")
        job_id = message.get("jobid")
        ref_id = _extract_ref_id_from_event(message)
        timestamp = _now()

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
        ref_id = f"{request_id}-{asset.source_name}"
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
        )

    def _build_metadata(self, run: JobRun) -> Dict[str, Any]:
        template_vars = {
            "request_id": run.request_id,
            "job_index": str(run.job_index),
            "timestamp": str(int(_now())),
            "timestamp_ms": str(int(_now() * 1000)),
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

        metadata["planner"] = self.planner
        metadata["project_id"] = self.project_id
        metadata["ref_id"] = run.ref_id
        metadata["ref_type"] = run.ref_type
        metadata["policy"] = metadata.get("policy", self.policy)
        metadata["stress_fault_profile"] = self.fault_profile
        metadata["uri"] = run.s3_uri
        return metadata

    def _upload_to_s3(self, run: JobRun) -> Dict[str, Any]:
        if run.input_mode != "upload":
            run.upload_started_at = _now()
            metadata = self._build_metadata(run)
            run.upload_finished_at = run.upload_started_at
            return metadata

        run.upload_started_at = _now()
        source_path = Path(run.source_path)
        metadata = self._build_metadata(run)
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
        return metadata

    async def _submit_http(self, run: JobRun, metadata: Dict[str, Any]) -> SubmitResult:
        start_time = _now()
        payload = {
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
            docs = DocumentArray(
                [
                    Document(text=f"stress-{run.request_id}-{i}")
                    for i in range(self.batch_size)
                ]
            )
            response_docs = None
            async for response in client.post(
                self.endpoint,
                inputs=docs,
                parameters={
                    "invoke_action": {
                        "action_type": "command",
                        "command": "job",
                        "action": "submit",
                        "name": self.queue_name,
                        "api_key": self.api_key,
                        "metadata": metadata,
                    }
                },
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
            metadata = await asyncio.to_thread(self._upload_to_s3, run)
            run.submit_started_at = _now()
            if self.protocol == "http":
                result = await self._submit_http(run, metadata)
            else:
                result = await self._submit_grpc(run, metadata)
            self._mark_submit_result(run, result)

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
            await asyncio.sleep(5.0)
            with self._state_lock:
                completed = 0
                failed = 0
                submitted = 0
                submit_failed = 0
                for run in self._runs_by_request_id.values():
                    if run.job_id:
                        submitted += 1
                    if run.terminal_status == "completed":
                        completed += 1
                    elif run.terminal_status == "failed":
                        failed += 1
                    elif run.terminal_status == "submit_failed":
                        submit_failed += 1
            self._logger.info(
                "Progress: submitted=%s/%s completed=%s failed=%s submit_failed=%s",
                submitted,
                self.job_count,
                completed,
                failed,
                submit_failed,
            )

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
        metrics = E2EMetrics(total_jobs=self.job_count)
        metrics.start_time = self.metrics.start_time
        metrics.end_time = self.metrics.end_time

        with self._state_lock:
            runs = list(self._runs_by_request_id.values())

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

        self.metrics = metrics

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

        try:
            if self.debug_sample_interval > 0:
                await self._capture_debug_snapshot("start")
                debug_sampler = asyncio.create_task(self._debug_sampler())

            tasks = []
            interval = 1.0 / self.submit_rate if self.submit_rate > 0 else 0.0
            for job_index in range(self.job_count):
                asset = self.input_assets[job_index % len(self.input_assets)]
                run = self._build_run(asset, job_index)
                self._register_run(run)
                tasks.append(asyncio.create_task(self._submit_run(run, semaphore)))
                if interval > 0:
                    await asyncio.sleep(interval)

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
        print(f"Jobs requested: {m.total_jobs}")
        print(f"Fault profile: {self.fault_profile}")
        print(f"Jobs submitted: {m.submitted_jobs}")
        print(f"Completed: {m.completed_jobs}")
        print(f"Failed: {m.failed_jobs}")
        print(f"Submit failed: {m.submit_failed_jobs}")
        print(f"Timed out waiting for events: {m.event_timeout_jobs}")
        print(f"Throughput: {m.throughput:.2f} jobs/s")

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

        print("=" * 78 + "\n")

    def write_json_report(self, output_path: str) -> None:
        with self._state_lock:
            runs = list(self._runs_by_request_id.values())
            debug_samples = list(self._debug_samples)

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

        payload = {
            "summary": {
                "total_jobs": self.metrics.total_jobs,
                "fault_profile": self.fault_profile,
                "submitted_jobs": self.metrics.submitted_jobs,
                "completed_jobs": self.metrics.completed_jobs,
                "failed_jobs": self.metrics.failed_jobs,
                "submit_failed_jobs": self.metrics.submit_failed_jobs,
                "event_timeout_jobs": self.metrics.event_timeout_jobs,
                "throughput": self.metrics.throughput,
                "debug_sample_count": len(debug_sample_payload),
            },
            "latencies_ms": {
                "submit": self.metrics.submit_latencies_ms,
                "scheduling": self.metrics.scheduling_latencies_ms,
                "queue_wait": self.metrics.queue_wait_ms,
                "execution": self.metrics.execution_latencies_ms,
                "end_to_end": self.metrics.end_to_end_latencies_ms,
            },
            "failure_reasons": self.metrics.failure_reasons,
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
                    "terminal_status": run.terminal_status,
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
        Path(output_path).write_text(json.dumps(payload, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gateway end-to-end stress tester for scheduler + LLM pipelines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/stress/gateway_e2e_stresser.py \\
      --config /home/gbugaj/dev/workflow/grapnel-g5/config.dev.json \\
      --input-dir /mnt/data/marie-ai/generators \\
      --job-count 50 \\
      --job-name gen5_extract \\
      --planner extract

  python tools/stress/gateway_e2e_stresser.py \\
      --config /home/gbugaj/dev/workflow/grapnel-g5/config.dev.json \\
      --s3-uri-manifest /tmp/stress-s3-uris.txt \\
      --job-count 20 \\
      --job-name gen5_extract \\
      --planner extract \\
      --submit-rate 2 \\
      --report-json /tmp/gateway-e2e-report.json

  python tools/stress/gateway_e2e_stresser.py \\
      --config /home/gbugaj/dev/workflow/grapnel-g5/config.dev.json \\
      --s3-uri s3://marie/gen5_extract/sample-001.tif \\
      --job-count 10 \\
      --job-name gen5_extract \\
      --planner extract \\
      --fault-profile chaos \\
      --aimock-admin-url http://localhost:4011
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to grapnel-style config JSON containing api_base_url, api_key, storage, and queue",
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

    parser.add_argument(
        "--job-count", type=int, required=True, help="Total jobs to submit"
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
        "--aimock-admin-url",
        type=str,
        default=None,
        help="Optional AIMock admin base URL used to set the active fault profile before the run, for example http://localhost:4011",
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
        "--report-json", type=str, default=None, help="Optional JSON report output path"
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
    if args.debug_sample_interval < 0:
        parser.error("--debug-sample-interval must be greater than or equal to zero")
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

    api_key = args.api_key or config_payload.get("api_key")
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
    if any(asset.local_path is not None for asset in input_assets) and not s3_config:
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
        ref_type=args.ref_type,
        policy=args.policy,
        project_id=args.project_id,
        upload_companion_meta=not args.skip_companion_meta_upload,
        batch_size=args.batch_size,
        debug_sample_interval=args.debug_sample_interval,
    )

    await stresser.run()
    stresser.print_report()
    if args.report_json:
        stresser.write_json_report(args.report_json)
        logger.info("Wrote JSON report to %s", args.report_json)


if __name__ == "__main__":
    asyncio.run(main())
