"""Submit a document extraction job, optionally wait and download its results."""

from __future__ import annotations

import argparse
import json
import math
import os
import ssl
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

import pika
import requests


def process_extract(
    uri: str,
    api_base_url: str,
    api_key: str,
    project_id: str,
    pipeline: str = "default",
    mode: str = "sparse",
    timeout: float = 30.0,
    queue_id: str | None = None,
    parse_mode: str = "auto",
) -> dict[str, Any]:
    """Submit an extraction job and return its acceptance response and job ID."""
    if uri.startswith("s3://"):
        key = uri.removeprefix("s3://").partition("/")[2]
        filename = key.rsplit("/", 1)[-1]
    else:
        filename = unquote(Path(urlsplit(uri).path).name)
    if not filename:
        raise ValueError("The input URI must identify a document")

    metadata = {
        "on": "extract_executor://document/extract",
        "project_id": project_id,
        "ref_id": filename,
        "ref_type": "extract",
        "uri": uri,
        "planner": "extract",
        "policy": "allow_all",
        "mode": mode,
        "run_params": {"parse_mode": parse_mode},
        "features": [{"type": "pipeline", "name": pipeline}],
    }
    if queue_id is not None:
        metadata["queue_id"] = queue_id
    payload = {
        "header": {},
        "parameters": {
            "invoke_action": {
                "action_type": "command",
                "command": "job",
                "action": "submit",
                "name": "extract",
                "api_key": api_key,
                "metadata": metadata,
            }
        },
        "data": [],
    }
    response = requests.post(
        f"{api_base_url.rstrip('/')}/api/v1/invoke",
        headers={"Authorization": f"Bearer {api_key}"},
        json=payload,
        timeout=timeout,
        allow_redirects=False,
    )
    response.raise_for_status()
    if 300 <= response.status_code < 400:
        raise RuntimeError("Gateway redirected the submission; use its final URL")
    body = response.json()
    result = body.get("parameters") if isinstance(body, dict) else None
    if not isinstance(result, dict):
        raise RuntimeError("Invalid gateway submission response")
    if result.get("status") != "ok":
        raise RuntimeError(
            result.get("msg") or result.get("error") or "Submission failed"
        )
    if not isinstance(result.get("job_id"), str) or not result["job_id"]:
        raise RuntimeError("Gateway submission response is missing job_id")
    return result


def configure_storage(storage_config: dict[str, Any]) -> None:
    """Register the example's S3 settings, including environment overrides."""
    from marie.storage import S3StorageHandler, StorageManager

    storage = dict(storage_config)
    for key in (
        "S3_ACCESS_KEY_ID",
        "S3_SECRET_ACCESS_KEY",
        "S3_SESSION_TOKEN",
        "S3_STORAGE_BUCKET_NAME",
        "S3_ENDPOINT_URL",
        "S3_REGION",
        "S3_ADDRESSING_STYLE",
    ):
        if key in os.environ:
            storage[key] = os.environ[key]
    os.environ["MARIE_S3_BUCKET"] = storage.get("S3_STORAGE_BUCKET_NAME", "marie")
    StorageManager.register_handler(S3StorageHandler(config=storage))


def upload_input(file_location: Path, storage_config: dict[str, Any]) -> str:
    """Upload a local document using the existing example storage configuration."""
    source = file_location.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input file not found: {source}")

    from marie.storage import StorageManager
    from marie.utils.asset_util import s3_asset_path

    configure_storage(storage_config)
    uri = s3_asset_path(ref_id=source.name, ref_type="extract", include_filename=True)
    if not StorageManager.write(str(source), uri, overwrite=True):
        raise RuntimeError(f"Could not upload {source}")
    return uri


def wait_for_completion(
    channel: pika.adapters.blocking_connection.BlockingChannel,
    queue: str,
    job_id: str,
    timeout: float,
) -> dict[str, Any]:
    """Wait for this job's terminal event, with a bounded timeout."""
    deadline = time.monotonic() + timeout
    try:
        for _, _, body in channel.consume(
            queue, auto_ack=True, inactivity_timeout=min(1.0, timeout)
        ):
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for job {job_id}")
            if body is None:
                continue
            try:
                event = json.loads(body)
            except (ValueError, UnicodeDecodeError):
                continue
            if not isinstance(event, dict) or event.get("jobid") != job_id:
                continue
            if event.get("event") == "extract.failed":
                raise RuntimeError(f"Job {job_id} failed")
            if event.get("event") == "extract.completed":
                return event
    finally:
        channel.cancel()
    raise RuntimeError(f"Event stream closed before job {job_id} completed")


def submit_and_wait(
    submission: dict[str, Any], queue_config: dict[str, Any], timeout: float
) -> dict[str, Any]:
    """Bind a private event queue before submitting, then wait for the job."""
    if not queue_config or not all(k in queue_config for k in ("username", "password")):
        raise ValueError("--wait requires the existing queue configuration")
    hostname = queue_config.get("hostname", "localhost")
    tls = queue_config.get("tls", False)
    parameters = pika.ConnectionParameters(
        host=hostname,
        port=int(queue_config.get("port", 5671 if tls else 5672)),
        virtual_host=queue_config.get("vhost", "/"),
        credentials=pika.PlainCredentials(
            queue_config["username"], queue_config["password"]
        ),
        ssl_options=(
            pika.SSLOptions(ssl.create_default_context(), hostname) if tls else None
        ),
        heartbeat=60,
        socket_timeout=10,
        blocked_connection_timeout=30,
    )
    with pika.BlockingConnection(parameters) as connection:
        channel = connection.channel()
        exchange = f"{submission['api_key']}.events"
        channel.exchange_declare(exchange=exchange, exchange_type="topic", durable=True)
        queue = channel.queue_declare(
            queue="", exclusive=True, auto_delete=True
        ).method.queue
        channel.queue_bind(exchange=exchange, queue=queue, routing_key="extract.#")
        accepted = process_extract(**submission)
        job_id = accepted["job_id"]
        print(f"Submitted job {job_id}; waiting for completion...", file=sys.stderr)
        return wait_for_completion(channel, queue, job_id, timeout)


def download_results(event: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    """Download the stored metadata and the files in its existing assets list."""
    from marie.storage import StorageManager
    from marie.utils.asset_util import s3_asset_path

    payload = event.get("payload")
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, dict) or not all(
        isinstance(payload.get(key), str) and payload[key]
        for key in ("ref_id", "ref_type")
    ):
        raise ValueError("Completion event is missing the document reference")
    root_uri = s3_asset_path(payload["ref_id"], payload["ref_type"]).rstrip("/") + "/"
    metadata_uri = root_uri + payload["ref_id"] + ".meta.json"
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_file = output_dir / metadata_uri.rsplit("/", 1)[-1]
    if not StorageManager.read_to_file(
        metadata_uri, str(metadata_file), overwrite=True
    ):
        raise RuntimeError("Could not download extraction metadata")
    metadata = json.loads(metadata_file.read_text())
    if not isinstance(metadata, dict) or not isinstance(
        metadata.get("assets", []), list
    ):
        raise ValueError("Invalid extraction metadata or assets list")
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(exist_ok=True)
    asset_count = 0
    for uri in metadata.get("assets", []):
        if uri == metadata_uri:
            continue
        if not isinstance(uri, str) or not uri.startswith(root_uri):
            raise ValueError("Unexpected asset URI in extraction metadata")
        destination = (assets_dir / uri.removeprefix(root_uri)).resolve()
        if not destination.is_relative_to(assets_dir):
            raise ValueError("Invalid asset path in extraction metadata")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not StorageManager.read_to_file(uri, str(destination), overwrite=True):
            raise RuntimeError(
                f"Could not download asset {destination.relative_to(assets_dir)}"
            )
        asset_count += 1
    return {
        "status": "completed",
        "job_id": event["jobid"],
        "metadata_file": str(metadata_file),
        "assets_dir": str(assets_dir),
        "asset_count": asset_count,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--uri", help="Existing document URI readable by the workers")
    source.add_argument("--input", type=Path, help="Local document to upload to S3")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.dev.json"),
        help="JSON API/storage configuration (default: examples/config.dev.json)",
    )
    parser.add_argument(
        "--api-base-url", help="Gateway origin, such as http://127.0.0.1:51000"
    )
    parser.add_argument("--project-id", help="Project ID for the submitted job")
    parser.add_argument("--pipeline", default="default")
    parser.add_argument(
        "--parse-mode",
        choices=("auto", "ocr"),
        default="auto",
        help="Use format-based plugin routing (auto, default) or force OCR",
    )
    parser.add_argument(
        "--mode",
        default="sparse",
        choices=("sparse", "word", "line", "raw_line", "multiline"),
    )
    parser.add_argument(
        "--timeout", type=float, default=30.0, help="HTTP submission timeout in seconds"
    )
    parser.add_argument(
        "--wait", action="store_true", help="Wait for extraction and download results"
    )
    parser.add_argument(
        "--wait-timeout",
        type=float,
        default=600.0,
        help="Completion timeout in seconds (default: 600)",
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        type=Path,
        help="Result directory for --wait (default: /tmp/marie/<job-id>)",
    )
    args = parser.parse_args(argv)
    if args.wait and (not math.isfinite(args.wait_timeout) or args.wait_timeout <= 0):
        parser.error("--wait-timeout must be a positive, finite number")

    try:
        config = json.loads(args.config.expanduser().read_text()) if args.config else {}
        if not isinstance(config, dict):
            raise ValueError("Configuration must be a JSON object")
        api_base_url = (
            args.api_base_url
            or os.environ.get("MARIE_API_BASE_URL")
            or config.get("api_base_url")
            or "http://127.0.0.1:51000"
        )
        api_key = os.environ.get("MARIE_API_KEY") or config.get("api_key")
        project_id = (
            args.project_id
            or os.environ.get("MARIE_PROJECT_ID")
            or config.get("project_id")
            or api_key
        )
        if not api_key:
            parser.error("Set MARIE_API_KEY or api_key in --config")
        uri = args.uri or upload_input(args.input, config.get("storage", {}))
        submission = dict(
            uri=uri,
            api_base_url=api_base_url,
            api_key=api_key,
            project_id=project_id,
            pipeline=args.pipeline,
            mode=args.mode,
            timeout=args.timeout,
            queue_id=config.get("default_queue_id"),
            parse_mode=args.parse_mode,
        )
        if args.wait:
            event = submit_and_wait(
                submission, config.get("queue", {}), args.wait_timeout
            )
            output_dir = args.output_dir or Path("/tmp/marie") / event["jobid"]
            if args.uri:
                configure_storage(config.get("storage", {}))
            result = download_results(event, output_dir)
        else:
            result = process_extract(**submission)
    except pika.exceptions.AMQPError as error:
        print(f"Event queue failed: {type(error).__name__}", file=sys.stderr)
        return 1
    except (OSError, ValueError, RuntimeError, requests.RequestException) as error:
        print(f"Extraction failed: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted; any submitted job continues running", file=sys.stderr)
        return 130
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
