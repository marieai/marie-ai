from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
import uuid
from copy import deepcopy
from datetime import datetime, timedelta
from functools import partial
from multiprocessing import Queue
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from aiohttp import ClientOSError
from docarray import DocList
from docarray.documents import TextDoc
from pydantic.tools import parse_obj_as

from marie import Client
from marie.logging_core.predefined import default_logger as logger
from marie.pipe.components import s3_asset_path
from marie.storage import StorageManager
from marie.utils.docs import frames_from_file
from marie.utils.image_utils import hash_frames_fast
from marie.utils.json import deserialize_value, load_json_file, store_json_object

# Add the parent directory of examples to the sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from examples.utils import (
    ServiceConfig,
    load_config,
    parse_args,
    setup_queue,
    setup_s3_storage,
)

job_to_file = {}
main_queue = Queue()
stop_event = threading.Event()
job_to_file_lock = threading.Lock()
job_to_file_path: str | None = None

# Configuration constants
REF_TYPE = "extract"
API_KEY = 'XXXX-XXXX-XXXX-XXXX'

EXTRACT_METADATA = {
    "on": "extract_executor://document/extract",
    "project_id": "project_id_000001",
    "ref_id": "doc_id_0001",
    "ref_type": "doc_type",
    "doc_id": "doc_id_0002",
    "doc_type": "doc_type",
    "uri": "s3://bucket/key",
    "policy": "allow_all",
    "planner": "extract",
    "type": "pipeline",
    "name": "default",
    "page_classifier": {"enabled": False},
    "page_splitter": {"enabled": False},
    "page_cleaner": {"enabled": False},
    "page_boundary": {"enabled": False},
    "template_matching": {"enabled": False, "definition_id": "0"},
}


def timer_func(func):
    """Timer decorator for performance measurement"""

    def wrap_func(*args, **kwargs):
        t1 = time.time()
        result = func(*args, **kwargs)
        t2 = time.time()
        logger.debug(f'Function {func.__name__!r} executed in {(t2 - t1):.4f}s')
        return result

    return wrap_func


def create_job_submit_request(metadata_json, queue_name: str):
    """Create job submission request with metadata"""
    metadata = deserialize_value(metadata_json) if metadata_json else {}
    param = {
        "invoke_action": {
            "action_type": "command",
            "api_key": API_KEY,
            "command": "job",
            "action": "submit",
            "name": queue_name,
            "metadata": metadata,
        }
    }
    docs = DocList[TextDoc]([TextDoc(text=f"Text : {_}") for _ in range(10)])
    return param, docs


def publish_extract(
    client: Client,
    s3_path: str,
    file_path: str,
    output_dir: str,
    queue_name: str,
    soft_sla: datetime = None,
    hard_sla: datetime = None,
) -> None:
    """
    Publish extract job to Marie server using new format
    """
    max_retries = 5
    backoff_factor = 2
    retry_delay = 1

    filename = os.path.basename(file_path)
    uid = filename

    meta = deepcopy(EXTRACT_METADATA)

    # Validate queue name
    if queue_name not in ["extract", "gen5_extract"]:
        raise ValueError(
            f"Invalid queue name: {queue_name}. Must be 'extract' or 'gen5_extract'"
        )

    ref_type = queue_name
    meta["planner"] = "extract" if queue_name == "extract" else "121880"
    meta["uri"] = s3_path
    meta["project_id"] = API_KEY
    meta["ref_id"] = uid
    meta["ref_type"] = ref_type
    meta["doc_id"] = uid
    meta["doc_type"] = ref_type
    meta["soft_sla"] = soft_sla.isoformat() if soft_sla else None
    meta["hard_sla"] = hard_sla.isoformat() if hard_sla else None

    parameters, docs = create_job_submit_request(
        metadata_json=meta, queue_name=queue_name
    )

    if not parameters:
        raise ValueError("Invalid command or action")

    logger.info(f"Sending request for: {s3_path}")
    logger.debug(f"Parameters: {parameters}")

    # Auth headers
    request_kwargs = {}
    headers = [("Authorization", f"Bearer {API_KEY}")]
    request_kwargs["headers"] = headers

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Attempt {attempt}: Sending request for {s3_path}")
            start_time = time.time()

            for resp in client.post(
                on="/api/v1/invoke",
                inputs=[],
                parameters=parameters,
                request_size=-1,
                return_responses=True,
                return_exceptions=True,
                asyncio=False,  # make synchronous to avoid per-thread event loops
                **request_kwargs,
            ):
                logger.info(f"Response: {resp}")
                json_result = deserialize_value(resp.parameters)
                logger.info(f"Adding job id: {json_result['job_id']}")

                # Update and persist atomically
                try:
                    # global job_to_file_path
                    with job_to_file_lock:
                        job_to_file[json_result["job_id"]] = {
                            "file": file_path,
                            "output_dir": output_dir,
                            "filename": filename,
                        }
                        if job_to_file_path:
                            store_json_object(dict(job_to_file), job_to_file_path)
                        else:
                            logger.warning(
                                "job_to_file_path is not set; cannot persist job_to_file."
                            )
                except Exception as persist_err:
                    logger.error(f"Failed to persist job_to_file: {persist_err}")

            end_time = time.time()
            elapsed_time = end_time - start_time
            logger.info(f"Request completed in {elapsed_time:.2f} seconds")
            return

        except ClientOSError as e:
            logger.error(f"Connection error on attempt {attempt}: {e}")
            if attempt == max_retries:
                raise ConnectionError(f"Max retries reached. Last error: {e}") from e
            retry_time = retry_delay * (backoff_factor ** (attempt - 1))
            logger.info(f"Retrying in {retry_time:.2f} seconds...")
            time.sleep(retry_time)
        except Exception as e:
            logger.error(f"Unexpected error on attempt {attempt}: {e}")
            raise


@timer_func
def process_request(
    client: Client,
    mode: str,
    file_location: str,
    output_dir: str,
    config: ServiceConfig,
    queue_name: str = "extract",
    soft_sla: datetime = None,
    hard_sla: datetime = None,
    stop_event: threading.Event = None,
) -> Any | None:
    """
    Process a single file request using the new format
    """
    if not os.path.exists(file_location):
        raise Exception(f"File not found: {file_location}")

    filename = os.path.basename(file_location)
    name = os.path.splitext(filename)[0]
    extension = os.path.splitext(filename)[1]

    logger.info(f"Processing file: {file_location}")
    logger.info(f"Queue: {queue_name}, Mode: {mode}")

    # Upload to S3
    s3_path = s3_asset_path(ref_id=filename, ref_type=queue_name, include_filename=True)

    StorageManager.mkdir("s3://marie")
    status = StorageManager.write(file_location, s3_path, overwrite=True)
    logger.info(f"Uploaded {file_location} to {s3_path}: {status}")

    # Submit job using new format
    uid = str(uuid.uuid4())
    main_queue.put(uid)

    try:
        publish_extract(
            client, s3_path, file_location, output_dir, queue_name, soft_sla, hard_sla
        )
        return {"status": "submitted", "s3_path": s3_path, "uid": uid}
    except Exception as e:
        logger.error(f"Error submitting job: {e}")
        return None


def process_dir(
    client: Client,
    src_dir: str,
    output_dir: str,
    config: ServiceConfig,
    queue_name: str = "extract",
    stop_event: threading.Event = None,
):
    """
    Process a directory of files using the new format
    """
    root_asset_dir = os.path.expanduser(src_dir)
    output_path = os.path.expanduser(output_dir)

    for img_path in Path(root_asset_dir).rglob("*"):
        if not img_path.is_file():
            continue

        logger.info(f"Processing: {img_path}")

        resolved_output_path = os.path.join(
            output_path, img_path.relative_to(root_asset_dir)
        )

        output_file_dir = os.path.dirname(resolved_output_path)
        filename = os.path.basename(resolved_output_path)
        name = os.path.splitext(filename)[0]
        extension = os.path.splitext(filename)[1]
        os.makedirs(output_file_dir, exist_ok=True)

        json_output_path = os.path.join(output_file_dir, f"{name}.json")

        if extension.lower() not in [".tif", ".tiff", ".png", ".jpg", ".jpeg"]:
            logger.warning(f"Skipping {img_path}: {extension} not supported")
            continue

        if os.path.exists(json_output_path):
            logger.info(f"Skipping {img_path}: {json_output_path} already exists")
            continue

        logger.info(f"Processing: {img_path}")

        # Calculate SLA times (optional)
        soft_sla = datetime.now() + timedelta(hours=2)
        hard_sla = soft_sla + timedelta(hours=4)

        json_result = process_request(
            client=client,
            mode="multiline",
            file_location=str(img_path),
            output_dir=output_file_dir,
            config=config,
            queue_name=queue_name,
            soft_sla=soft_sla,
            hard_sla=hard_sla,
            stop_event=stop_event,
        )

        logger.info(f"Result: {json_result}")


def message_handler(stop_event, message):
    """
    Handle message from queue
    """
    completed_event = False
    try:
        if isinstance(message, str):
            message = json.loads(message)

        event = message["event"]
        jobid = message["jobid"]

        logger.info(
            f"Message handler: queue size={main_queue.qsize()}, event={event}, jobid={jobid}"
        )

        if event != "extract.completed":
            return

        logger.info(f"Processing completed message: {message}")
        completed_event = True
        payload = json.loads(message["payload"])

        ref_id = payload["ref_id"]
        ref_type = payload["ref_type"]

        logger.info(f"Ref ID: {ref_id}, Ref Type: {ref_type}")

        s3_root_path = s3_asset_path(ref_id, ref_type)
        s3_path = os.path.join(s3_root_path, f"{ref_id}.meta.json")
        connected = StorageManager.ensure_connection("s3://", silence_exceptions=True)

        if not connected:
            logger.error("Error restoring assets: Could not connect to S3")
            return None

        if not StorageManager.exists(s3_path):
            logger.error(f"Error restoring assets: {s3_path} does not exist")
            return None

        if jobid not in job_to_file:
            logger.error(f"Error restoring assets: {jobid} not found in job_to_file")
            return None

        details = job_to_file[jobid]
        output_dir = details["output_dir"]
        filename = details["filename"]
        src_file = details["file"]
        name = os.path.splitext(filename)[0]
        output_path = os.path.join(output_dir, f"{name}.json")

        logger.info(f"Downloading results: {s3_path} to {output_path}")

        from marie.utils.utils import ensure_exists

        frames = frames_from_file(src_file)
        file_hash = hash_frames_fast(frames)

        logger.info(f"Extracted hash: {file_hash}")

        # Create local asset directory
        cache_dir = os.path.expanduser("~/.marie")
        generators_dir = os.path.join(cache_dir, "generators", file_hash)
        ensure_exists(generators_dir)

        logger.info(f"Copying {s3_root_path} to {generators_dir}")
        StorageManager.copy_remote(s3_root_path, generators_dir, overwrite=True)

        # copy the individual meta file
        StorageManager.read_to_file(s3_path, output_path)

        logger.info("Copying completed")

    except Exception as e:
        logger.error(f"Error in message handler: {e}")
    finally:
        if completed_event and main_queue.qsize() > 0:
            main_queue.get(False)

        if main_queue.empty():
            pass


if __name__ == "__main__":
    stop_event = threading.Event()
    args = parse_args()

    raw_config = load_config(args.config)
    storage_config = raw_config["storage"]
    queue_config = raw_config["queue"]
    config = parse_obj_as(ServiceConfig, raw_config)
    config.working_dir = args.output_dir

    # Use the API key from config
    API_KEY = config.api_key

    # Initialize job_to_file backing path and load if exists

    job_to_file_path = os.path.join(
        os.path.expanduser(config.working_dir), "job_to_file.json"
    )
    try:
        os.makedirs(os.path.dirname(job_to_file_path), exist_ok=True)
    except Exception as e:
        logger.warning(f"Unable to ensure job_to_file directory: {e}")
    if os.path.exists(job_to_file_path):
        logger.info(f"Loading job_to_file from {job_to_file_path}")
        job_to_file = load_json_file(job_to_file_path)

    print('job_to_file_path : ', job_to_file_path)
    print('Total jobs in job_to_file : ', len(job_to_file))

    setup_s3_storage(storage_config)
    setup_queue(
        queue_config,
        config.api_key,
        "extract",
        "extract.#",
        stop_event,
        None,
        partial(message_handler, stop_event),
    )

    api_base_url = raw_config["api_base_url"]

    try:
        parsed_url = urlparse(api_base_url)
        protocol = parsed_url.scheme or "http"
        netloc = parsed_url.netloc or parsed_url.path
        host = netloc.split(":")[0]
        port = netloc.split(":")[1] if ":" in netloc else "51000"
        address = f"{host}:{port}"
        print(f"Connecting to {address} using {protocol} protocol")
    except Exception as e:
        logger.error(f"Failed to parse API base URL '{api_base_url}': {e}")
        raise

    client = Client(
        host=host, port=int(port), protocol=protocol, request_size=-1, asyncio=False
    )

    # Default queue name - can be made configurable
    queue_name = "extract"

    if os.path.isfile(args.input):
        process_request(
            client=client,
            mode="multiline",
            file_location=args.input,
            output_dir=args.output_dir,
            config=config,
            queue_name=queue_name,
            stop_event=stop_event,
        )
    else:
        process_dir(
            client=client,
            src_dir=args.input,
            output_dir=args.output_dir,
            config=config,
            queue_name=queue_name,
            stop_event=stop_event,
        )

    while True:
        time.sleep(5000)
        logger.info("Main thread is alive")
