from __future__ import annotations

import base64
import json
import logging
import os
import threading
import time
import uuid
from functools import partial
from multiprocessing import Queue
from pathlib import Path

import requests
from pydantic import BaseModel
from pydantic.tools import parse_obj_as

from examples.utils import online, setup_queue, setup_s3_storage
from marie.pipe.components import s3_asset_path
from marie.storage import StorageManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(file_path):
    with open(file_path, "r") as json_file:
        config = json.load(json_file)
    return config


class ExtractConfig(BaseModel):
    api_base_url: str
    api_key: str
    default_queue_id: str


# kv_store = InMemoryKV()
job_to_file = {}

main_queue = Queue()


def process_extract(
    mode: str,
    file_location: str,
    config: ExtractConfig,
    stop_event: threading.Event = None,
) -> str:
    if not os.path.exists(file_location):
        raise Exception(f"File not found : {file_location}")

    api_base_url = config.api_base_url
    api_key = config.api_key
    queue_id = config.default_queue_id
    endpoint_url = f"{api_base_url}/document/extract"

    auth_headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json; charset=utf-8",
    }

    logger.info(endpoint_url)
    if False and not online(api_base_url):
        # stop_event.set()
        raise Exception(f"API server is not online : {endpoint_url}")

    # Prepare file for upload
    with open(file_location, "rb") as file:
        encoded_bytes = base64.b64encode(file.read())
        base64_str = encoded_bytes.decode("utf-8")

    filename = os.path.basename(file_location)
    name = os.path.splitext(filename)[0]
    extension = os.path.splitext(filename)[1]

    s3_path = s3_asset_path(
        ref_id=filename, ref_type="batch_document_ocr", include_filename=True
    )
    status = StorageManager.write(file_location, s3_path, overwrite=True)
    logger.info(f"Uploaded {file_location} to {s3_path} : {status}")
    uid = str(uuid.uuid4())

    json_payload = {
        "queue_id": queue_id,
        # "data": base64_str,
        "uri": s3_path,
        "mode": mode,
        "output": "json",
        "doc_id": f"{uid}",
        "doc_type": "batch_document_ocr",
        "features": [
            {
                "type": "pipeline",
                "name": "default",
                "page_classifier": {
                    "enabled": False,
                },
                "page_splitter": {
                    "enabled": False,
                },
                "page_cleaner": {
                    "enabled": False,
                },
            }
        ],
    }

    # Upload file to api
    logger.info(f"Uploading to marie-ai for processing : {file}")

    main_queue.put(uid)
    start = time.time()
    result = requests.post(
        endpoint_url,
        headers=auth_headers,
        json=json_payload,
    )

    # if result.status_code != 200:
    #     stop_event.set()
    #     raise Exception(f"Error : {result}")

    json_result = result.json()
    delta = time.time() - start
    print(f"Request time : {delta}")
    return json_result


def process_dir(
    src_dir: str, output_dir: str, stop_event: threading.Event, config: ExtractConfig
):
    root_asset_dir = os.path.expanduser(src_dir)
    output_path = os.path.expanduser(output_dir)

    for img_path in Path(root_asset_dir).rglob("*"):
        if not img_path.is_file():
            continue

        print(img_path)

        resolved_output_path = os.path.join(
            output_path, img_path.relative_to(root_asset_dir)
        )
        output_dir = os.path.dirname(resolved_output_path)
        filename = os.path.basename(resolved_output_path)
        name = os.path.splitext(filename)[0]
        extension = os.path.splitext(filename)[1]
        os.makedirs(output_dir, exist_ok=True)
        json_output_path = os.path.join(output_dir, f"{name}.json")

        if extension.lower() not in [".tif", ".tiff", ".png", ".jpg", ".jpeg"]:
            logger.warning(f"Skipping {img_path} : {extension} not supported")
            continue

        if os.path.exists(json_output_path):
            logger.warning(f"Skipping {img_path} : {json_output_path} already exists")
            continue

        json_result = process_extract(
            mode="multiline",
            file_location=img_path,
            stop_event=stop_event,
            config=config,
        )

        print(json_result)

        job_to_file[json_result["jobid"]] = {
            "file": img_path,
            "output_dir": output_dir,
            "filename": filename,
        }

        # break


def message_handler(stop_event, message):
    """
    Handle message from queue
    :param stop_event:
    :param message:
    :return:
    """
    completed_event = False
    try:
        if isinstance(message, str):
            message = json.loads(message)

        event = message["event"]
        jobid = message["jobid"]

        print("message_handler : ", main_queue.qsize(), event, jobid)

        if event != "extract.completed":
            return

        print(message)
        completed_event = True
        payload = json.loads(message["payload"])
        ref_id = payload["metadata"]["ref_id"]
        ref_type = payload["metadata"]["ref_type"]

        s3_root_path = s3_asset_path(ref_id, ref_type)
        s3_path = os.path.join(s3_root_path, "results", f"{ref_id}.json")
        connected = StorageManager.ensure_connection("s3://", silence_exceptions=True)

        if not connected:
            logger.error(f"Error restoring assets : Could not connect to S3")
            return None

        if not StorageManager.exists(s3_path):
            logger.error(f"Error restoring assets : {s3_path} does not exist")
            return None

        if jobid not in job_to_file:
            logger.error(f"Error restoring assets : {jobid} not found in job_to_file")
            return None

        details = job_to_file[jobid]

        output_dir = details["output_dir"]
        filename = details["filename"]
        name = os.path.splitext(filename)[0]
        output_path = os.path.join(output_dir, f"{name}.json")

        logger.info(f"Downloading results : {s3_path} to {output_path}")
        StorageManager.read_to_file(s3_path, output_path, overwrite=True)
    except Exception as e:
        logger.error(e)
    finally:
        if completed_event and main_queue.qsize() > 0:
            main_queue.get(False)

        if main_queue.empty():
            pass
            # stop_event.set()


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        default="config.dev.json",
        help="Path to the config file.",
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the input directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory the output images will be written to.",
    )
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    stop_event = threading.Event()
    args = parse_args()

    raw_config = load_config(args.config)
    storage_config = raw_config["storage"]
    queue_config = raw_config["queue"]
    config = parse_obj_as(ExtractConfig, raw_config)

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

    process_dir(
        src_dir=args.input_dir,
        output_dir=args.output_dir,
        stop_event=stop_event,
        config=config,
    )

    # join current thread / wait for event or we will get "cannot schedule new futures after interpreter shutdown"

    while True:
        time.sleep(10)
    # get curren thread
    current_thread = threading.current_thread()
    current_thread.join()

    # stop_event.wait()
    if False:
        while not stop_event.is_set():
            stop_event.wait()
            print("Exiting")
            # sys.exit(0) # exit the main thread
            os._exit(0)  # exit all threads

    # cleanup empty files, this can happen for example when the file is not an image or service fails
    #  find $dir -size 0 -type f -delete
