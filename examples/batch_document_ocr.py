import _thread
import base64
import json
import logging
import os
import signal
import sys
import threading
import time
import uuid
from functools import partial
from multiprocessing import Queue
from pathlib import Path

import requests

from examples.utils import setup_queue, online
from marie.ocr.extract_pipeline import s3_asset_path
from marie.storage import StorageManager
from marie.storage.s3_storage import S3StorageHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

api_base_url = "http://127.0.0.1:51000/api"
endpoint_url = f"{api_base_url}/document/extract"

default_queue_id = "0000-0000-0000-0000"
api_key = "mau_t6qDi1BcL1NkLI8I6iM8z1va0nZP01UQ6LWecpbDz6mbxWgIIIZPfQ"

auth_headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json; charset=utf-8",
}

job_to_file = {}

main_queue = Queue()


def process_extract(
    queue_id: str, mode: str, file_location: str, stop_event: threading.Event = None
) -> str:
    if not os.path.exists(file_location):
        raise Exception(f"File not found : {file_location}")

    logger.info(endpoint_url)
    if False and not online(api_base_url):
        stop_event.set()
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

    logger.info(f"Uploaded {file_location} to {s3_path}")

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


def process_dir(src_dir: str, output_dir: str, stop_event: threading.Event):
    root_asset_dir = os.path.expanduser(src_dir)
    output_path = os.path.expanduser(output_dir)

    for img_path in Path(root_asset_dir).rglob("*"):
        if not img_path.is_file():
            continue
        # if extension.lower() not in [".tif", ".tiff", ".png", ".jpg", ".jpeg"]:
        #     continue
        #
        print(img_path)

        resolved_output_path = os.path.join(
            output_path, img_path.relative_to(root_asset_dir)
        )
        output_dir = os.path.dirname(resolved_output_path)
        filename = os.path.basename(resolved_output_path)
        name = os.path.splitext(filename)[0]
        extension = os.path.splitext(filename)[1]
        os.makedirs(output_dir, exist_ok=True)

        json_result = process_extract(
            queue_id=default_queue_id,
            mode="multiline",
            file_location=img_path,
            stop_event=stop_event,
        )

        print(json_result)
        job_to_file[json_result["jobid"]] = {
            "file": img_path,
            "output_dir": output_dir,
            "filename": filename,
        }


def setup_storage():
    handler = S3StorageHandler(
        config={
            "S3_ACCESS_KEY_ID": "MARIEACCESSKEY",
            "S3_SECRET_ACCESS_KEY": "MARIESECRETACCESSKEY",
            "S3_STORAGE_BUCKET_NAME": "marie",
            "S3_ENDPOINT_URL": "http://localhost:8000",
            "S3_ENDPOINT_URLZZ": "http://gext-05.rms-asp.com:8000",
            "S3_ADDRESSING_STYLE": "path",
        }
    )

    # export AWS_ACCESS_KEY_ID=MARIEACCESSKEY; export AWS_SECRET_ACCESS_KEY=MARIESECRETACCESSKEY;  aws s3 ls --endpoint-url http://localhost:8000
    StorageManager.register_handler(handler=handler)
    StorageManager.ensure_connection("s3://")


def message_handler(stop_event, message):
    print("message_handler : ", main_queue.qsize())
    completed_event = False
    try:
        if isinstance(message, str):
            message = json.loads(message)

        event = message["event"]
        jobid = message["jobid"]

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
            stop_event.set()


if __name__ == "__main__":
    stop_event = threading.Event()
    setup_storage()

    connection_config = {
        "hostname": "localhost",
    }

    setup_queue(
        connection_config,
        api_key,
        "extract",
        "extract.#",
        stop_event,
        None,
        partial(message_handler, stop_event),
        # lambda x: print(f"callback: {x}"),
    )

    process_dir(
        "~/datasets/private/medical_page_classification/small",
        "/tmp/medical_page_classification",
        stop_event,
    )
    # join current thread / wait for event or we will get "cannot schedule new futures after interpreter shutdown"
    while not stop_event.is_set():
        stop_event.wait()
        print("Exiting")
        # sys.exit(0) # exit the main thread
        os._exit(0)  # exit all threads
