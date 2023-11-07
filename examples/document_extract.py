import base64
import json
import os
import threading
import time
import uuid

import requests

from examples.utils import setup_queue, online, setup_s3_storage
from marie.pipe.components import s3_asset_path
from marie.storage import StorageManager
from marie.utils.json import store_json_object
from marie.utils.utils import ensure_exists
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

api_base_url = "http://127.0.0.1:51000/api"
api_base_url = "http://172.20.10.12:51000/api"
endpoint_url = f"{api_base_url}/document/extract"

default_queue_id = "0000-0000-0000-0000"
api_key = "mau_t6qDi1BcL1NkLI8I6iM8z1va0nZP01UQ6LWecpbDz6mbxWgIIIZPfQ"

auth_headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json; charset=utf-8",
}


def process_extract(
    queue_id: str, mode: str, file_location: str, stop_event: threading.Event = None
) -> str:
    if not os.path.exists(file_location):
        raise Exception(f"File not found : {file_location}")

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

    s3_path = s3_asset_path(ref_id=filename, ref_type="lbx", include_filename=True)

    status = StorageManager.write(file_location, s3_path, overwrite=True)
    logger.info(f"Uploaded {file_location} to {s3_path} : {status}")
    uid = str(uuid.uuid4())

    # Treat the image as a single word.
    # WORD = "word"
    # Sparse text. Find as much text as possible in no particular order.
    # SPARSE = "sparse"
    # Treat the image as a single text line.
    # LINE = "line"
    # Raw line. Treat the image as a single text line, NO bounding box detection performed.
    # RAW_LINE = "raw_line"
    # Multiline. Treat the image as multiple text lines, NO bounding box detection performed.
    # MULTI_LINE = "multiline"

    # Attributes
    # data[null]=> base 64 encoded image
    # mode['word]=> extraction mode
    # output['json']=> json,text,pdf

    json_payload = {
        "queue_id": queue_id,
        # "data": base64_str,
        # "doc_id": f"extract-{uid}",
        "uri": s3_path,
        "doc_id": f"{filename}",
        "doc_type": "lbxid",
        # "uri": "s3://marie/incoming/PID_1764_8829_0_179519650.tif",
        "mode": mode,
        "output": "json",
        "features": [
            {
                "type": "pipeline",
                "name": "default",
                "page_classifier": {
                    "enabled": True,
                },
                "page_splitter": {
                    "enabled": True,
                },
                "ocr": {
                    "document": {
                        "model": "default",
                    },
                    "region": {
                        "model": "best",
                    },
                },
            }
        ],
    }

    print(f"Uploading to marie-ai for processing : {file}")
    print(endpoint_url)

    NITER = 1
    json_result = None

    for k in range(NITER):
        start = time.time()
        result = requests.post(
            endpoint_url,
            headers=auth_headers,
            json=json_payload,
        )

        if result.status_code != 200:
            stop_event.set()
            raise Exception(f"Error : {result}")

        json_result = result.json()
        print(json_result)
        delta = time.time() - start
        print(f"Request time : {delta}")

    return json_result


def message_handler(message):
    try:
        if isinstance(message, str):
            message = json.loads(message)

        event = message["event"]
        jobid = message["jobid"]

        print(f"event: {event}, jobid: {jobid}")
        print(message)

        if event != "extract.completed":
            return

        payload = json.loads(message["payload"])
        store_json_object(payload, "/tmp/marie/extract.json")
        store_json_object(message, "/tmp/marie/extract-event.json")

    except Exception as e:
        print(e)
        return


if __name__ == "__main__":
    ensure_exists("/tmp/marie")
    stop_event = threading.Event()

    storage_config = {
        "S3_ACCESS_KEY_ID": "MARIEACCESSKEY",
        "S3_SECRET_ACCESS_KEY": "MARIESECRETACCESSKEY",
        "S3_STORAGE_BUCKET_NAME": "marie",
        "S3_ENDPOINT_URLXX": "http://localhost:8000",
        "S3_ENDPOINT_URL": "http://172.16.11.163:8000",
        "S3_ADDRESSING_STYLE": "path",
    }

    setup_s3_storage(storage_config)

    connection_config = {
        "hostname": "localhost",
        "port": 5672,
        "username": "guest",
        "password": "guest",
    }

    setup_queue(
        connection_config,
        api_key,
        "extract",
        "extract.#",
        stop_event,
        [
            "extract.completedXXX",
            "extract.failedXXX",
        ],  # this will not work if we have multiple requests
        # lambda x: print(f"callback: {x}"),
        message_handler,
    )

    # Specify the path to the file you would like to process
    src = os.path.expanduser("~/tmp/PID_1028_7826_0_157684456.tif")
    src = os.path.expanduser("~/tmp/PID_1925_9289_0_157186264.png")
    # src = os.path.expanduser("~/tmp/page-level-classification/182972842.tif")
    # src = os.path.expanduser("~/tmp/page-level-classification/164770867.tif")
    # src = os.path.expanduser("~/tmp/page-level-classification/44035/181785075_1.png")

    json_result = process_extract(
        queue_id=default_queue_id,
        mode="multiline",
        file_location=src,
        stop_event=stop_event,
    )

    print(json_result)
