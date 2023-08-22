import base64
import glob
import os
import sys
import threading
import time
import uuid

import requests
from PIL import Image

from marie.executor.ner.utils import visualize_icr
from marie.utils.utils import ensure_exists
from examples.utils import setup_queue

api_base_url = "http://127.0.0.1:51000/api"
default_queue_id = "0000-0000-0000-0000"
api_key = "mau_t6qDi1BcL1NkLI8I6iM8z1va0nZP01UQ6LWecpbDz6mbxWgIIIZPfQ"

auth_headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json; charset=utf-8",
}


def online(api_ulr) -> bool:
    r = requests.head(api_ulr)
    print(r.status_code)
    # The 308 (Permanent Redirect)
    return r.status_code == 200 or r.status_code == 308


def process_extract(
    queue_id: str, mode: str, file_location: str, stop_event: threading.Event = None
) -> str:
    if not os.path.exists(file_location):
        raise Exception(f"File not found : {file_location}")
    endpoint_url = f"{api_base_url}/extract"
    # upload_url = f"{api_base_url}/overlay"
    # upload_url = f"{api_base_url}/ner/{queue_id}"
    # upload_url = f"{api_base_url}/ner"

    print(endpoint_url)
    if False and not online(api_base_url):
        stop_event.set()
        raise Exception(f"API server is not online : {endpoint_url}")

    # Prepare file for upload
    with open(file_location, "rb") as file:
        encoded_bytes = base64.b64encode(file.read())
        base64_str = encoded_bytes.decode("utf-8")

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

    uid = str(uuid.uuid4())
    queue_id = "0000-0000-0000-0000"

    json_payload = {"data": base64_str, "mode": mode, "output": "assets"}
    # json_payload = {"data": base64_str, "mode": mode, "output": "assets"}

    json_payload = {
        "queue_id": queue_id,
        "data": base64_str,
        # "uri": "s3://marie/incoming/PID_1764_8829_0_179519650.tif",
        "mode": mode,
        "output": "json",
        "doc_id": f"extract-{uid}",
        "doc_type": "lbx",
        # "features": [{"type": "LABEL_DETECTION", "maxResults": 1}],
    }

    # print(json_payload)
    # Upload file to api
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


def process_dir(image_dir: str):
    for idx, img_path in enumerate(
        glob.glob(os.path.join(os.path.expanduser(image_dir), "*.*"))
    ):
        try:
            icr_data = process_extract(
                queue_id=default_queue_id, mode="multiline", file_location=img_path
            )
            print(icr_data)
            image = Image.open(src).convert("RGB")
            visualize_icr(image, icr_data)
        except Exception as e:
            print(e)
            # raise e


if __name__ == "__main__":
    ensure_exists("/tmp/snippet")

    # Specify the path to the file you would like to process
    src = "./set-001/test/fragment-003.png"

    if False:
        process_dir("~/dataset/assets-private/corr-indexer/validation")

    stop_event = threading.Event()

    setup_queue(
        api_key,
        "events",
        stop_event,
        "extract.completed",
        lambda x: print(f"callback: {x}"),
    )

    if True:
        src = os.path.expanduser("~/tmp/PID_1028_7826_0_157684456.tif")
        print(src)

        json_result = process_extract(
            queue_id=default_queue_id,
            mode="multiline",
            file_location=src,
            stop_event=stop_event,
        )

        print(json_result)
        # image = Image.open(src).convert("RGB")
        # visualize_icr(image, icr_data)
