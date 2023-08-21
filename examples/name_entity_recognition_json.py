import base64
import glob
import os
import time
import uuid

import requests
from PIL import Image

from marie.executor.ner.utils import visualize_icr
from marie.utils.utils import ensure_exists

api_base_url = "http://192.168.102.65:51000/api"

default_queue_id = "0000-0000-0000-0000"
api_key = "mau_t6qDi1BcL1NkLI8I6iM8z1va0nZP01UQ6LWecpbDz6mbxWgIIIZPfQ"

auth_headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json; charset=utf-8",
}


def online(api_ulr) -> bool:
    import requests

    r = requests.head(api_ulr)
    # The 308 (Permanent Redirect)
    return r.status_code == 200 or r.status_code == 308


def process_extract(queue_id: str, mode: str, file_location: str) -> str:
    if not os.path.exists(file_location):
        raise Exception(f"File not found : {file_location}")
    upload_url = f"{api_base_url}/ner/{queue_id}"
    upload_url = f"{api_base_url}/extract"
    # upload_url = f"{api_base_url}/overlay"
    # upload_url = f"{api_base_url}/ner/{queue_id}"
    # upload_url = f"{api_base_url}/ner"

    print(api_base_url)
    print(upload_url)
    if False and not online(api_base_url):
        raise Exception(f"API server is not online : {api_base_url}")

    # Prepare file for upload
    with open(file_location, "rb") as file:
        encoded_bytes = base64.b64encode(file.read())
        base64_str = encoded_bytes.decode("utf-8")

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
    }

    # print(json_payload)
    # Upload file to api
    print(f"Uploading to marie-ai for processing : {file}")
    print(upload_url)

    NITER = 1
    json_result = None

    for k in range(NITER):
        start = time.time()
        result = requests.post(
            upload_url,
            headers=auth_headers,
            json=json_payload,
        )

        if result.status_code != 200:
            print(result.text)
            raise Exception(f"Error : {result}")

        json_result = result.json()
        print(json_result)
        delta = time.time() - start
        print(f"Request time : {delta}")

    return json_result


def process_dir(image_dir: str):
    for idx, img_path in enumerate(glob.glob(os.path.join(image_dir, "*.*"))):
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


def setup_queue(api_key: str):
    print(f"Setting up queue for : {api_key}")


if __name__ == "__main__":
    ensure_exists("/tmp/snippet")

    # Specify the path to the file you would like to process
    src = "./set-001/test/fragment-003.png"

    if False:
        process_dir("/home/greg/dataset/assets-private/corr-indexer/validation")

    setup_queue(api_key)

    if True:
        src = "~/tmp/PID_1925_9289_0_157186264.tif"
        src = "~/tmp/PID_1028_7826_0_157684456.tif"
        src = os.path.expanduser(src)
        print(src)

        icr_data = process_extract(
            queue_id=default_queue_id, mode="multiline", file_location=src
        )

        print(icr_data)
        image = Image.open(src).convert("RGB")
        visualize_icr(image, icr_data)
