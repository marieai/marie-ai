import base64
import glob
import os
import time
import uuid

import requests
from PIL import Image

from marie.executor.ner.utils import visualize_icr
from marie.utils.utils import ensure_exists

api_base_url = "http://172.83.14.129:6000/api"  # Traefic loadballancer
api_base_url = "http://192.168.102.65:51000/api"
api_base_url = "http://192.168.1.14:51000/api"
api_base_url = "http://192.168.102.65:51000/api"
api_base_url = "http://192.168.102.65:51000/api"
# api_base_url = "http://184.105.3.112:51000/api"
# api_base_url = "http://traefik.localhost:5000/api"

api_base_url = "http://127.0.0.1:51000/api"

default_queue_id = "0000-0000-0000-0000"
api_key = "MY_API_KEY"
auth_headers = {"Authorization": f"Bearer {api_key}"}
headers = ({"Content-Type": "application/json; charset=utf-8"},)


def online(api_ulr) -> bool:
    import requests

    r = requests.head(api_ulr)
    # The 308 (Permanent Redirect)
    return r.status_code == 200 or r.status_code == 308


def process_extract(queue_id: str, mode: str, file_location: str) -> str:
    if not os.path.exists(file_location):
        raise Exception(f"File not found : {file_location}")
    upload_url = f"{api_base_url}/extract/{queue_id}"
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
        json_payload = {"data": base64_str, "mode": mode, "output": "assets"}
        json_payload = {
            "queue_id": uid,
            "data": base64_str,
            "mode": mode,
            "output": "json",
            "doc_id": f"greg-{uid}",
            "doc_type": "overlay",
            # "features": [{"type": "LABEL_DETECTION", "maxResults": 1}],
        }

        # print(json_payload)
        # Upload file to api
        print(f"Uploading to marie-ai for processing : {file}")
        print(upload_url)

        auth_headers = [
            {"Authorization": f"Bearer {api_key}"},
            {"Content-Type": "application/json; charset=utf-8"},
        ]

        for k in range(1):
            start = time.time()
            result = requests.post(
                upload_url,
                headers={"Content-Type": "application/json; charset=utf-8"},
                json=json_payload,
            )
            json_result = result.json()
            print(json_result)
            # txt = json_result.text
            # print(txt)
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


if __name__ == "__main__":
    ensure_exists("/tmp/snippet")

    # Specify the path to the file you would like to process
    src = "./set-001/test/fragment-003.png"
    # src = "./set-001/test/fragment-002.png"

    if False:
        # process_dir("/home/gbugaj/tmp/")
        process_dir("/home/greg/dataset/assets-private/corr-indexer/validation")

    if True:
        src = "~/tmp/image5839050414130576656-0.tif"
        src = "~/tmp/PID_1925_9289_0_157186264.tif"
        # src = "~/datasets/dataset/medprov/PID/171131488/PID_1971_9380_0_171131488.tif"
        src = os.path.expanduser(src)
        print(src)

        icr_data = process_extract(
            queue_id=default_queue_id, mode="multiline", file_location=src
        )

        print(icr_data)
        image = Image.open(src).convert("RGB")
        visualize_icr(image, icr_data)
