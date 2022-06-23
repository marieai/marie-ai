import base64
import os
import time

import requests
from PIL import Image, ImageDraw, ImageFont

from marie.utils.utils import ensure_exists

api_base_url = "http://127.0.0.1:5000/api"
# api_base_url = "http://65.49.54.125:5100/api"
# api_base_url = "http://192.168.1.12:5000/api"
# api_base_url = "http://184.105.180.8:5000/api"

# api_base_url = "http://184.105.180.28:6003/api"
# api_base_url = "http://184.105.180.40:6003/api"
# api_base_url = "http://184.105.180.39:6003/api"
# api_base_url = "http://184.105.180.27:6003/api"

# api_base_url = "http://184.105.180.46:6000/api"
# api_base_url = "http://184.105.180.47:6000/api"
# api_base_url = "http://184.105.180.41:6001/api" ## A5000
# api_base_url = "http://184.105.180.51:6000/api"
# api_base_url = "http://172.83.13.224:6000/api"

default_queue_id = "0000-0000-0000-0000"
api_key = "MY_API_KEY"
auth_headers = {"Authorization": f"Bearer {api_key}"}


def online(api_ulr) -> bool:
    import requests

    r = requests.head(api_ulr)
    # The 308 (Permanent Redirect)
    return r.status_code == 200 or r.status_code == 308


def process_extract(queue_id: str, mode: str, file_location: str) -> str:
    if not os.path.exists(file_location):
        raise Exception(f"File not found : {file_location}")
    upload_url = f"{api_base_url}/extract/{queue_id}"

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

        json_payload = {"data": base64_str, "mode": mode, "output": "json"}
        # print(json_payload)
        # Upload file to api
        print(f"Uploading to marie-ai for processing : {file}")

        for k in range(1):
            start = time.time()
            result = requests.post(upload_url, headers=auth_headers, json=json_payload)
            json_result = result.json()
            print(json_result)
            # txt = json_result.text
            # print(txt)
            delta = time.time() - start
            print(f"Request time : {delta}")

        return json_result


def visualize_icr(image, icr_data):
    viz_img = image.copy()
    size = 18
    draw = ImageDraw.Draw(viz_img, "RGBA")
    try:
        font = ImageFont.truetype(os.path.join("./assets/fonts", "FreeMono.ttf"), size)
    except Exception as ex:
        print(ex)
        font = ImageFont.load_default()

    print(f"pages = {len(icr_data)}")

    for i, icr_page in enumerate(icr_data):
        for j, item in enumerate(icr_page["words"]):
            box = item["box"]
            text = item["text"]
            line = item["line"]
            text = f"{i} : {line} - {text} "

            draw.rectangle(
                [box[0], box[1], box[0] + box[2], box[1] + box[3]],
                outline="#993300",
                fill=(0, 180, 0, 125),
                width=1,
            )
            draw.text(
                (box[0], box[1]), text=text, fill="blue", font=font, stroke_width=0
            )

        viz_img.show()
        viz_img.save("/tmp/snippet/extract.png")


if __name__ == "__main__":
    ensure_exists("/tmp/snippet")

    # Specify the path to the file you would like to process
    src = "./set-001/test/fragment-003.png"
    # src = "./set-001/test/fragment-002.png"
    # src = "/home/gbugaj/dev/sk-snippet-dump/hashed/AARON_JANES/38585416cc1f22103336f3419390c9ce.tif"
    # src = "/home/greg/corr-indexer/testdeck-raw-01/images/corr-indexing/test/152658540_0.png"
    # src = "/home/greg/dataset/medprov/PID/150300431/PID_576_7188_0_150300431.tif"
    # src = "/home/gbugaj/dataset/private/corr-indexer/dataset/training_data/images/152611424_1.png"
    src = "/home/greg/dataset/assets-private/corr-indexer/dataset/training_data/images/152624545_1.png"
    icr_data = process_extract(
        queue_id=default_queue_id, mode="multiline", file_location=src
    )
    print(icr_data)
    image = Image.open(src).convert("RGB")
    visualize_icr(image, icr_data)
