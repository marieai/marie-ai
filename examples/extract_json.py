import base64
import os
import time

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont

from marie.utils.utils import ensure_exists

api_base_url = "http://127.0.0.1:5000/api"
# api_base_url = "http://172.83.15.97:6000/api"  # marie-009
# api_base_url = "http://184.105.180.27:6000/api"  # marie-008
# api_base_url = "http://184.105.180.21:6000/api"  # marie-006
# api_base_url = "http://184.105.180.25:6000/api"  # marie-007
# api_base_url = "http://172.83.13.210:6000/api"  # marie-004
# api_base_url = "http://184.105.180.15:6000/api"  # marie-003
api_base_url = "http://184.105.180.6:6000/api"  # marie-0
#

# api_base_url = "http://184.105.180.8:5000/api" # Traefic loadballancer

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
    upload_url = f"{api_base_url}/ner/{queue_id}"

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

        json_payload = {"data": base64_str, "mode": mode, "output": "assets"}
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

    for j, item in enumerate(icr_data):
        lines_bboxes = item["meta"]["line_bboxes"]
        for k, box in enumerate(lines_bboxes):
            print(box)
            x, y, w, h = box
            draw.rectangle(
                [x, y, x + w, y + h],
                outline="#993300",
                fill=(
                    int(np.random.random() * 256),
                    int(np.random.random() * 256),
                    int(np.random.random() * 256),
                    125,
                ),
                width=1,
            )
            #
            # draw.rectangle(
            #     [box[0], box[1], box[0] + box[2], box[1] + box[3]],
            #     outline="#993300",
            #     fill=(0, 180, 0, 125),
            #     width=1,
            # )

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
            draw.text((box[0], box[1]), text=text, fill="blue", font=font, stroke_width=0)

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
    src = "/home/gbugaj/dataset/private/corr-indexer/dataset/training_data/images/152606114_2.png"
    src = "/home/gbugaj/tmp/PID_886_7652_0_157518994.tif"
    src = "/home/gbugaj/tmp/PID_1925_9289_0_157186264.tif"
    # src = "/home/gbugaj/Downloads/tiffs/Pages from 433142 - SVB 06.14.2022 Correspondence 0.pdf - RP-25.tif" # {'error': 'Invalid line number : -1, this looks like a bug/vertical line : [147, 882, 75, 12]'}
    # src = "/home/gbugaj/Downloads/tiffs/Pages from 433142 - SVB 06.14.2022 Correspondence 0.pdf - RP-28.tif"
    icr_data = process_extract(queue_id=default_queue_id, mode="multiline", file_location=src)
    print(icr_data)
    image = Image.open(src).convert("RGB")
    visualize_icr(image, icr_data)


# Model load elapsed: 102.28931069374084
# OCR file  : /home/app-svc/.marie/ocr/0ff60be73d3a16058e62fd859db644eb47039d8a.json
# NER file  : /home/app-svc/.marie/annotation/0ff60be73d3a16058e62fd859db644eb47039d8a.json
