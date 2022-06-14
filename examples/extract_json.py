import base64
import os
import time

import requests
from PIL import Image, ImageDraw, ImageFont

api_base_url = "http://127.0.0.1:5100/api"
api_base_url = "http://65.49.54.125:5100/api"
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

        # modes can be [word, line, sparse]
        # "word" -> will not perform BoundingBox detection and use the size of the image as bbox
        json_payload = {"data": base64_str, "mode": mode}

        print(json_payload)
        # Upload file to api
        print(f"Uploading to marie-ai for processing : {file}")

        for k in range(1):
            start = time.time()
            json_result = requests.post(upload_url, headers=auth_headers, json=json_payload).json()
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

    for i, item in enumerate(icr_data["words"]):
        box = item["box"]
        text = item["text"]
        draw.rectangle(
            [box[0], box[1], box[0] + box[2], box[1] + box[3]], outline="#993300", fill=(0, 180, 0, 125), width=1
        )
        draw.text((box[0], box[1]), text=text, fill="blue", font=font, stroke_width=0)

    viz_img.show()


if __name__ == "__main__":
    # Specify the path to the file you would like to process
    src = "./set-001/test/fragment-003.png"
    # src = "./set-001/test/fragment-002.png"
    # src = "/home/gbugaj/dev/sk-snippet-dump/hashed/AARON_JANES/38585416cc1f22103336f3419390c9ce.tif"
    icr_data = process_extract(queue_id=default_queue_id, mode="sparse", file_location=src)
    print(icr_data)
    image = Image.open(src).convert("RGB")
    visualize_icr(image, icr_data)
