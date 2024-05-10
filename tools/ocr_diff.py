import base64
import glob
import os
from io import BytesIO

import torch as torch
from PIL import Image

from marie.boxes import BoxProcessorUlimDit, PSMode
from marie.constants import __model_path__
from marie.document import TrOcrProcessor
from marie.utils.json import load_json_file, store_json_object

use_cuda = torch.cuda.is_available()


def build_ocr_engines():
    # return None, None, None

    box_processor = BoxProcessorUlimDit(
        models_dir="/mnt/data/marie-ai/model_zoo/unilm/dit/text_detection",
        cuda=use_cuda,
    )

    #
    ocr1_processor = TrOcrProcessor(
        model_name_or_path=os.path.join(
            __model_path__, "trocr", "trocr-large-printed.pt"
        ),
        cuda=use_cuda,
    )

    ocr2_processor = TrOcrProcessor(
        model_name_or_path="/data/models/unilm/trocr/ft_SROIE_LINES_SET27/checkpoint_best.pt",
        cuda=use_cuda,
    )

    return box_processor, ocr1_processor, ocr2_processor


from PIL import Image, ImageDraw, ImageFont


def create_image_with_text(snippet, text1, text2, output_path):
    # Create a new image with space for the text
    width, height = snippet.size
    new_image = Image.new("RGB", (width, height + 60), (255, 255, 255))

    # Paste the snippet onto the new image
    new_image.paste(snippet, (0, 0))

    # Get a drawing context
    draw = ImageDraw.Draw(new_image)

    # Load a font (this font should be available on your system)
    # font = ImageFont.truetype("arial", 15)

    # Load the default font
    font = ImageFont.load_default()

    # Draw the text onto the new image
    draw.text((10, height + 10), text1, font=font, fill=(0, 0, 0))
    draw.text((10, height + 30), text2, font=font, fill=(0, 0, 0))

    # Save the new image
    new_image.save(output_path)


def process_image(img_path, box_processor, ocr1_processor, ocr2_processor):
    image = Image.open(img_path).convert("RGB")
    name = os.path.basename(img_path)
    name = os.path.splitext(name)[0]
    # image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    (
        boxes,
        fragments,
        lines,
        _,
        lines_bboxes,
    ) = box_processor.extract_bounding_boxes("gradio", "field", image, PSMode.SPARSE)

    result1, overlay_image1 = ocr1_processor.recognize(
        "gradio ", "00000", image, boxes, fragments, lines, return_overlay=True
    )

    result2, overlay_image2 = ocr2_processor.recognize(
        "gradio ", "00000", image, boxes, fragments, lines, return_overlay=True
    )

    output_dir = os.path.expanduser("~/tmp/ocr-diffs/v3")
    output_dir_raw = os.path.join(output_dir, "raw")
    output_dir_diff = os.path.join(output_dir, "diff")
    output_dir_ocr1 = os.path.join(output_dir, "ocr1")
    output_dir_ocr2 = os.path.join(output_dir, "ocr2")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir_raw, exist_ok=True)
    os.makedirs(output_dir_diff, exist_ok=True)
    os.makedirs(output_dir_ocr1, exist_ok=True)
    os.makedirs(output_dir_ocr2, exist_ok=True)

    # iterate over both results and save them
    for idx, (
        word1,
        word2,
    ) in enumerate(zip(result1["words"], result2["words"])):

        print("word : ", word1, word2)
        # filter only for words that contain digits
        if not any(char.isdigit() for char in word1["text"]):
            continue

        if word1["text"] != word2["text"]:
            print("DIFFERENT")
            print(word1)
            print(word2)
            print("----")
            conf1 = word1["confidence"]
            conf2 = word2["confidence"]

            mix_word_len = min(len(word1["text"]), len(word2["text"]))
            if mix_word_len < 3:
                print("skipping short word : " + word1["text"] + " " + word2["text"])
                continue

            # clip the image snippet from the original image
            box = word1["box"]
            box = [int(x) for x in box]
            x, y, w, h = box
            # convert form xywh to xyxy
            converted = [
                box[0],
                box[1],
                box[0] + box[2],
                box[1] + box[3],
            ]
            word_img = image.crop(converted)
            word_img.save(os.path.join(output_dir_raw, f"{name}_{idx}_snippet.png"))

            # Convert the image to bytes
            buffered = BytesIO()
            word_img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            # create a json file with the two words and their image snippets
            if conf1 > conf2:
                dump_dir = output_dir_ocr1
            else:
                dump_dir = output_dir_ocr2

            diff_obj = {
                "word1": word1,
                "word2": word2,
                "confidence1": conf1,
                "confidence2": conf2,
                "snippet": img_str,
            }

            word_img.save(os.path.join(dump_dir, f"{name}_{idx}_snippet.png"))
            store_json_object(diff_obj, os.path.join(dump_dir, f"{name}_{idx}.json"))
            create_image_with_text(
                word_img,
                word1["text"],
                word2["text"],
                os.path.join(output_dir_diff, f"{name}_{idx}_diff.png"),
            )


def process_dir(image_dir: str, box_processor, ocr1_processor, ocr2_processor):
    import random

    items = glob.glob(os.path.join(image_dir, "*.*"))
    random.shuffle(items)

    for idx, img_path in enumerate(items):
        try:
            print(img_path)
            process_image(img_path, box_processor, ocr1_processor, ocr2_processor)
        except Exception as e:
            print(e)
            raise e


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    box_processor, ocr1_processor, ocr2_processor = build_ocr_engines()

    process_dir(
        "/home/greg/datasets/funsd_dit/IMAGES/LbxIDImages_boundingBox_6292023",
        box_processor,
        ocr1_processor,
        ocr2_processor,
    )
