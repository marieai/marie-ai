import glob
import io
import json
from pathlib import Path
import cv2
from PIL import Image, ImageDraw, ImageFont

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import torch

import numpy as np
from transformers.utils import check_min_version

from PIL import Image
from transformers import LayoutLMv2Processor, LayoutLMv2FeatureExtractor, LayoutLMv2ForTokenClassification, \
    LayoutLMv2TokenizerFast

# https://programtalk.com/vs4/python/huggingface/transformers/tests/layoutlmv2/test_processor_layoutlmv2.py/
# https://github.com/huggingface/transformers/blob/d3ae2bd3cf9fc1c3c9c9279a8bae740d1fd74f34/tests/layoutlmv2/test_processor_layoutlmv2.py

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
from boxes.box_processor import PSMode
from numpyencoder import NumpyEncoder
from utils.image_utils import viewImage, read_image
from utils.utils import ensure_exists

check_min_version("4.5.0")
logger = logging.getLogger(__name__)

# Calling this from here prevents : "AttributeError: module 'detectron2' has no attribute 'config'"
from detectron2.config import get_cfg

from boxes.craft_box_processor import BoxProcessorCraft
from document.trocr_icr_processor import TrOcrIcrProcessor


def from_json_file(filename):
    with io.open(filename, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)
        return data


def unnormalize_box(bbox, width, height):
    return [
        width * (bbox[0] / 1000),
        height * (bbox[1] / 1000),
        width * (bbox[2] / 1000),
        height * (bbox[3] / 1000),
    ]


def normalize_bbox(bbox, size):
    return [
        int(1000 * bbox[0] / size[0]),
        int(1000 * bbox[1] / size[1]),
        int(1000 * bbox[2] / size[0]),
        int(1000 * bbox[3] / size[1]),
    ]


def iob_to_label(label):
    label = label[2:]
    if not label:
        return "other"
    return label



def obtain_words(src_image):
    image = read_image(src_image)

    work_dir_boxes = ensure_exists("/tmp/boxes")
    work_dir_icr = ensure_exists("/tmp/icr")

    boxp = BoxProcessorCraft(work_dir=work_dir_boxes, models_dir="./model_zoo/craft", cuda=True)
    icrp = TrOcrIcrProcessor(work_dir=work_dir_icr, cuda=True)


    key = "funsd"
    boxes, img_fragments, lines, _ = boxp.extract_bounding_boxes(key, "field", image, PSMode.SPARSE)
    results, overlay_image = icrp.recognize(key, "test", image, boxes, img_fragments, lines)

    print(boxes)
    print(results)
    # change from xywy -> xyxy
    x0 = 0
    y0 = 0

    for word in results["words"]:
        x, y, w, h = word["box"]
        w_box = [x0 + x, y0 + y, x0 + x + w, y0 + y + h]
        word["box"] = w_box

    return results

def main_image(src_image):
    # labels = ["O", "B-HEADER", "I-HEADER", "B-QUESTION", "I-QUESTION", "B-ANSWER", "I-ANSWER"]
    labels = ["O", 'B-MEMBER_NAME', 'I-MEMBER_NAME', 'B-MEMBER_NAME_ANSWER', 'I-MEMBER_NAME_ANSWER', 'B-MEMBER_NUMBER', 'I-MEMBER_NUMBER', 'B-MEMBER_NUMBER_ANSWER', 'I-MEMBER_NUMBER_ANSWER', 'B-PAN', 'I-PAN', 'B-PAN_ANSWER', 'I-PAN_ANSWER', 'B-DOS', 'I-DOS', 'B-DOS_ANSWER', 'I-DOS_ANSWER', 'B-PATIENT_NAME', 'I-PATIENT_NAME', 'B-PATIENT_NAME_ANSWER', 'I-PATIENT_NAME_ANSWER']
    logger.info("Labels : {}", labels)

    id2label = {v: k for v, k in enumerate(labels)}
    label2id = {k: v for v, k in enumerate(labels)}

    # prepare for the model
    # we do not want to use the pytesseract
    # LayoutLMv2FeatureExtractor requires the PyTesseract library but it was not found in your environment. You can install it with pip:
    processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased", revision="no_ocr")

    # Method:2 Create Layout processor with custom future extractor
    # feature_extractor = LayoutLMv2FeatureExtractor(apply_ocr=False)
    # tokenizer = LayoutLMv2TokenizerFast.from_pretrained("microsoft/layoutlmv2-base-uncased")
    # processor = LayoutLMv2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    # Display vocabulary
    # print(tokenizer.get_vocab())

    image = Image.open(src_image).convert("RGB")
    # image.show()

    width, height = image.size

    # Next, let's move everything to the GPU, if it's available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## Need to obtain boxes and OCR for the document    
    from pathlib import Path
    home = str(Path.home())
    ocr_output_dir = os.path.join(home, '.marie')

    print(f"output dir : {ocr_output_dir}")
    filename = src_image.split("/")[-1]
    json_path = os.path.join(home, '.marie', f'{filename}.json')

    if not os.path.exists(json_path):
        raise Exception(f"OCR File not found : {json_path}")

    # results = from_json_file("/tmp/ocr-results.json")
    results = from_json_file(json_path)
    # results = obtain_words(image)
    # visualize_icr(image, results)

    words = []
    boxes = []

    for i, word in enumerate(results["words"]):
        words.append(word["text"].lower())
        # words.append(word["text"])
        box_norm = normalize_bbox(word["box"], (width, height))
        boxes.append(box_norm)

        # This is to prevent following error
        # The expanded size of the tensor (516) must match the existing size (512) at non-singleton dimension 1.
        print(len(boxes))
        if len(boxes) == 512:
            print('Clipping MAX boxes at 512')
            break


    assert len(words) == len(boxes)
    print(words)
    print(boxes)
    print(len(words))

    encoded_inputs = processor(image, words, boxes=boxes, return_tensors="pt")
    expected_keys = ["attention_mask", "bbox", "image", "input_ids", "token_type_ids"]
    actual_keys = sorted(list(encoded_inputs.keys()))

    print("Expected Keys : ", expected_keys)
    print("Actual Keys : ", actual_keys)

    for key in expected_keys:
        print(f"key: {key}")
        print(encoded_inputs[key])

    for k, v in encoded_inputs.items():
        encoded_inputs[k] = v.to(device)

    from pathlib import Path
    home = str(Path.home())
    model_dir = os.path.join(home, './tmp/models/layoutlmv2-finetuned-gb', "checkpoint-11500")
    # model_dir = "/tmp/models/layoutlmv2-finetuned-cord/checkpoint-12000"
    print(f"output dir : {model_dir}")

    # load the fine-tuned model from the hub
    # model = LayoutLMv2ForTokenClassification.from_pretrained("/tmp/models/layoutlmv2-finetuned-cord")
    model = LayoutLMv2ForTokenClassification.from_pretrained(model_dir)
    model.to(device)

    # forward pass
    outputs = model(**encoded_inputs)
    print(outputs.logits.shape)

    logits = outputs.logits
    # probability = softmax(outputs.logits, axis=-1)
    # print(outputs.logits)
    # print('Probabilities : ')
    # print(probability)

    # https://discuss.pytorch.org/t/logits-vs-log-softmax/95979
    import torch.nn.functional as F
    print('Probas from logits:\n', F.softmax(logits, dim=-1))
    print('Log-softmax:\n', F.log_softmax(logits, dim=-1))
    print('Difference between logits and log-softmax:\n', logits - F.log_softmax(logits, dim=-1))
    print('Probas from log-softmax:\n', F.softmax(F.log_softmax(logits, dim=-1), dim=-1))

    # Let's create the true predictions, true labels (in terms of label names) as well as the true boxes.

    predictions = outputs.logits.argmax(-1).squeeze().tolist()
    token_boxes = encoded_inputs.bbox.squeeze().tolist()

    width, height = image.size

    true_predictions = [id2label[prediction] for prediction in predictions]
    true_boxes = [unnormalize_box(box, width, height) for box in token_boxes]

    print(true_predictions)
    print(true_boxes)

    def iob_to_label(label):
        label = label[2:]
        if not label:
            return "other"
        return label

    label2color = {"question": "blue", "answer": "green", "header": "orange", "other": "violet"}

    label2color = {"pan": "blue", "pan_answer": "green",
                   "dos": "orange", "dos_answer": "violet",
                   "member": "blue", "member_answer": "green",
                   "member_number": "blue", "member_number_answer": "green",
                   "member_name": "blue", "member_name_answer": "green",
                   "patient_name": "blue", "patient_name_answer": "green",
                   "other": "red"
                   }

    draw = ImageDraw.Draw(image, 'RGBA')
    font = ImageFont.load_default()

    for prediction, box in zip(true_predictions, true_boxes):
        # don't draw other 
        label = prediction[2:]
        if not label:
            continue
        predicted_label = iob_to_label(prediction).lower()
        draw.rectangle(box, outline=label2color[predicted_label], width=1,  fill=(0, 255, 0, 50))
        draw.text((box[0] + 10, box[1] - 10), text=predicted_label, fill='red', font=font, width=1)
    
    del draw

    # image.show()
    image.save(f"/tmp/tensors/{filename}")


def visualize_icr(image, icr_data):
    viz_img = image.copy()

    draw = ImageDraw.Draw(viz_img)
    font = ImageFont.load_default()

    for i, item in enumerate(icr_data["words"]):
        box = item["box"]
        text = item["text"]
        draw.rectangle(box, outline="red", width=1)
        # draw.text((box[0], box[1]), text=text, fill="blue", font=font, stroke_width=1)

    viz_img.show()


def hash_file(filename):
   """"This function returns the SHA-1 hash
   of the file passed into it"""
   import hashlib

   # make a hash object
   h = hashlib.sha1()

   # open file for reading in binary mode
   with open(filename,'rb') as file:

       # loop till the end of the file
       chunk = 0
       while chunk != b'':
           # read only 1024 bytes at a time
           chunk = file.read(1024)
           h.update(chunk)

   # return the hex representation of digest
   return h.hexdigest()


def ocr_dir(src_dir, filename_filter="*.png"):
    from pathlib import Path
    home = str(Path.home())
    output_dir = os.path.join(home, '.marie')

    print(f"output dir : {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    for idx, _path in enumerate(glob.glob(os.path.join(src_dir, filename_filter))):
        filename = _path.split("/")[-1]
        json_path = os.path.join(home, '.marie', f'{filename}.json')
        print(f" : {_path}")
        print(f" : {filename}")
        print(f" : {json_path}")
        image = Image.open(image_path).convert("RGB")
        # image.show()
        results = obtain_words(image)
        print(results)

        with open(json_path, "w") as json_file:
            json.dump(
                results,
                json_file,
                sort_keys=True,
                separators=(",", ": "),
                ensure_ascii=False,
                indent=4,
                cls=NumpyEncoder,
            )

        visualize_icr(image, results)
        break


if __name__ == "__main__":
    os.putenv("TOKENIZERS_PARALLELISM", "false")
    image_path = "/home/greg/dataset/assets-private/corr-indexer/dataset/train_dataset/images/152606114_2.png"
    image_path = "/home/gbugaj/dataset/private/corr-indexer-converted/dataset/testing_data/images/152658535_2.png"
    image_path = "/home/greg/dataset/assets-private/corr-indexer-converted/dataset/testing_data/images/152658533_2.png"
    
    fname = "152658775_13.png" # F
    fname = "152658533_2.png"  # F
    fname = "152658535_2.png"  # VG
    fname = "152658536_0.png"  # ERROR
    fname = "152658538_2.png"  # P
    fname = "152658540_0.png"  # F
    fname = "152658541_2.png"  # P
    fname = "152658548_2.png"  # P
    fname = "152658549_2.png"  # G
    fname = "152658552_2.png"  # F
    fname = "152658551_4.png"  # F
    fname = "152658547_2.png"  # F
    fname = "152658671_2.png"  # P
    fname = "152658679_0.png"  # F

    fname = "152658538_2.png"
    
    # image_path = f"/home/greg/dataset/assets-private/corr-indexer-converted/dataset/testing_data/images/{fname}"
    # image_path = f"/home/gbugaj/dataset/private/corr-indexer-converted/dataset/testing_data/images/{fname}"
    image_path = f"/home/gbugaj/dataset/private/corr-indexer/testdeck-raw-01/images/corr-indexing/test/{fname}"

    from pathlib import Path
    home = str(Path.home())
    json_path = os.path.join(home, '.marie', f'{fname}.json')
    # json_path = f"/tmp/ocr-results.json"

    print(f'json_path : {json_path}')
    print(f'image_path : {image_path}')

    # ocr_dir("/home/gbugaj/dataset/private/corr-indexer/testdeck-raw-01/images/corr-indexing/test", fname)
    main_image(image_path)

    # message = hash_file(image_path)
    # print(message)
    # ocr_dir("/home/greg/dataset/assets-private/corr-indexer-converted/dataset/testing_data/images")
    # 
 
    if False:
        from pathlib import Path 
        home = str(Path.home())
        marie_home = os.path.join(home, '.marie')
        for idx, _path in enumerate(glob.glob(os.path.join(marie_home, "*"))):
            filename = _path.split("/")[-1].replace(".json", "")
            print(filename)
            # json_path = os.path.join(home, '.marie', f'{filename}.json')
            image_path = f"/home/gbugaj/dataset/private/corr-indexer/testdeck-raw-01/images/corr-indexing/test/{filename}"
            print(image_path)
            try:
                main_image(image_path)
            except Exception as e:
                print(e)

        
    if False:
        image = Image.open(image_path).convert("RGB")
        results = from_json_file(json_path)
        visualize_icr(image, results)

    if False:
        image = Image.open(image_path).convert("RGB")
        image.show()
        results = obtain_words(image)
        x0 = 0
        y0 = 0
        
        for word in results["words"]:
            x, y, w, h = word["box"]
            w_box = [x0 + x, y0 + y, x0 + x + w, y0 + y + h]
            word["box"] = w_box
        print(results)

        json_path = os.path.join("/tmp/ocr-results.json")
        with open(json_path, "w") as json_file:
            json.dump(
                results,
                json_file,
                sort_keys=True,
                separators=(",", ": "),
                ensure_ascii=False,
                indent=4,
                cls=NumpyEncoder,
            )

        visualize_icr(image, results)

# happy gra day love from izabella