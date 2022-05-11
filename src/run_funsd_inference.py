import io
import json

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

    boxp = BoxProcessorCraft(work_dir=work_dir_boxes, models_dir="./model_zoo/craft", cuda=False)
    icrp = TrOcrIcrProcessor(work_dir=work_dir_icr, cuda=False)

    key = "funsd"
    boxes, img_fragments, lines, _ = boxp.extract_bounding_boxes(key, "field", image, PSMode.SPARSE)
    result, overlay_image = icrp.recognize(key, "test", image, boxes, img_fragments, lines)

    print(boxes)
    print(result)

    return result


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
    image.show()

    width, height = image.size

    # Next, let's move everything to the GPU, if it's available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## Need to obtain boxes and OCR for the document

    results = from_json_file("/tmp/ocr-results.json") 
    # results = obtain_words(image)

    words = []
    boxes = []
    for i, word in enumerate(results["words"]):
        words.append(word["text"].lower())
        box_norm = normalize_bbox(word["box"], (width, height))
        boxes.append(box_norm)


    assert len(words) == len(boxes)
    print(words)
    print(boxes)

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

    # load the fine-tuned model from the hub
    # model = LayoutLMv2ForTokenClassification.from_pretrained("nielsr/layoutlmv2-finetuned-funsd")
    # model = LayoutLMv2ForTokenClassification.from_pretrained("/home/gbugaj/dev/unilm/layoutlmft/examples/checkpoints")
    # model = LayoutLMv2ForTokenClassification.from_pretrained("/tmp/models/layoutlmv2-finetuned-cord")
    model = LayoutLMv2ForTokenClassification.from_pretrained("/tmp/models/layoutlmv2-finetuned-cord/checkpoint-12000")

    # model = torch.load("/home/greg/dev/unilm/layoutlmft/examples/tuned/layoutlmv2-finetuned-funsd-torch.pth")
    # model = torch.load("/home/gbugaj/dev/unilm/layoutlmft/examples/tuned/layoutlmv2-finetuned-funsd-torch.pth")
    # model = torch.load("/home/gbugaj/dev/unilm/layoutlmft/examples/tuned/layoutlmv2-finetuned-funsd-torch_epoch_1.pth")

    model.to(device)

    # forward pass
    outputs = model(**encoded_inputs)
    print(outputs.logits.shape)

    
    # Let's create the true predictions, true labels (in terms of label names) as well as the true boxes.

    # predictions
    predictions = outputs.logits.argmax(-1).squeeze().tolist()
    token_boxes = encoded_inputs.bbox.squeeze().tolist()

    width, height = image.size

    true_predictions = [id2label[prediction] for prediction in predictions]
    true_boxes = [unnormalize_box(box, width, height) for box in token_boxes]

    print(true_predictions)
    print(true_boxes)

    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

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

    for prediction, box in zip(true_predictions, true_boxes):
        # don't draw other 
        label = prediction[2:]
        if not label:
            continue

        predicted_label = iob_to_label(prediction).lower()
        draw.rectangle(box, outline=label2color[predicted_label], width=1)
        draw.text((box[0] + 10, box[1] - 10), text=predicted_label, fill=label2color[predicted_label], font=font, width=1)

    image.show()


def visualize_icr(image, icr_data):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for i, item in enumerate(icr_data["words"]):
        box = item["box"]
        text = item["text"]
        draw.rectangle(box, outline="red")
        draw.text((box[0], box[1]), text=text, fill="blue", font=font)

    image.show()


if __name__ == "__main__":
    os.putenv("TOKENIZERS_PARALLELISM", "false")

    image_path = "/home/greg/dataset/assets-private/corr-indexer/dataset/train_dataset/images/152606114_2.png"
    image_path = "/home/gbugaj/dataset/private/corr-indexer-converted/dataset/testing_data/images/152658535_2.png"

    # image_path = "/tmp/snippet/resized_152625510_2.png"
    main_image(image_path)

    if False:
        image = Image.open(image_path).convert("RGB")
        results = from_json_file("/tmp/ocr-results.json")
        visualize_icr(image, results)

    if False:
        image = Image.open(image_path).convert("RGB")
        image.show()

        results = obtain_words(image)
        words = []
        boxes = []
        word_labels = []

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

