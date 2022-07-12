from builtins import print
import torch
from docarray import DocumentArray
from torch.backends import cudnn

from marie import Executor, requests, __model_path__

from marie.logging.logger import MarieLogger
from marie.utils.utils import ensure_exists
from marie.utils.overlap import find_overlap_horizontal
from marie.utils.overlap import merge_bboxes_as_block

import cv2
from PIL import Image, ImageDraw, ImageFont
import logging
import os
from typing import Optional
import torch

import numpy as np

from PIL import Image
from transformers import (
    LayoutLMv2Processor,
    LayoutLMv2FeatureExtractor,
    LayoutLMv2ForTokenClassification,
    LayoutLMv2TokenizerFast,
)

from transformers.utils import check_min_version

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
from marie.boxes.line_processor import find_line_number
from marie.executor import TextExtractionExecutor
from marie.executor.text_extraction_executor import CoordinateFormat
from marie.utils.docs import (
    docs_from_file,
    array_from_docs,
    docs_from_image,
    load_image,
    __convert_frames,
)

from marie.utils.image_utils import viewImage, read_image, hash_file
from marie.utils.json import store_json_object, load_json_file
from pathlib import Path

# Calling this from here prevents : "AttributeError: module 'detectron2' has no attribute 'config'"
from detectron2.config import get_cfg

check_min_version("4.5.0")
logger = logging.getLogger(__name__)


def get_marie_home():
    return os.path.join(str(Path.home()), ".marie")


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


def obtain_ocr(src_image: str, text_executor: TextExtractionExecutor):
    """
    Obtain OCR words
    """
    docs = docs_from_file(src_image)
    frames = array_from_docs(docs)
    kwa = {"payload": {"output": "json", "mode": "sparse", "format": "xyxy"}}
    results = text_executor.extract(docs, **kwa)

    return results, frames


def create_processor():
    """prepare for the model"""

    # we do not want to use the pytesseract
    # LayoutLMv2FeatureExtractor requires the PyTesseract library but it was not found in your environment. You can install it with pip:
    # processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased", revision="no_ocr")

    # Method:2 Create Layout processor with custom future extractor
    # feature_extractor = LayoutLMv2FeatureExtractor(apply_ocr=False)
    feature_extractor = LayoutLMv2FeatureExtractor(apply_ocr=False)
    tokenizer = LayoutLMv2TokenizerFast.from_pretrained(
        "microsoft/layoutlmv2-large-uncased"
    )
    processor = LayoutLMv2Processor(
        feature_extractor=feature_extractor, tokenizer=tokenizer
    )

    return processor


def create_model_for_token_classification(model_dir: str):
    """
    Create token classification model
    """
    print(f"LayoutLMv2ForTokenClassification dir : {model_dir}")
    model = LayoutLMv2ForTokenClassification.from_pretrained(model_dir)
    # Next, let's move everything to the GPU, if it's available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model, device


def get_label_info():
    labels = [
        "O",
        "B-MEMBER_NAME",
        "I-MEMBER_NAME",
        "B-MEMBER_NAME_ANSWER",
        "I-MEMBER_NAME_ANSWER",
        "B-MEMBER_NUMBER",
        "I-MEMBER_NUMBER",
        "B-MEMBER_NUMBER_ANSWER",
        "I-MEMBER_NUMBER_ANSWER",
        "B-PAN",
        "I-PAN",
        "B-PAN_ANSWER",
        "I-PAN_ANSWER",
        "B-DOS",
        "I-DOS",
        "B-DOS_ANSWER",
        "I-DOS_ANSWER",
        "B-PATIENT_NAME",
        "I-PATIENT_NAME",
        "B-PATIENT_NAME_ANSWER",
        "I-PATIENT_NAME_ANSWER",
        "B-HEADER",
        "I-HEADER",
        "B-DOCUMENT_CONTROL",
        "I-DOCUMENT_CONTROL",
        "B-LETTER_DATE",
        "I-LETTER_DATE",
        "B-PARAGRAPH",
        "I-PARAGRAPH",
        "B-ADDRESS",
        "I-ADDRESS",
        "B-QUESTION",
        "I-QUESTION",
        "B-ANSWER",
        "I-ANSWER",
        "B-PHONE",
        "I-PHONE",
        "B-URL",
        "I-URL",
        "B-GREETING",
        "I-GREETING",
    ]

    logger.info(f"Labels : {labels}")

    id2label = {v: k for v, k in enumerate(labels)}
    label2id = {k: v for v, k in enumerate(labels)}

    return id2label, label2id


def get_label_colors():
    return {
        "pan": "blue",
        "pan_answer": "green",
        "dos": "orange",
        "dos_answer": "violet",
        "member": "blue",
        "member_answer": "green",
        "member_number": "blue",
        "member_number_answer": "green",
        "member_name": "blue",
        "member_name_answer": "green",
        "patient_name": "blue",
        "patient_name_answer": "green",
        "paragraph": "purple",
        "greeting": "blue",
        "address": "orange",
        "question": "blue",
        "answer": "aqua",
        "document_control": "grey",
        "header": "brown",
        "letter_date": "deeppink",
        "url": "darkorange",
        "phone": "darkmagenta",
        "other": "red",
    }


def draw_box(draw, box, text, fill_color, font):
    # in  xywh -> x1y1x2y2
    draw.rectangle(
        [(box[0], box[1]), (box[0] + box[2], box[1] + box[3])],
        outline="red",
        fill=fill_color,
        width=1,
    )
    if text is not None:
        draw.text(
            (box[0] + 10, box[1] - 10),
            text=f"{text}",
            fill="red",
            font=font,
            width=1,
        )


def visualize_prediction(
    output_filename, frame, true_predictions, true_boxes, true_scores
):
    image = frame.copy()
    # https://stackoverflow.com/questions/54165439/what-are-the-exact-color-names-available-in-pils-imagedraw
    label2color = get_label_colors()
    draw = ImageDraw.Draw(image, "RGBA")
    font = get_font(14)

    for prediction, box, score in zip(true_predictions, true_boxes, true_scores):
        # don't draw other
        label = prediction[2:]
        if not label:
            continue

        predicted_label = iob_to_label(prediction).lower()
        draw.rectangle(box, outline=label2color[predicted_label.lower()], width=1)
        draw.text(
            (box[0] + 10, box[1] - 10),
            text=f"{predicted_label} : {score}",
            fill="red",
            font=font,
            width=1,
        )

    # image.show()
    image.save(output_filename)
    del draw


def visualize_extract_kv(output_filename, frame, kv_results):
    """Visualize KV prediction"""
    image = frame.copy()
    draw = ImageDraw.Draw(image, "RGBA")
    font = get_font(10)

    for i, kv in enumerate(kv_results):
        question = kv["value"]["question"]
        answer = kv["value"]["answer"]

        q_box = question["bbox"]
        a_box = answer["bbox"]

        draw_box(
            draw,
            q_box,
            None,
            get_random_color(),
            font,
        )

        draw_box(
            draw,
            a_box,
            None,
            get_random_color(),
            font,
        )

    image.save(output_filename)
    del draw


def get_random_color():
    return (
        np.random.randint(50, 255),
        np.random.randint(50, 255),
        np.random.randint(50, 255),
        70,
    )


def get_font(size):
    try:
        font = ImageFont.truetype(os.path.join("./assets/fonts", "FreeSans.ttf"), size)
    except Exception as ex:
        font = ImageFont.load_default()

    return font


def visualize_icr(frames, results, filename):
    assert len(frames) == len(results)

    for page_idx, (image, result) in enumerate(zip(frames, results)):
        # convert from numpy to PIL
        img = image.copy()
        # we can have frames as both PIL and CV images
        if not isinstance(img, Image.Image):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            viz_img = Image.fromarray(img)
        else:
            viz_img = img

        size = 14
        draw = ImageDraw.Draw(viz_img, "RGBA")
        font = get_font(size)

        words_all = []
        words = np.array(result["words"])
        lines_bboxes = np.array(result["meta"]["lines_bboxes"])

        for i, item in enumerate(words):
            box = item["box"]
            text = f'({i}){item["text"]}'
            words_all.append(text)

            # get text size
            text_size = font.getsize(text)
            button_size = (text_size[0] + 8, text_size[1] + 8)
            # create image with correct size and black background
            button_img = Image.new("RGBA", button_size, color=(150, 255, 150, 150))
            # put text on button with 10px margins
            button_draw = ImageDraw.Draw(button_img, "RGBA")
            button_draw.text(
                (4, 4), text=text, font=font, stroke_width=0, fill=(0, 0, 0, 0), width=1
            )
            # draw.rectangle(box, outline="red", width=1)
            # draw.text((box[0], box[1]), text=text, fill="blue", font=font, stroke_width=0)
            # put button on source image in position (0, 0)
            viz_img.paste(button_img, (box[0], box[1]))

        for i, box in enumerate(lines_bboxes):
            draw_box(
                draw,
                box,
                None,  # f"{q_text} : {q_confidence}",
                get_random_color(),
                font,
            )

        if filename is None:
            viz_img.save(f"/tmp/tensors/visualize_icr_{page_idx}.png")
        else:
            viz_img.save(f"/tmp/tensors/viz_{filename}_{page_idx}.png")

        del viz_img
        st = " ".join(words_all)
        print(st)

    # viz_img.show()


def get_ocr_line_bbox(bbox, frame, text_executor):
    box = np.array(bbox).astype(np.int32)
    x, y, w, h = box
    img = frame
    if isinstance(frame, Image.Image):
        img = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)

    snippet = img[y : y + h, x : x + w :]
    docs = docs_from_image(snippet)
    kwa = {"payload": {"output": "json", "mode": "raw_line"}}
    results = text_executor.extract(docs, **kwa)

    if len(results) > 0:
        words = results[0]["words"]
        if len(words) > 0:
            word = words[0]
            return word["text"], word["confidence"]
    return "", 0.0


def main_image(
    src_image: str,
    model,
    device,
    text_executor: Optional[TextExtractionExecutor] = None,
):
    if not os.path.exists(src_image):
        print(f"File not found : {src_image}")
        return
    # Obtain OCR results
    file_hash = hash_file(src_image)
    root_dir = get_marie_home()

    ensure_exists(os.path.join(root_dir, "ocr"))
    ensure_exists(os.path.join(root_dir, "annotation"))

    ocr_json_path = os.path.join(root_dir, "ocr", f"{file_hash}.json")
    annotation_json_path = os.path.join(root_dir, "annotation", f"{file_hash}.json")

    print(f"Root      : {root_dir}")
    print(f"SrcImg    : {src_image}")
    print(f"Hash      : {file_hash}")
    print(f"OCR file  : {ocr_json_path}")
    print(f"NER file  : {annotation_json_path}")

    if not os.path.exists(ocr_json_path) and text_executor is None:
        raise Exception(f"OCR File not found : {ocr_json_path}")

    loaded, frames = load_image(src_image, img_format="pil")
    if not loaded:
        raise Exception(f"Unable to load image file: {src_image}")

    if not os.path.exists(ocr_json_path):
        results, frames = obtain_ocr(src_image, text_executor)
        # convert CV frames to PIL frame
        frames = __convert_frames(frames, img_format="pil")
        store_json_object(results, ocr_json_path)

    results = load_json_file(ocr_json_path)
    visualize_icr(frames, results, file_hash)

    assert len(results) == len(frames)
    annotations = []
    processor = create_processor()
    id2label, label2id = get_label_info()

    for k, (result, image) in enumerate(zip(results, frames)):
        if not isinstance(image, Image.Image):
            raise "Frame should have been an PIL.Image instance"

        # image.show()
        size = image.size
        width = image.size[0]
        height = image.size[1]
        words = []
        boxes = []

        for i, word in enumerate(result["words"]):
            words.append(word["text"].lower())
            box_norm = normalize_bbox(word["box"], (width, height))
            boxes.append(box_norm)

            # This is to prevent following error
            # The expanded size of the tensor (516) must match the existing size (512) at non-singleton dimension 1.
            # print(len(boxes))
            if len(boxes) == 512:
                print("Clipping MAX boxes at 512")
                break

        assert len(words) == len(boxes)

        encoded_inputs = processor(
            image,
            words,
            boxes=boxes,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Debug tensor info
        if False:
            img_tensor = encoded_inputs["image"]
            img = Image.fromarray(
                (img_tensor[0].cpu()).numpy().astype(np.uint8).transpose(1, 2, 0)
            )
            img.save(f"/tmp/tensors/tensor_{file_hash}_{k}.png")

        for ek, ev in encoded_inputs.items():
            encoded_inputs[ek] = ev.to(device)

        # forward pass
        outputs = model(**encoded_inputs)

        # Let's create the true predictions, true labels (in terms of label names) as well as the true boxes.
        # outputs.logits returns TokenClassifierOutput
        # logits are non-normalized probabilities
        # we are going to apply softmax to normalize them for each class

        normalized_logits = outputs.logits.softmax(dim=-1).squeeze().tolist()
        predictions = outputs.logits.argmax(-1).squeeze().tolist()
        token_boxes = encoded_inputs.bbox.squeeze().tolist()

        # get predictions
        true_predictions = [id2label[prediction] for prediction in predictions]
        true_boxes = [unnormalize_box(box, width, height) for box in token_boxes]
        true_scores = [
            round(normalized_logits[i][val], 6) for i, val in enumerate(predictions)
        ]

        # show detail scores
        if False:
            for i, val in enumerate(predictions):
                tp = true_predictions[i]
                score = normalized_logits[i][val]
                print(f" >> {tp} : {score}")

        annotation = {
            "meta": {"imageSize": {"width": width, "height": height}, "page": k},
            "predictions": true_predictions,
            "boxes": true_boxes,
            "scores": true_scores,
        }

        output_filename = f"/tmp/tensors/prediction_{file_hash}_{k}.png"
        visualize_prediction(
            output_filename, image, true_predictions, true_boxes, true_scores
        )

        annotations.append(annotation)
    store_json_object(annotations, annotation_json_path)
    return annotations


def aggregate_results(
    src_image: str, text_executor: Optional[TextExtractionExecutor] = None
):
    if not os.path.exists(src_image):
        raise FileNotFoundError(src_image)

    # Obtain OCR results
    file_hash = hash_file(src_image)
    root_dir = get_marie_home()
    ocr_json_path = os.path.join(root_dir, "ocr", f"{file_hash}.json")
    annotation_json_path = os.path.join(root_dir, "annotation", f"{file_hash}.json")

    print(f"OCR file  : {ocr_json_path}")
    print(f"NER file  : {annotation_json_path}")

    if not os.path.exists(ocr_json_path):
        raise FileNotFoundError(ocr_json_path)

    if not os.path.exists(ocr_json_path):
        raise FileNotFoundError(annotation_json_path)

    loaded, frames = load_image(src_image, img_format="pil")
    if not loaded:
        raise Exception(f"Unable to load image file: {src_image}")

    ocr_results = load_json_file(ocr_json_path)
    annotation_results = load_json_file(annotation_json_path)
    assert len(annotation_results) == len(ocr_results) == len(frames)

    # need to normalize all data from XYXY to XYWH as the NER process required XYXY and assets were saved XYXY format
    logger.info("Changing coordinate format from xyxy->xyhw")
    for data in ocr_results:
        for word in data["words"]:
            word["box"] = CoordinateFormat.convert(
                word["box"], CoordinateFormat.XYXY, CoordinateFormat.XYWH
            )

    for data in annotation_results:
        for i, box in enumerate(data["boxes"]):
            box = CoordinateFormat.convert(
                box, CoordinateFormat.XYXY, CoordinateFormat.XYWH
            )
            data["boxes"][i] = box
            # print(f" {i} : {box}")

    aggregated_kv = []
    print(f"frames len = : {len(frames)}")
    for i, (ocr, ann, frame) in enumerate(zip(ocr_results, annotation_results, frames)):
        print(f"Processing page # {i}")
        logger.info(f"Processing page # {i}")
        # lines and boxes are already in the right reading order TOP->BOTTOM, LEFT-TO-RIGHT so no need to sort
        lines_bboxes = np.array(ocr["meta"]["lines_bboxes"])
        true_predictions = ann["predictions"]
        true_boxes = ann["boxes"]
        true_scores = ann["scores"]

        viz_img = frame.copy()
        draw = ImageDraw.Draw(viz_img, "RGBA")
        font = get_font(14)
        # aggregate boxes into their lines
        groups = {}
        for j, (prediction, pred_box, pred_score) in enumerate(
            zip(true_predictions, true_boxes, true_scores)
        ):
            # discard 'O' other
            label = prediction[2:]
            if not label:
                continue
            # two labels that need to be removed [0.0, 0.0, 0.0, 0.0]  [2578.0, 3 3292.0, 0.0, 0.0]
            if pred_box == [0.0, 0.0, 0.0, 0.0] or pred_box[2] == 0 or pred_box[3] == 0:
                continue
            line_number = find_line_number(lines_bboxes, pred_box)
            if line_number not in groups:
                groups[line_number] = []
            groups[line_number].append(j)

        # aggregate boxes into key/value pairs via simple state machine for each line
        aggregated_keys = {}

        for line_idx, line_box in enumerate(lines_bboxes):
            if line_idx not in groups:
                logger.debug(f"Line does not have any groups : {line_idx} : {line_box}")
                continue

            # Debug line overlay
            if False:
                xy = [
                    (line_box[0], line_box[1]),
                    (line_box[0] + line_box[2], line_box[1] + line_box[3]),
                ]
                draw.rectangle(
                    xy,
                    outline="red",
                    fill=get_random_color(),
                    width=1,
                )

            prediction_indexes = np.array(groups[line_idx])

            expected_keys = [
                "PAN",
                "PAN_ANSWER",
                "PATIENT_NAME",
                "PATIENT_NAME_ANSWER",
                "DOS",
                "DOS_ANSWER",
                "MEMBER_NAME",
                "MEMBER_NAME_ANSWER",
                "MEMBER_NUMBER",
                "MEMBER_NUMBER_ANSWER",
                "QUESTION",
                "ANSWER",  # Only collect ANSWERs for now
                # "LETTER_DATE",
                # "PHONE",
                # "URL"
                # "ADDRESS",
            ]

            # expected_keys = ["PAN", "PAN_ANSWER"]
            line_aggregator = []

            color_map = {"ADDRESS": get_random_color()}
            # color_map = {"ADDRESS": get_random_color()}

            for key in expected_keys:
                aggregated = []
                skip_to = -1
                for m in range(0, len(prediction_indexes)):
                    if skip_to != -1 and m <= skip_to:
                        continue
                    pred_idx = prediction_indexes[m]
                    prediction = true_predictions[pred_idx]
                    label = prediction[2:]
                    aggregator = []

                    if label == key:
                        for n in range(m, len(prediction_indexes)):
                            pred_idx = prediction_indexes[n]
                            prediction = true_predictions[pred_idx]
                            label = prediction[2:]
                            if label != key:
                                break
                            aggregator.append(pred_idx)
                            skip_to = n

                    if len(aggregator) > 0:
                        aggregated.append(aggregator)

                if len(aggregated) > 0:
                    line_aggregator.append({"key": key, "groups": aggregated})

            true_predictions = np.array(true_predictions)
            true_boxes = np.array(true_boxes)
            true_scores = np.array(true_scores)

            for line_agg in line_aggregator:
                field = line_agg["key"]
                group_indexes = line_agg["groups"]

                for group_index in group_indexes:
                    bboxes = true_boxes[group_index]
                    scores = true_scores[group_index]
                    group_score = round(np.average(scores), 6)
                    # create a bounding box around our blocks which could be possibly overlapping or being split
                    group_bbox = merge_bboxes_as_block(bboxes)

                    key_result = {
                        "line": line_idx,
                        "key": field,
                        "bbox": group_bbox,
                        "score": group_score,
                    }

                    if line_idx not in aggregated_keys:
                        aggregated_keys[line_idx] = []
                    aggregated_keys[line_idx].append(key_result)

                    color = (
                        color_map[field] if field in color_map else get_random_color()
                    )

                    draw_box(
                        draw,
                        group_bbox,
                        None,
                        color,
                        font,
                    )

        # check if we have possible overlaps when there is a mislabeled token, this could be a flag
        # B-PAN I-PAN I-PAN B-PAN-ANS I-PAN

        for key in expected_keys:
            for ag_key in aggregated_keys.keys():
                row_items = aggregated_keys[ag_key]
                bboxes = [row["bbox"] for row in row_items if row["key"] == key]
                visited = [False for _ in range(0, len(bboxes))]
                to_merge = {}

                for idx in range(0, len(bboxes)):
                    if visited[idx]:
                        continue
                    visited[idx] = True
                    box = bboxes[idx]
                    overlaps, indexes, scores = find_overlap_horizontal(box, bboxes)
                    to_merge[ag_key] = [idx]

                    for _, overlap_idx in zip(overlaps, indexes):
                        visited[overlap_idx] = True
                        to_merge[ag_key].append(overlap_idx)

                for _k, idxs in to_merge.items():
                    items = aggregated_keys[_k]
                    items = np.array(items)
                    # there is nothing to merge, except the original block
                    if len(idxs) == 1:
                        continue

                    idxs = np.array(idxs)
                    picks = items[idxs]
                    remaining = np.delete(items, idxs)

                    score_avg = round(np.average([item["score"] for item in picks]), 6)
                    block = merge_bboxes_as_block([item["bbox"] for item in picks])

                    new_item = picks[0]
                    new_item["score"] = score_avg
                    new_item["bbox"] = block

                    aggregated_keys[_k] = np.concatenate(([new_item], remaining))

        expected_pair = [
            ["PAN", ["PAN_ANSWER", "ANSWER"]],
            ["PATIENT_NAME", ["PATIENT_NAME_ANSWER", "ANSWER"]],
            ["DOS", ["DOS_ANSWER", "ANSWER"]],
            ["MEMBER_NAME", ["MEMBER_NAME_ANSWER", "ANSWER"]],
            ["MEMBER_NUMBER", ["MEMBER_NUMBER_ANSWER", "ANSWER"]],
            ["QUESTION", ["ANSWER"]],
        ]

        for pair in expected_pair:
            expected_question = pair[0]
            expected_answer = pair[1]

            for k in aggregated_keys.keys():
                ner_keys = aggregated_keys[k]
                found_question = None
                found_answer = None

                for ner_key in ner_keys:
                    key = ner_key["key"]
                    if expected_question == key:
                        found_question = ner_key
                        continue
                    # find the first match
                    if found_question is not None and found_answer is None:
                        # find the first match
                        for exp_key in expected_answer:
                            if key in exp_key:
                                found_answer = ner_key
                                break

                        if found_answer is not None:
                            bbox_q = found_question["bbox"]
                            bbox_a = found_answer["bbox"]

                            if bbox_a[0] < bbox_q[0]:
                                logger.warning("Answer is not on the right of question")
                                continue

                            category = found_question["key"]
                            kv_result = {
                                "page": i,
                                "category": category,
                                "value": {
                                    "question": found_question,
                                    "answer": found_answer,
                                },
                            }

                            aggregated_kv.append(kv_result)

        viz_img.save(f"/tmp/tensors/extract_{file_hash}_{i}.png")

    # Decorate our answers with proper TEXT
    kv_indexed = {}

    for agg_result in aggregated_kv:
        page_index = int(agg_result["page"])
        frame = frames[page_index]
        question = agg_result["value"]["question"]
        answer = agg_result["value"]["answer"]

        w1, c1 = get_ocr_line_bbox(question["bbox"], frame, text_executor)
        w2, c2 = get_ocr_line_bbox(answer["bbox"], frame, text_executor)

        question["text"] = {"text": w1, "confidence": c1}
        answer["text"] = {"text": w2, "confidence": c2}

        if page_index not in kv_indexed:
            kv_indexed[page_index] = []

        kv_indexed[page_index].append(agg_result)

    # visualize results per page
    if True:
        for k in range(0, len(frames)):
            output_filename = f"/tmp/tensors/kv_{file_hash}_{k}.png"
            items = [row for row in aggregated_kv if int(row["page"]) == k]
            visualize_extract_kv(output_filename, frames[k], items)

    logger.info(f"aggregated_kv : {aggregated_kv}")
    return aggregated_kv


class NerExtractionExecutor(Executor):
    """
    Executor for extracting text.
    Text extraction can either be executed out over the entire image or over selected regions of interests (ROIs)
    aka bounding boxes.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.show_error = True  # show prediction errors
        self.logger = MarieLogger(
            getattr(self.metas, "name", self.__class__.__name__)
        ).logger

        self.logger.info("NER Extraction Executor")
        # sometimes we have CUDA/GPU support but want to only use CPU
        has_cuda = torch.cuda.is_available()
        if os.environ.get("MARIE_DISABLE_CUDA"):
            has_cuda = False

        if has_cuda:
            cudnn.benchmark = False
            cudnn.deterministic = False
        # ~/dev/marie-ai/model_zoo/ner-rms-corr/checkpoints-tuned-pan
        ensure_exists("/tmp/tensors")
        models_dir: str = os.path.join(
            __model_path__, "ner-rms-corr", "fp16-56k-checkpoint-8500"
        )
        models_dir: str = os.path.join(
            __model_path__, "ner-rms-corr", "checkpoint-best"
        )
        # models_dir: str = os.path.join(__model_path__, "ner-rms-corr", "checkpoints-tuned-pan", "checkpoint-3500")

        self.model, self.device = create_model_for_token_classification(models_dir)
        self.text_executor = TextExtractionExecutor()

    def info(self, **kwargs):
        logger.info(f"Self : {self}")
        return {"index": "ner-complete"}

    # @requests()
    def extract(self, docs: Optional[DocumentArray] = None, **kwargs):
        """
        Args:
            docs : Documents to process, they will be none for now
        """

        queue_id: str = kwargs.get("queue_id", "0000-0000-0000-0000")
        checksum: str = kwargs.get("checksum", "0000-0000-0000-0000")
        image_src: str = kwargs.get("img_path", None)

        for key, value in kwargs.items():
            print("The value of {} is {}".format(key, value))

        root_dir = f"/tmp/ner/{queue_id}/{checksum}"
        ensure_exists(root_dir)

        main_image(image_src, self.model, self.device, self.text_executor)
        ner_results = aggregate_results(image_src, self.text_executor)

        return ner_results
