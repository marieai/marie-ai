import os
import time
import warnings

import cv2
import torch
from builtins import print

from docarray import DocumentArray
from torch.backends import cudnn
import torch.nn.functional as nn

from marie import Executor, requests, __model_path__, __marie_home__
from marie.executor.ner.utils import (
    normalize_bbox,
    unnormalize_box,
    iob_to_label,
    get_font,
    get_random_color,
    draw_box,
    visualize_icr,
    visualize_prediction,
    visualize_extract_kv,
)

from marie.logging.logger import MarieLogger
from marie.timer import Timer
from marie.utils.utils import ensure_exists
from marie.utils.overlap import find_overlap_horizontal
from marie.utils.overlap import merge_bboxes_as_block

from PIL import Image, ImageDraw, ImageFont
import logging
from typing import Optional, List, Any, Tuple, Dict, Union

import numpy as np

from PIL import Image
from transformers import AutoModelForTokenClassification, AutoProcessor

from transformers import (
    LayoutLMv3Processor,
    LayoutLMv3FeatureExtractor,
    LayoutLMv3ForTokenClassification,
    LayoutLMv3TokenizerFast,
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
    convert_frames,
)

from marie.utils.image_utils import viewImage, read_image, hash_file
from marie.utils.json import store_json_object, load_json_file
from pathlib import Path

# Calling this from here prevents : "AttributeError: module 'detectron2' has no attribute 'config'"
from detectron2.config import get_cfg

check_min_version("4.5.0")
logger = logging.getLogger(__name__)


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
    # Method:2 Create Layout processor with custom future extractor
    # Max model size is 512, so we will need to handle any documents larger than that
    feature_extractor = LayoutLMv3FeatureExtractor(apply_ocr=False)
    tokenizer = LayoutLMv3TokenizerFast.from_pretrained(
        "microsoft/layoutlmv3-large"
        # only_label_first_subword = True
    )
    processor = LayoutLMv3Processor(
        feature_extractor=feature_extractor, tokenizer=tokenizer
    )

    return processor


def load_model(model_dir: str, fp16: bool, device):
    """
    Create token classification model
    """
    print(f"TokenClassification dir : {model_dir}")
    labels, _, _ = get_label_info()
    model = AutoModelForTokenClassification.from_pretrained(
        model_dir, num_labels=len(labels)
    )

    model.to(device)
    return model


def get_label_info():
    labels = [
        "O",
        "B-MEMBER_NAME",
        "I-MEMBER_NAME",
        "B-MEMBER_NUMBER",
        "I-MEMBER_NUMBER",
        "B-PAN",
        "I-PAN",
        "B-PATIENT_NAME",
        "I-PATIENT_NAME",
        "B-DOS",
        "I-DOS",
        "B-DOS_ANSWER",
        "I-DOS_ANSWER",
        "B-PATIENT_NAME_ANSWER",
        "I-PATIENT_NAME_ANSWER",
        "B-MEMBER_NAME_ANSWER",
        "I-MEMBER_NAME_ANSWER",
        "B-MEMBER_NUMBER_ANSWER",
        "I-MEMBER_NUMBER_ANSWER",
        "B-PAN_ANSWER",
        "I-PAN_ANSWER",
        "B-ADDRESS",
        "I-ADDRESS",
        "B-GREETING",
        "I-GREETING",
        "B-HEADER",
        "I-HEADER",
        "B-LETTER_DATE",
        "I-LETTER_DATE",
        "B-PARAGRAPH",
        "I-PARAGRAPH",
        "B-QUESTION",
        "I-QUESTION",
        "B-ANSWER",
        "I-ANSWER",
        "B-DOCUMENT_CONTROL",
        "I-DOCUMENT_CONTROL",
        "B-PHONE",
        "I-PHONE",
        "B-URL",
        "I-URL",
        "B-CLAIM_NUMBER",
        "I-CLAIM_NUMBER",
        "B-CLAIM_NUMBER_ANSWER",
        "I-CLAIM_NUMBER_ANSWER",
        "B-BIRTHDATE",
        "I-BIRTHDATE",
        "B-BIRTHDATE_ANSWER",
        "I-BIRTHDATE_ANSWER",
        "B-BILLED_AMT",
        "I-BILLED_AMT",
        "B-BILLED_AMT_ANSWER",
        "I-BILLED_AMT_ANSWER",
        "B-PAID_AMT",
        "I-PAID_AMT",
        "B-PAID_AMT_ANSWER",
        "I-PAID_AMT_ANSWER",
        "B-CHECK_AMT",
        "I-CHECK_AMT",
        "B-CHECK_AMT_ANSWER",
        "I-CHECK_AMT_ANSWER",
        "B-CHECK_NUMBER",
        "I-CHECK_NUMBER",
        "B-CHECK_NUMBER_ANSWER",
        "I-CHECK_NUMBER_ANSWER",
    ]

    logger.info(f"Labels : {labels}")

    id2label = {v: k for v, k in enumerate(labels)}
    label2id = {k: v for v, k in enumerate(labels)}

    return labels, id2label, label2id


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
        "claim_number": "darkmagenta",
        "claim_number_answer": "green",
        "birthdate": "green",
        "birthdate_answer": "red",
        "billed_amt": "green",
        "billed_amt_answer": "orange",
        "paid_amt": "green",
        "paid_amt_answer": "blue",
        "check_amt": "orange",
        "check_amt_answer": "darkmagenta",
        "check_number": "orange",
        "check_number_answer": "blue",
    }


@Timer(text="OCR Line in {:.4f} seconds")
def get_ocr_line_bbox(bbox, frame, text_executor):
    show_time = True
    t0 = time.time()

    box = np.array(bbox).astype(np.int32)
    x, y, w, h = box
    img = frame
    if isinstance(frame, Image.Image):
        img = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)

    snippet = img[y : y + h, x : x + w :]
    docs = docs_from_image(snippet)
    kwa = {"payload": {"output": "json", "mode": "raw_line"}}
    results = text_executor.extract(docs, **kwa)

    t1 = time.time()
    if show_time:
        print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    if len(results) > 0:
        words = results[0]["words"]
        if len(words) > 0:
            word = words[0]
            return word["text"], word["confidence"]
    return "", 0.0


class NerExtractionExecutor(Executor):
    """
    Executor for extracting text.
    Text extraction can either be executed out over the entire image or over selected regions of interests (ROIs)
    aka bounding boxes.
    """

    def __init__(
        self, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs
    ):
        super().__init__(**kwargs)
        self.show_error = True  # show prediction errors
        self.logger = MarieLogger(
            getattr(self.metas, "name", self.__class__.__name__)
        ).logger

        self.logger.info(f"NER Extraction Executor : {pretrained_model_name_or_path}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # sometimes we have CUDA/GPU support but want to only use CPU
        use_cuda = torch.cuda.is_available()
        if os.environ.get("MARIE_DISABLE_CUDA"):
            use_cuda = False
            self.device = "cpu"

        if use_cuda:
            try:
                from torch._C import _cudnn

                # It seems good practice to turn off cudnn.benchmark when turning on cudnn.deterministic
                cudnn.benchmark = True
                cudnn.deterministic = False
            except ImportError:
                pass

        ensure_exists("/tmp/tensors")
        ensure_exists("/tmp/tensors/json")

        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        init_configuration = {}

        if os.path.isfile(pretrained_model_name_or_path):
            warnings.warn("Expected model directory")

        self.debug_visuals = False
        self.model = load_model(pretrained_model_name_or_path, True, self.device)
        self.processor = create_processor()
        self.text_executor: Optional[TextExtractionExecutor] = TextExtractionExecutor()

    def info(self, **kwargs):
        logger.info(f"Self : {self}")
        return {"index": "ner-complete"}

    def _filter(
        self, values: List[Any], probabilities: List[float], threshold: float
    ) -> List[Any]:
        return [
            value for probs, value in zip(probabilities, values) if probs >= threshold
        ]

    def inference(
        self,
        image: Any,
        words: List[Any],
        boxes: List[Any],
        labels: List[str],
        threshold: float,
    ) -> Tuple[List, List, List]:
        """Run Inference on the model with given processor"""
        logger.info(f"Performing inference")
        model = self.model
        processor = self.processor
        device = self.device
        id2label = {v: k for v, k in enumerate(labels)}

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        logger.info(
            f"Tokenizer parallelism: {os.environ.get('TOKENIZERS_PARALLELISM', 'true')}"
        )

        # image = # Image.open(eg["path"]).convert("RGB")
        width, height = image.size
        # https://huggingface.co/docs/transformers/model_doc/layoutlmv2#transformers.LayoutLMv2ForTokenClassification
        # Encode the image

        encoding = processor(
            # fmt: off
            image,
            words,
            boxes=boxes,
            truncation=True,
            return_offsets_mapping=True,
            padding="max_length",
            return_tensors="pt"
            # fmt: on
        )
        offset_mapping = encoding.pop("offset_mapping")

        # Debug tensor info
        if False:
            # img_tensor = encoded_inputs["image"] # v2
            img_tensor = encoding["pixel_values"]  # v3
            img = Image.fromarray(
                (img_tensor[0].cpu()).numpy().astype(np.uint8).transpose(1, 2, 0)
            )
            # img.save(f"/tmp/tensors/tensor_{file_hash}_{frame_idx}.png")

        # ensure proper device placement
        for ek, ev in encoding.items():
            encoding[ek] = ev.to(device)

        # Perform forward pass
        with torch.no_grad():
            outputs = model(**encoding)
            # Get the predictions and probabilities
            probs = (
                nn.softmax(outputs.logits.squeeze(), dim=1).max(dim=1).values.tolist()
            )
            _predictions = outputs.logits.argmax(-1).squeeze().tolist()
            _token_boxes = encoding.bbox.squeeze().tolist()
            normalized_logits = outputs.logits.softmax(dim=-1).squeeze().tolist()

        # TODO : Filer the results
        # Filter the predictions and bounding boxes based on a threshold
        # predictions = _filter(_predictions, probs, threshold)
        # token_boxes = _filter(_token_boxes, probs, threshold)
        predictions = _predictions
        token_boxes = _token_boxes

        # Only keep non-subword predictions
        is_subword = np.array(offset_mapping.squeeze().tolist())[:, 0] != 0
        true_predictions = [
            id2label[pred]
            for idx, pred in enumerate(predictions)
            if not is_subword[idx]
        ]
        true_boxes = [
            unnormalize_box(box, width, height)
            for idx, box in enumerate(token_boxes)
            if not is_subword[idx]
        ]

        true_scores = [
            round(normalized_logits[idx][val], 6)
            for idx, val in enumerate(predictions)
            if not is_subword[idx]
        ]

        all_predictions = []
        all_boxes = []
        all_scores = []

        all_predictions.append(true_predictions)
        all_boxes.append(true_boxes)
        all_scores.append(true_scores)

        assert len(true_predictions) == len(true_boxes) == len(true_scores)
        return all_predictions, all_boxes, all_scores

    def postprocess(self, frames, annotation_results, ocr_results, file_hash):
        """Post-process extracted data"""

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

        aggregated_kv = []
        aggregated_meta = []

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
            "LETTER_DATE",
            "PHONE",
            "URL",
            "CLAIM_NUMBER",
            "CLAIM_NUMBER_ANSWER",
            "BIRTHDATE",
            "BIRTHDATE_ANSWER",
            "BILLED_AMT",
            "BILLED_AMT_ANSWER",
            "PAID_AMT",
            "PAID_AMT_ANSWER",
            # "ADDRESS",
        ]

        # expected_keys = ["PAN", "PAN_ANSWER"]

        # expected KV pairs
        expected_pair = [
            ["PAN", ["PAN_ANSWER", "ANSWER"]],
            ["CLAIM_NUMBER", ["CLAIM_NUMBER_ANSWER", "ANSWER"]],
            ["BIRTHDATE", ["BIRTHDATE_ANSWER", "ANSWER"]],
            ["PATIENT_NAME", ["PATIENT_NAME_ANSWER", "ANSWER"]],
            ["DOS", ["DOS_ANSWER", "ANSWER"]],
            ["MEMBER_NAME", ["MEMBER_NAME_ANSWER", "ANSWER"]],
            ["MEMBER_NUMBER", ["MEMBER_NUMBER_ANSWER", "ANSWER"]],
            ["BILLED_AMT", ["BILLED_AMT_ANSWER"]],
            ["PAID_AMT", ["PAID_AMT_ANSWER"]],
            ["QUESTION", ["ANSWER"]],
        ]

        for i, (ocr, ann, frame) in enumerate(
            zip(ocr_results, annotation_results, frames)
        ):
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
                if (
                    pred_box == [0.0, 0.0, 0.0, 0.0]
                    or pred_box[2] == 0
                    or pred_box[3] == 0
                ):
                    continue

                line_number = find_line_number(lines_bboxes, pred_box)
                if line_number not in groups:
                    groups[line_number] = []
                groups[line_number].append(j)

            # aggregate boxes into key/value pairs via simple state machine for each line
            aggregated_keys = {}

            for line_idx, line_box in enumerate(lines_bboxes):
                if line_idx not in groups:
                    logger.debug(
                        f"Line does not have any groups : {line_idx} : {line_box}"
                    )
                    continue

                prediction_indexes = np.array(groups[line_idx])
                line_aggregator = []
                color_map = {"ADDRESS": get_random_color()}

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
                            color_map[field]
                            if field in color_map
                            else get_random_color()
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

                        score_avg = round(
                            np.average([item["score"] for item in picks]), 6
                        )
                        block = merge_bboxes_as_block([item["bbox"] for item in picks])

                        new_item = picks[0]
                        new_item["score"] = score_avg
                        new_item["bbox"] = block

                        aggregated_keys[_k] = np.concatenate(([new_item], remaining))

            # expected fields groups that indicate that the field could have been present but there was not associated
            possible_fields = {
                "PAN": ["PAN", "PAN_ANSWER"],
                "PATIENT_NAME": ["PATIENT_NAME", "PATIENT_NAME_ANSWER"],
                "DOS": ["DOS", "DOS_ANSWER"],
                "MEMBER_NAME": ["MEMBER_NAME", "MEMBER_NAME_ANSWER"],
                "MEMBER_NUMBER": ["MEMBER_NUMBER", "MEMBER_NUMBER_ANSWER"],
                "CLAIM_NUMBER": ["CLAIM_NUMBER", "CLAIM_NUMBER_ANSWER"],
                "BIRTHDATE": ["BIRTHDATE", "BIRTHDATE_ANSWER"],
                "BILLED_AMT": ["BILLED_AMT", "BILLED_AMT_ANSWER"],
                "PAID_AMT": ["PAID_AMT", "PAID_AMT_ANSWER"],
            }

            possible_field_meta = {}

            for field in possible_fields.keys():
                fields = possible_fields[field]
                possible_field_meta[field] = {"page": i, "found": False, "fields": []}
                for k in aggregated_keys.keys():
                    ner_keys = aggregated_keys[k]
                    for ner_key in ner_keys:
                        key = ner_key["key"]
                        if key in fields:
                            possible_field_meta[field]["found"] = True
                            possible_field_meta[field]["fields"].append(key)

            aggregated_meta.append({"page": i, "fields": possible_field_meta})

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
                                    logger.warning(
                                        "Answer is not on the right of question"
                                    )
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

        self.decorate_kv_with_text(aggregated_kv, frames)
        # visualize results per page
        if self.debug_visuals:
            for k in range(0, len(frames)):
                output_filename = f"/tmp/tensors/kv_{file_hash}_{k}.png"
                items = [row for row in aggregated_kv if int(row["page"]) == k]
                visualize_extract_kv(output_filename, frames[k], items)

        logger.info(f"aggregated_kv : {aggregated_kv}")
        results = {"meta": aggregated_meta, "kv": aggregated_kv}

        return results

    def decorate_kv_with_text(self, aggregated_kv, frames):
        """Decorate our answers with proper TEXT"""
        regions = []

        def create_region(field_id, page_index, bbox):
            box = np.array(bbox).astype(np.int32)
            x, y, w, h = box
            return {
                "id": field_id,
                "pageIndex": page_index,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
            }

        # performing secondary OCR yields much better results as we are using
        # RAW-LINE as our segmentation method
        # aggregate results for OCR extraction
        for k, agg_result in enumerate(aggregated_kv):
            page_index = int(agg_result["page"])
            category = agg_result["category"]

            regions.append(
                create_region(
                    f"{category}_{k}_k",
                    page_index,
                    agg_result["value"]["question"]["bbox"],
                )
            )
            regions.append(
                create_region(
                    f"{category}_{k}_v",
                    page_index,
                    agg_result["value"]["answer"]["bbox"],
                )
            )
        kwa = {
            "payload": {
                "output": "json",
                "mode": "raw_line",
                "format": "xywh",
                "filter_snippets": True,
                "regions": regions,
            }
        }
        region_results = self.text_executor.extract(docs_from_image(frames), **kwa)
        region_results = region_results["regions"]
        # merge results
        for k, agg_result in enumerate(aggregated_kv):
            category = agg_result["category"]
            for region in region_results:
                rid = region["id"]
                if rid == f"{category}_{k}_k":
                    agg_result["value"]["question"]["text"] = {
                        "text": region["text"],
                        "confidence": region["confidence"],
                    }
                if rid == f"{category}_{k}_v":
                    agg_result["value"]["answer"]["text"] = {
                        "text": region["text"],
                        "confidence": region["confidence"],
                    }

    def preprocess(self, src_image: str):
        """Pre-process src image for NER extraction"""

        if not os.path.exists(src_image):
            print(f"File not found : {src_image}")
            return

        # Obtain OCR results
        file_hash = hash_file(src_image)
        root_dir = __marie_home__

        ensure_exists(os.path.join(root_dir, "ocr"))
        ensure_exists(os.path.join(root_dir, "annotation"))

        ocr_json_path = os.path.join(root_dir, "ocr", f"{file_hash}.json")
        annotation_json_path = os.path.join(root_dir, "annotation", f"{file_hash}.json")

        print(f"Root      : {root_dir}")
        print(f"SrcImg    : {src_image}")
        print(f"Hash      : {file_hash}")
        print(f"OCR file  : {ocr_json_path}")
        print(f"NER file  : {annotation_json_path}")

        if not os.path.exists(ocr_json_path) and self.text_executor is None:
            raise Exception(f"OCR File not found : {ocr_json_path}")

        loaded, frames = load_image(src_image, img_format="pil")
        if not loaded:
            raise Exception(f"Unable to load image file: {src_image}")

        if os.path.exists(ocr_json_path):
            ocr_results = load_json_file(ocr_json_path)
            if "error" in ocr_results:
                msg = ocr_results["error"]
                logger.info(f"Retrying document > {file_hash} due to : {msg}")
                os.remove(ocr_json_path)

        if not os.path.exists(ocr_json_path):
            ocr_results, frames = obtain_ocr(src_image, self.text_executor)
            # convert CV frames to PIL frame
            frames = convert_frames(frames, img_format="pil")
            store_json_object(ocr_results, ocr_json_path)

        if ocr_results is None or "error" in ocr_results:
            return False, frames, [], [], ocr_results, file_hash

        if self.debug_visuals:
            visualize_icr(frames, ocr_results, file_hash)

        assert len(ocr_results) == len(frames)
        boxes = []
        words = []

        for k, (result, image) in enumerate(zip(ocr_results, frames)):
            if not isinstance(image, Image.Image):
                raise "Frame should have been an PIL.Image instance"
            boxes.append([])
            words.append([])

            for i, word in enumerate(result["words"]):
                words[k].append(word["text"])
                boxes[k].append(
                    normalize_bbox(word["box"], (image.size[0], image.size[1]))
                )
                # This is to prevent following error
                # The expanded size of the tensor (516) must match the existing size (512) at non-singleton dimension 1.
                if len(boxes[k]) == 512:
                    print("Clipping MAX boxes at 512")
                    break
            assert len(words[k]) == len(boxes[k])
        assert len(frames) == len(boxes) == len(words)
        return True, frames, boxes, words, ocr_results, file_hash

    def process(self, frames, boxes, words, file_hash):
        """process NER extraction"""

        annotations = []
        labels, id2label, label2id = get_label_info()

        for k, (_image, _boxes, _words) in enumerate(zip(frames, boxes, words)):
            if not isinstance(_image, Image.Image):
                raise "Frame should have been an PIL.Image instance"

            width = _image.size[0]
            height = _image.size[1]

            all_predictions, all_boxes, all_scores = self.inference(
                _image,
                _words,
                _boxes,
                labels,
                0.1,
            )

            true_predictions = all_predictions[0]
            true_boxes = all_boxes[0]
            true_scores = all_scores[0]

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
            annotations.append(annotation)

            if self.debug_visuals:
                output_filename = f"/tmp/tensors/prediction_{file_hash}_{k}.png"
                visualize_prediction(
                    output_filename,
                    _image,
                    true_predictions,
                    true_boxes,
                    true_scores,
                    label2color=get_label_colors(),
                )

        annotation_json_path = os.path.join(
            __marie_home__, "annotation", f"{file_hash}.json"
        )
        ensure_exists(os.path.join(__marie_home__, "annotation"))

        store_json_object(annotations, annotation_json_path)
        return annotations

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

        loaded, frames, boxes, words, ocr_results, file_hash = self.preprocess(
            image_src
        )
        annotations = self.process(frames, boxes, words, file_hash)
        ner_results = self.postprocess(frames, annotations, ocr_results, file_hash)

        return ner_results
