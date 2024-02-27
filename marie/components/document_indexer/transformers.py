import os
from pprint import pprint
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from docarray import DocList
from PIL import Image, ImageDraw
from pydantic import BaseModel
from torch.nn import Module
from transformers import (
    AutoModelForTokenClassification,
    LayoutLMv3FeatureExtractor,
    LayoutLMv3Processor,
    LayoutLMv3TokenizerFast,
)

from marie.constants import __marie_home__, __model_path__
from marie.executor.ner.utils import (
    draw_box,
    get_font,
    get_random_color,
    normalize_bbox,
    unnormalize_box,
    visualize_extract_kv,
    visualize_prediction,
)
from marie.logging.logger import MarieLogger
from marie.models.utils import initialize_device_settings
from marie.utils.overlap import find_overlap_horizontal, merge_bboxes_as_block

from ...api.docs import BatchableMarieDoc, MarieDoc
from ...boxes import PSMode
from ...boxes.line_processor import find_line_number, line_merge
from ...logging.profile import TimeContext
from ...ocr import CoordinateFormat, OcrEngine
from ...ocr.ocr_engine import reset_bbox_cache
from ...ocr.util import get_known_ocr_engines
from ...registry.model_registry import ModelRegistry
from ...utils.docs import convert_frames, frames_from_docs
from ...utils.image_utils import hash_frames_fast
from ...utils.json import load_json_file, store_json_object
from ...utils.utils import ensure_exists
from .base import BaseDocumentIndexer


class DebugVisualsHandler:
    def __init__(self, debug_visuals, frame):
        self.debug_visuals = debug_visuals

        self.viz_img = frame.copy()
        self.draw = ImageDraw.Draw(self.viz_img, "RGBA")
        self.font = get_font(14)

    def draw_box(
        self, bbox, text: Optional[str] = None, color_map: dict[str, str] = None
    ) -> None:
        if color_map is None:
            color_map = {}
        color = color_map[text] if text in color_map else get_random_color()

        draw_box(
            self.draw,
            bbox,
            text,
            color,
            self.font,
        )

    def save(self, output_filename: str):
        self.viz_img.save(output_filename)


class LineGroup(BaseModel):
    bbox: list
    key: str
    line: int
    score: float

    class Config:
        arbitrary_types_allowed = False


class EntityGroup(BaseModel):
    bbox: list
    key: str
    group: list[LineGroup]

    class Config:
        arbitrary_types_allowed = False


class TransformersDocumentIndexer(BaseDocumentIndexer):
    """
    Transformer based model for document indexing(Named Entity Recognition) using the HuggingFace's transformers framework
    (https://github.com/huggingface/transformers).
    """

    def __init__(
        self,
        model_name_or_path: Union[str, os.PathLike],
        model_version: Optional[str] = None,
        tokenizer: Optional[str] = None,
        use_gpu: bool = True,
        top_k: Optional[int] = 1,
        task: str = "transformers-document-indexer",
        batch_size: int = 16,
        use_auth_token: Optional[Union[str, bool]] = None,
        devices: Optional[List[Union[str, "torch.device"]]] = None,
        show_error: Optional[Union[str, bool]] = True,
        ocr_engine: Optional[OcrEngine] = None,
        **kwargs,
    ):
        """
        Initializes the document indexer for Named Entity Extraction.

        :param model_name_or_path: The name or path of the model to be used.
        :param model_version: The version of the model. Defaults to None.
        :param tokenizer: The tokenizer to be used. Defaults to None.
        :param use_gpu: Whether to use GPU for processing. Defaults to True.
        :param top_k: The number of top results to return. Defaults to 1.
        :param task: The task to be performed. Defaults to "transformers-document-indexer".
        :param batch_size: The size of the batch to be processed. Defaults to 16.
        :param use_auth_token: The authentication token to be used. Defaults to None.
        :param devices: The devices to be used for processing. Defaults to None.
        :param show_error: Whether to show errors. Defaults to True.
        :param ocr_engine: The OCR engine to be used. Defaults to None, in which case the default OCR engine is used.
        :param kwargs: Additional keyword arguments.
        :returns: None
        """

        super().__init__(**kwargs)
        self.logger = MarieLogger(self.__class__.__name__).logger
        self.show_error = show_error  # show prediction errors
        self.batch_size = batch_size
        self.task = task
        self.logger.info(f"Document indexer : {model_name_or_path}")

        resolved_devices, _ = initialize_device_settings(
            devices=devices, use_cuda=use_gpu, multi_gpu=False
        )
        if len(resolved_devices) > 1:
            self.logger.warning(
                "Multiple devices are not supported in %s inference, using the first device %s.",
                self.__class__.__name__,
                resolved_devices[0],
            )
        self.device = resolved_devices[0]

        registry_kwargs = {
            "__model_path__": __model_path__,
            "use_auth_token": use_auth_token,
        }

        model_name_or_path = ModelRegistry.get(
            model_name_or_path,
            version=None,
            raise_exceptions_for_missing_entries=True,
            **registry_kwargs,
        )
        assert os.path.exists(model_name_or_path)
        self.logger.info(f"Resolved model : {model_name_or_path}")

        ensure_exists("/tmp/tensors")
        ensure_exists("/tmp/tensors/json")

        # TODO: config could be loaded from a file or passed as a parameter
        config_path = os.path.join(model_name_or_path, "marie.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                "Expected config 'marie.json' not found in model directory"
            )

        self.logger.info(f"NER loading from : {config_path}")
        self.init_configuration = load_json_file(config_path)

        self.debug_visuals = self.init_configuration["debug"]["visualize"]["enabled"]
        self.debug_visuals_overlay = self.init_configuration["debug"]["visualize"][
            "overlay"
        ]
        self.debug_visuals_icr = self.init_configuration["debug"]["visualize"]["icr"]
        self.debug_visuals_ner = self.init_configuration["debug"]["visualize"]["ner"]
        self.debug_visuals_prediction = self.init_configuration["debug"]["visualize"][
            "prediction"
        ]

        self.debug_scores = self.init_configuration["debug"]["scores"]
        self.debug_colors = self.init_configuration["debug"]["colors"]

        self.labels = self.init_configuration["labels"]

        self.model = self.__load_model(
            model_name_or_path, self.labels, self.device.type
        )
        self.processor = self.__create_processor()
        if ocr_engine:
            self.ocr_engine = ocr_engine
        else:
            self.logger.warning("Using default OCR engine for Document Indexing")
            ocr_engines = get_known_ocr_engines(device="cuda", engine="default")
            self.ocr_engine = ocr_engines["default"]

    def __create_processor(self):
        """prepare for the model"""
        # Method:2 Create Layout processor with custom future extractor
        # Max model size is 512, so we will need to handle any documents larger than that
        feature_extractor = LayoutLMv3FeatureExtractor(
            apply_ocr=False, do_resize=True, resample=Image.BILINEAR
        )
        tokenizer = LayoutLMv3TokenizerFast.from_pretrained(
            "microsoft/layoutlmv3-large",
            only_label_first_subword=False,
        )

        processor = LayoutLMv3Processor(
            feature_extractor=feature_extractor, tokenizer=tokenizer
        )

        return processor

    def __load_model(self, model_dir: str, labels: list[str], device: str):
        """
        Create token classification model
        """
        labels, id2label, label2id = self.get_label_info(labels)
        model = AutoModelForTokenClassification.from_pretrained(
            model_dir, num_labels=len(labels), label2id=label2id, id2label=id2label
        )

        model.eval()
        model.to(device)

        return model

    def optimize_model(self, model: nn.Module) -> Callable | Module:
        """Optimizes the model for inference. This method is called by the __init__ method."""
        if True:
            return model
        try:
            with TimeContext("Compiling model", logger=self.logger):
                # Optimize model for Inference time
                for param in model.parameters():
                    param.grad = None
                print("**** COMPILING NER Model***")
                import torch._dynamo as dynamo

                # Default torchinductor causes OOM when running on 24GB GPU, cache memory is never relased
                # Switching to use cudagraphs
                # torch._dynamo.config.set("inductor", "cache_memory", 0)
                # mode options: default, reduce-overhead, max-autotune
                # default, reduce-overhead, max-autotune
                # ['aot_ts_nvfuser', 'cudagraphs', 'inductor', 'ipex', 'nvprims_nvfuser', 'onnxrt', 'tensorrt', 'tvm']

                model = torch.compile(
                    model, mode="max-autotune", fullgraph=True, backend="cudagraphs"
                )
                print("**** COMPILED ***")

                return model
        except Exception as err:
            self.logger.warning(f"Model compile not supported: {err}")
            return model

    def get_label_info(self, labels: list[str]):
        self.logger.debug(f"Labels : {labels}")

        id2label = {v: k for v, k in enumerate(labels)}
        label2id = {k: v for v, k in enumerate(labels)}

        return labels, id2label, label2id

    def _filter(
        self, values: List[Any], probabilities: List[float], threshold: float
    ) -> List[Any]:
        return [
            value for probs, value in zip(probabilities, values) if probs >= threshold
        ]

    def preprocess(
        self, frames: List, words: List[List[str]], boxes: List[List[List[int]]]
    ) -> Tuple[List, List[List[str]], List[List[List[int]]]]:
        """Preprocess the input data for inference. This method is called by the predict method.
        :param frames: The frames to be preprocessed.
        :param words: The words to be preprocessed.
        :param boxes: The boxes to be preprocessed, in the format (x, y, w, h).
        :returns: The preprocessed frames, words, and boxes (normalized).
        """
        assert len(frames) == len(boxes) == len(words)
        frames = convert_frames(frames, img_format="pil")
        normalized_boxes = []

        for frame, box_set, word_set in zip(frames, boxes, words):
            if not isinstance(frame, Image.Image):
                raise ValueError("Frame should have been an PIL.Image instance")
            nbox = []
            for i, box in enumerate(box_set):
                nbox.append(normalize_bbox(box_set[i], (frame.size[0], frame.size[1])))
            normalized_boxes.append(nbox)

        assert len(frames) == len(normalized_boxes) == len(words)

        return frames, words, normalized_boxes

    def predict(
        self,
        documents: DocList[MarieDoc],
        words: List[List[str]],
        boxes: List[List[List[int]]],
        batch_size: Optional[int] = None,
    ) -> DocList[MarieDoc]:
        if batch_size is None:
            batch_size = self.batch_size

        if len(documents) == 0:
            return documents

        if self.task == "transformers-document-indexer":
            assert (
                words is not None and boxes is not None
            ), "words and boxes must be provided for sequence classification"
            assert (
                len(documents) == len(words) == len(boxes)
            ), "documents, words and boxes must have the same length"

        frames = frames_from_docs(documents)
        file_hash = hash_frames_fast(frames)
        frames, words, boxes_normal = self.preprocess(frames, words, boxes)

        batchable_docs = DocList(
            BatchableMarieDoc(
                tensor=doc.tensor,
                words=word,
                boxes=box,
            )
            for doc, word, box in zip(documents, words, boxes_normal)
        )
        annotations = []

        for k, (batchable_doc, _image) in enumerate(zip(batchable_docs, frames)):
            t = batchable_doc.tensor
            w = batchable_doc.words
            b = batchable_doc.boxes

            width = _image.size[0]
            height = _image.size[1]

            true_predictions, true_boxes, true_scores = self.inference(
                image=_image, words=w, boxes=b, labels=self.labels, threshold=0.5
            )

            # show detail scores
            if self.debug_scores:
                for i, val in enumerate(true_predictions):
                    tp = true_predictions[i]
                    score = true_scores[i]
                    self.logger.debug(f" >> {tp} : {score}")

            annotation = {
                "meta": {"imageSize": {"width": width, "height": height}, "page": k},
                "predictions": true_predictions,
                "boxes": true_boxes,
                "scores": true_scores,
            }
            annotations.append(annotation)

            if self.debug_visuals and self.debug_visuals_prediction:
                output_filename = f"/tmp/tensors/prediction_{file_hash}_{k}.png"
                visualize_prediction(
                    output_filename,
                    _image,
                    true_predictions,
                    true_boxes,
                    true_scores,
                    label2color=self.debug_colors,
                )

        ensure_exists(os.path.join(__marie_home__, "annotation"))
        annotation_json_path = os.path.join(
            __marie_home__, "annotation", f"{file_hash}.json"
        )
        store_json_object(annotations, annotation_json_path)

        results = self.postprocess(frames, annotations, words, boxes, file_hash)
        # TODO : persist results
        # self.persist(ref_id, ref_type, ner_results)

        for k, document in enumerate(documents):
            if self.task == "transformers-document-indexer":
                filtered_meta = [val for val in results["meta"] if val["page"] == k]
                filtered_kv = [val for val in results["kv"] if val["page"] == k]
                filtered_ner = [val for val in results["ner"] if val["page"] == k]
                filtered_groups = [val for val in results["groups"] if val["page"] == k]

                document.tags["indexer"] = {
                    "page": k,
                    "meta": filtered_meta,
                    "kv": filtered_kv,
                    "ner": filtered_ner,
                    "groups": filtered_groups,
                }
            else:
                raise ValueError(f"Unsupported task: {self.task}")
        return documents

    def decorate_aggregates_with_text(
        self, aggregated_kv: list[dict], frames: List[Image.Image]
    ):
        """Decorate our answers with proper TEXT
        Performing secondary OCR yields much better results as we are using RAW-LINE as our segmentation method
        """
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

        # aggregate results for OCR extraction
        for k, agg_result in enumerate(aggregated_kv):
            page_index = int(agg_result["page"])
            category = agg_result["category"]

            if "question" in agg_result["value"]:
                regions.append(
                    create_region(
                        f"{category}_{k}_k",
                        page_index,
                        agg_result["value"]["question"]["bbox"],
                    )
                )

            if "answer" in agg_result["value"]:
                regions.append(
                    create_region(
                        f"{category}_{k}_v",
                        page_index,
                        agg_result["value"]["answer"]["bbox"],
                    )
                )

        # nothing to decorate
        if len(regions) == 0:
            return

        region_results = self.ocr_engine.extract(
            frames,
            PSMode.RAW_LINE,
            CoordinateFormat.XYWH,
            regions,
            **{"filter_snippets": True},
        )
        reset_bbox_cache()

        # possible failure in extracting data for region
        if "regions" not in region_results:
            self.logger.warning("No regions returned")
            return
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

    def inference(
        self,
        image: Any,
        words: List[Any],
        boxes: List[Any],
        labels: List[str],
        threshold: float,
    ) -> Tuple[List, List, List]:
        """Run Inference

        :param image: The image to be processed. This can be a PIL.Image or numpy.
        :param words: The words to be processed.
        :param boxes: The boxes to be processed, in the format (x, y, w, h). Boxes should be normalized to 1000
        :param labels: The labels to be used for inference.
        :param threshold: The threshold to be used for filtering the results.
        :returns: The predictions, boxes (normalized), and scores.
        """
        self.logger.info(f"Performing inference")
        model = self.model
        processor = self.processor
        device = self.device

        labels, id2label, label2id = self.get_label_info(labels)
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        self.logger.info(
            f"Tokenizer parallelism: {os.environ.get('TOKENIZERS_PARALLELISM', 'true')}"
        )

        width, height = image.size
        # partition the words and boxes into batches of 512 tokens with 128 stride
        # stride is the number of tokens to move forward, this is to handle long documents that are larger than 512 tokens
        # there will be overlap between the batches, so we will need to handle that later

        self.logger.info("Named Entity Inference")
        self.logger.info(f"Words : {len(words)} ::  {words}")
        self.logger.info(f"Boxes : {len(boxes)}")
        assert len(words) == len(boxes)

        encoding = processor(
            # fmt: off
            image,
            words,
            boxes=boxes,
            truncation=True,
            return_offsets_mapping=True,
            return_overflowing_tokens=True,
            stride=128,
            padding="max_length",
            return_tensors="pt",
            max_length=512,
            # fmt: on
        )

        offset_mapping_batched = encoding.pop("offset_mapping")
        overflow_to_sample_mapping = encoding.pop("overflow_to_sample_mapping")
        # encoding["pixel_values"] = torch.stack(encoding["pixel_values"], dim=0)

        # Debug tensor info
        self.debug_visuals = True
        if self.debug_visuals:
            img_tensor = encoding["pixel_values"]
            img = Image.fromarray(
                (img_tensor[0].cpu()).numpy().astype(np.uint8).transpose(1, 2, 0)
            )
            img.save(f"/tmp/tensors/tensor.png")

        # ensure proper device placement
        for k in encoding.keys():
            if k != "pixel_values":
                encoding[k] = encoding[k].to(device)
            else:
                encoding[k] = torch.cat([x.unsqueeze(0) for x in encoding[k]]).to(
                    device
                )

        # Perform forward pass
        with torch.inference_mode():
            outputs = model(**encoding)
            # Get the predictions and probabilities
            # The model outputs logits of shape (batch_size, seq_len, num_labels).
            logits = outputs.logits
            batch_size, seq_len, num_labels = logits.shape
            # Get the predictions and bounding boxes by batch and convert to list
            predictions_batched = logits.argmax(-1).squeeze().tolist()
            token_boxes_batched = encoding.bbox.squeeze().tolist()
            normalized_logits_batched = (
                outputs.logits.softmax(dim=-1).squeeze().tolist()
            )

            # If batch size is 1, convert to list
            if batch_size == 1:
                predictions_batched = [predictions_batched]
                token_boxes_batched = [token_boxes_batched]
                normalized_logits_batched = [normalized_logits_batched]

            out_prediction = []
            out_boxes = []
            out_scores = []

            for batch_index in range(batch_size):
                # Get the predictions and bounding boxes for the current batch
                predictions = predictions_batched[batch_index]
                token_boxes = token_boxes_batched[batch_index]
                normalized_logits = normalized_logits_batched[batch_index]
                offset_mapping = offset_mapping_batched[batch_index]

                # TODO : Filer the results
                # Filter the predictions and bounding boxes based on a threshold
                # predictions = _filter(_predictions, probs, threshold)
                # token_boxes = _filter(_token_boxes, probs, threshold)

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
                # convert boxes from float to int
                true_boxes = [[int(b) for b in box] for box in true_boxes]

                true_scores = [
                    round(normalized_logits[idx][val], 6)
                    for idx, val in enumerate(predictions)
                    if not is_subword[idx]
                ]

                assert len(true_predictions) == len(true_boxes) == len(true_scores)

                # Not sure why we have this, but we need to remove [0, 0, 0, 0] boxes
                true_predictions = [
                    pred
                    for pred, box in zip(true_predictions, true_boxes)
                    if box != [0, 0, 0, 0]
                ]
                true_boxes = [box for box in true_boxes if box != [0, 0, 0, 0]]
                true_scores = [
                    score
                    for score, box in zip(true_scores, true_boxes)
                    if box != [0, 0, 0, 0]
                ]

                # check if there are duplicate boxes (example : 159000444_1.png)
                # why are there duplicate boxes??
                for box in true_boxes:
                    if true_boxes.count(box) > 1:
                        self.logger.warning(f"Duplicate box found : {box}")
                        current_idx = true_boxes.index(box)
                        true_predictions.pop(current_idx)
                        true_boxes.pop(current_idx)
                        true_scores.pop(current_idx)

                if batch_index > 0:
                    for idx, box in enumerate(out_boxes):
                        if box in true_boxes:
                            current_idx = true_boxes.index(box)
                            if true_scores[current_idx] >= out_scores[idx]:
                                out_prediction[idx] = true_predictions[current_idx]
                                out_scores[idx] = true_scores[current_idx]

                            true_predictions.pop(current_idx)
                            true_boxes.pop(current_idx)
                            true_scores.pop(current_idx)

                out_prediction.extend(true_predictions)
                out_boxes.extend(true_boxes)
                out_scores.extend(true_scores)

        original_boxes = []
        for box in boxes:
            original_boxes.append([int(b) for b in unnormalize_box(box, width, height)])

        # align words and boxes with predictions and scores
        out_prediction, out_boxes, out_scores = self.align_predictions(
            words, original_boxes, out_prediction, out_boxes, out_scores
        )

        assert len(out_prediction) == len(words) == len(boxes)
        return out_prediction, out_boxes, out_scores

    def align_predictions(
        self,
        words,
        original_boxes: [],
        out_prediction: [],
        out_boxes: [],
        out_scores: [],
    ) -> Tuple[List[str], List[List[int]], List[float]]:
        """
        Aligns the predictions with the words and boxes.

        :param words: The words to be aligned with the predictions.
        :param original_boxes: The original boxes to be aligned with the predictions.
        :param out_prediction: The predictions to be aligned.
        :param out_boxes: The boxes to be aligned with the predictions.
        :param out_scores: The scores of the predictions.
        :returns: Aligned predictions, boxes, and scores.
        """

        aligned_prediction = []
        aligned_boxes = []
        aligned_scores = []

        for idx, word in enumerate(words):
            box = original_boxes[idx]
            if box in out_boxes:
                current_idx = out_boxes.index(box)
                aligned_prediction.append(out_prediction[current_idx])
                aligned_boxes.append(out_boxes[current_idx])
                aligned_scores.append(out_scores[current_idx])
            else:
                raise ValueError(f"Box not found for alignment: {box}")

        return aligned_prediction, aligned_boxes, aligned_scores

    def group_composite_entities(
        self,
        entities_to_group: list[dict],
        lines_bboxes: list[list[int]],
        true_predictions: list[str],
        true_boxes: list[list[int]],
        true_scores: list[float],
        frame,
    ) -> dict[str, dict[str, EntityGroup]]:

        visualizer = DebugVisualsHandler(self.debug_visuals, frame)

        grouped_entities = {}

        for entity in entities_to_group:
            expected_keys = entity["entities"]
            entity_name = entity["name"]

            filtered_predictions = []
            filtered_boxes = []
            filtered_scores = []

            for pred_idx, (prediction, pred_box, pred_score) in enumerate(
                zip(true_predictions, true_boxes, true_scores)
            ):
                if prediction[2:] in expected_keys:
                    filtered_predictions.append(prediction)
                    filtered_boxes.append(pred_box)
                    filtered_scores.append(pred_score)

            groups = self.group_by_line(
                lines_bboxes, filtered_boxes, filtered_predictions, filtered_scores
            )
            aggregated_keys = self.aggregate_groups_by_line(
                expected_keys,
                groups,
                lines_bboxes,
                filtered_boxes,
                filtered_predictions,
                filtered_scores,
                visualizer,
            )
            self.fix_misslabeled_tokens(expected_keys, aggregated_keys)
            visualizer.save(f"/tmp/tensors/extract_group_{entity_name}.png")

            last_line = 0
            group_id = 0
            max_line_diff = 2
            collected_groups = {}

            for key, groups in aggregated_keys.items():
                for group in groups:
                    line_diff = group.line - last_line
                    if last_line != 0 and line_diff > max_line_diff:
                        group_id += 1
                    if group_id not in collected_groups:
                        collected_groups[group_id] = []
                    collected_groups[group_id].append(group)
                    last_line = group.line

            merge_groups = {}

            for group_id, group in collected_groups.items():
                group = sorted(group, key=lambda x: x.bbox[0])
                bboxes = [group.bbox for group in group]
                visited = [False for _ in range(0, len(bboxes))]
                for idx in range(0, len(bboxes)):
                    if visited[idx]:
                        continue
                    ag_key = f"{group_id}_{idx}"
                    visited[idx] = True
                    overlaps, indexes, scores = find_overlap_horizontal(
                        bboxes[idx], bboxes
                    )
                    merge_groups[ag_key] = [group[idx]]
                    for _, overlap_idx in zip(overlaps, indexes):
                        visited[overlap_idx] = True
                        merge_groups[ag_key].append(group[overlap_idx])

            for key, group in merge_groups.items():
                sorted_group = sorted(group, key=lambda x: x.line)
                bboxes = [g.bbox for g in sorted_group]
                block = merge_bboxes_as_block(bboxes)
                merge_groups[key] = EntityGroup(
                    bbox=block,
                    key=f"{entity_name}_{key}",
                    group=sorted_group,
                )

            grouped_entities[entity_name] = merge_groups

            for key, group in merge_groups.items():
                visualizer = DebugVisualsHandler(self.debug_visuals, frame)

                self.logger.info(f"Group : {key} : {group}")
                bbox = group.bbox
                visualizer.draw_box(bbox, key, color_map=None)
                visualizer.save(f"/tmp/tensors/extract_group_{entity_name}_{key}.png")

        return grouped_entities

    def postprocess(
        self,
        frames: List[Image.Image],
        annotations: List[dict],
        words: List[List[str]],
        boxes: List[List[List[int]]],
        file_hash,
    ):
        """Postprocess the results of the inference. This method is called by the predict method.
        :param frames: The frames to be postprocessed.
        :param annotations: The annotations to be postprocessed.
        :param words:  The words to be processed.
        :param boxes:   The boxes to be processed, in the format (x, y, w, h). Boxes should be normalized to 1000
        :param file_hash: The hash of the file to be postprocessed.
        :returns: The postprocessed results.
        """

        self.logger.info(f"Postprocessing results")
        assert len(annotations) == len(words) == len(boxes) == len(frames)
        for k, _image in enumerate(frames):
            if not isinstance(_image, Image.Image):
                raise "Frame should have been an PIL.Image instance"

        aggregated_ner = []
        aggregated_groups = []
        aggregated_kv = []
        aggregated_meta = []

        # expected NER and key/value pairs
        expected_ner = self.init_configuration["expected_ner"]
        expected_keys = self.init_configuration["expected_keys"]
        expected_pair = self.init_configuration["expected_pair"]
        entities_to_group = self.init_configuration["entities_to_group"]

        for i, (_boxes, _words, annotation, frame) in enumerate(
            zip(boxes, words, annotations, frames)
        ):
            self.logger.info(f"Processing page # {i}")
            frame_box = frame
            if isinstance(frame, Image.Image):
                frame_box = np.array(frame)
            lines_bboxes = line_merge(frame_box, _boxes)
            # lines and boxes are already in the right reading order TOP->BOTTOM, LEFT-TO-RIGHT so no need to sort
            true_predictions = annotation["predictions"]
            true_boxes = annotation["boxes"]
            true_scores = annotation["scores"]
            visualizer = DebugVisualsHandler(self.debug_visuals, frame)

            grouped_entities = {}
            if entities_to_group is not None and len(entities_to_group) > 0:
                grouped_entities = self.group_composite_entities(
                    entities_to_group,
                    lines_bboxes,
                    true_predictions,
                    true_boxes,
                    true_scores,
                    frame,
                )
            groups = self.group_by_line(
                lines_bboxes, true_boxes, true_predictions, true_scores
            )
            aggregated_keys = self.aggregate_groups_by_line(
                expected_keys,
                groups,
                lines_bboxes,
                true_boxes,
                true_predictions,
                true_scores,
                visualizer,
            )
            self.fix_misslabeled_tokens(expected_keys, aggregated_keys)

            # expected fields groups that indicate that the field could have been present
            # but it might not have been associated with KV pair mapping, does not apply to NER
            possible_fields = self.init_configuration["possible_fields"]
            possible_field_meta = {}

            for field in possible_fields.keys():
                fields = possible_fields[field]
                possible_field_meta[field] = {"found": False, "fields": []}
                for k in aggregated_keys.keys():
                    ner_keys = aggregated_keys[k]
                    for ner_key in ner_keys:
                        key = ner_key.key
                        if (
                            key in fields
                            and key not in possible_field_meta[field]["fields"]
                        ):
                            possible_field_meta[field]["found"] = True
                            possible_field_meta[field]["fields"].append(key)

            aggregated_meta.append({"page": i, "fields": possible_field_meta})

            # Aggregate KV pairs, this can overlap with NER tags so caution need to be taken
            for pair in expected_pair:
                expected_question = pair[0]
                expected_answer = pair[1]

                for k in aggregated_keys.keys():
                    ner_keys = aggregated_keys[k]

                    found_key = None
                    found_val = None

                    for ner_key in ner_keys:
                        key = ner_key.key
                        if expected_question == key:
                            found_key = ner_key
                            continue
                        # find the first match
                        if found_key is not None and found_val is None:
                            # find the first match
                            for exp_key in expected_answer:
                                if key in exp_key:
                                    found_val = ner_key
                                    break

                            if found_val is not None:
                                bbox_q = found_key.bbox
                                bbox_a = found_val.bbox

                                if bbox_a[0] < bbox_q[0]:
                                    self.logger.warning(
                                        "Answer is not on the right of question"
                                    )
                                    continue

                                kv_result = {
                                    "page": i,
                                    "category": found_key.key,
                                    "value": {
                                        "question": found_key.__dict__,
                                        "answer": found_val.__dict__,
                                    },
                                }

                                aggregated_kv.append(kv_result)

            # Collect NER tags
            for tag in expected_ner:
                for k in aggregated_keys.keys():
                    ner_keys = aggregated_keys[k]
                    for ner_key in ner_keys:
                        key = ner_key.key
                        if key == tag:
                            ner_result = {
                                "page": i,
                                "category": tag,
                                "value": {
                                    "answer": ner_key.__dict__,
                                },
                            }
                            aggregated_ner.append(ner_result)

            # Collect grouped entities
            pprint(grouped_entities)
            for entity, groups in grouped_entities.items():

                for group_id, group in groups.items():
                    aggregated_entity_groups = []
                    self.logger.info(f"Group : {entity} : {group_id} : {group}")

                    for item in group.group:
                        self.logger.info(f"Group : {entity} : {group_id} : {item}")
                        ner_result = {
                            "page": i,
                            "category": f"{entity}_{group_id}_{item.key}",
                            "value": {
                                "answer": {
                                    "bbox": item.bbox,
                                },
                            },
                        }
                        aggregated_entity_groups.append(ner_result)
                    self.decorate_aggregates_with_text(aggregated_entity_groups, frames)

                    entity_texts = "\n".join(
                        [
                            item["value"]["answer"]["text"]["text"]
                            for item in aggregated_entity_groups
                        ]
                    )
                    entity_text_confidence = round(
                        np.average(
                            [
                                item["value"]["answer"]["text"]["confidence"]
                                for item in aggregated_entity_groups
                            ]
                        ),
                        6,
                    )

                    aggregated_groups.append(
                        {
                            "page": i,
                            "category": entity,
                            "value": {
                                "answer": {
                                    "bbox": group.bbox,
                                    "text": entity_texts,
                                    "confidence": entity_text_confidence,
                                },
                            },
                        }
                    )

            if self.debug_visuals and self.debug_visuals_overlay:
                visualizer.save(f"/tmp/tensors/extract_{file_hash}_{i}.png")

        self.decorate_aggregates_with_text(aggregated_ner, frames)
        self.decorate_aggregates_with_text(aggregated_kv, frames)

        results = {
            "meta": aggregated_meta,
            "kv": aggregated_kv,
            "ner": aggregated_ner,
            "groups": aggregated_groups,
        }
        self.logger.debug(f" results : {results}")

        # visualize results per page
        if self.debug_visuals and self.debug_visuals_ner:
            for k in range(0, len(frames)):
                output_filename = f"/tmp/tensors/ner_{file_hash}_{k}.png"
                items = []
                items.extend([row for row in aggregated_kv if int(row["page"]) == k])
                items.extend([row for row in aggregated_ner if int(row["page"]) == k])
                items.extend(
                    [row for row in aggregated_groups if int(row["page"]) == k]
                )

                visualize_extract_kv(output_filename, frames[k], items)

        return results

    def fix_misslabeled_tokens(
        self, expected_keys: list[str], aggregated_keys: dict[int, LineGroup]
    ):
        """Check if we have possible overlaps when there is a mislabeled token, this could be a flag
            Strategy used here is a horizontal overlap, if we have it then we will aggregate them
            B-PAN I-PAN I-PAN B-PAN-ANS I-PAN

        :param expected_keys: list of expected keys
        :param aggregated_keys:
        :return:
        """
        if self.init_configuration["mislabeled_token_strategy"] == "aggregate":
            for key in expected_keys:
                for ag_key in aggregated_keys.keys():
                    row_items = aggregated_keys[ag_key]
                    bboxes = [row.bbox for row in row_items if row.key == key]
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
                        items = np.array(aggregated_keys[_k])
                        # there is nothing to merge, except the original block
                        if len(idxs) == 1:
                            continue

                        idxs = np.array(idxs)
                        picks = items[idxs]
                        remaining = np.delete(items, idxs)

                        score_avg = round(np.average([item.score for item in picks]), 6)
                        block = merge_bboxes_as_block([item.bbox for item in picks])

                        new_item = picks[0]
                        new_item.score = score_avg
                        new_item.bbox = block

                        aggregated_keys[_k] = np.concatenate(([new_item], remaining))
        else:
            raise NotImplementedError(
                f"Mislabeled token strategy not supported : {self.init_configuration['mislabeled_token_strategy']}"
            )

    def aggregate_groups_by_line(
        self,
        expected_keys: list[str],
        groups: dict[int, list[int]],
        lines_bboxes: list[list[int]],
        true_boxes: list[list[int]],
        true_predictions: list[str],
        true_scores: list[float],
        visualizer: Optional[DebugVisualsHandler] = None,
    ) -> dict:
        """aggregate boxes into key/value pairs via simple state machine for each line"""
        aggregated_keys = {}
        for line_idx, line_box in enumerate(lines_bboxes):
            if line_idx not in groups:
                self.logger.debug(
                    f"Line does not have any groups : {line_idx} : {line_box}"
                )
                continue

            line_aggregation = self.group_horizontal_span(
                expected_keys, groups[line_idx], true_predictions
            )

            true_boxes = np.array(true_boxes)
            true_scores = np.array(true_scores)

            for line_agg in line_aggregation:
                field = line_agg["key"]
                for group_index in line_agg["groups"]:
                    group_score = round(np.average(true_scores[group_index]), 6)
                    group_bbox = merge_bboxes_as_block(true_boxes[group_index])
                    key_result = LineGroup(
                        **{
                            "line": line_idx,
                            "key": field,
                            "bbox": group_bbox,
                            "score": group_score,
                        }
                    )
                    if line_idx not in aggregated_keys:
                        aggregated_keys[line_idx] = []
                    aggregated_keys[line_idx].append(key_result)
                    if self.debug_visuals and visualizer is not None:
                        visualizer.draw_box(
                            group_bbox,
                            field,
                            color_map=self.init_configuration["debug"]["colors"],
                        )
        return aggregated_keys

    def group_horizontal_span(
        self, expected_keys: list[str], prediction_indexes: [], true_predictions
    ) -> list:
        """group horizontal spans of the same key"""

        line_aggregator = []
        for key in expected_keys:
            spans = []
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
                    spans.append(aggregator)

            if len(spans) > 0:
                line_aggregator.append({"key": key, "groups": spans})
        return line_aggregator

    def group_by_line(
        self,
        lines_bboxes: list[list[int]],
        true_boxes: list[list[int]],
        true_predictions: list[str],
        true_scores: list[float],
    ) -> dict:
        """Aggregate prediction by their line numbers
        :param lines_bboxes: list of lines bounding boxes
        :param true_boxes: list of boxes
        :param true_predictions: list of predictions
        :param true_scores: list of scores
        """

        groups = {}
        for pred_idx, (prediction, pred_box, pred_score) in enumerate(
            zip(true_predictions, true_boxes, true_scores)
        ):
            # discard 'O' other
            label = prediction[2:]
            if not label:
                continue
            # two labels that need to be removed [0.0, 0.0, 0.0, 0.0]  [2578.0, 3 3292.0, 0.0, 0.0]
            if np.array_equal(pred_box, [0.0, 0.0, 0.0, 0.0]) or (
                pred_box[2] == 0 and pred_box[3] == 0
            ):
                continue

            line_number = find_line_number(lines_bboxes, pred_box)
            if line_number not in groups:
                groups[line_number] = []
            groups[line_number].append(pred_idx)

        return groups
