import os
import traceback
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from docarray import DocList
from PIL import Image
from torch.nn import Module
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    LayoutLMv3FeatureExtractor,
    LayoutLMv3ImageProcessor,
    LayoutLMv3Processor,
    LayoutLMv3TokenizerFast,
    pipeline,
)

from marie.components.document_indexer.base import BaseDocumentIndexer
from marie.constants import __model_path__
from marie.executor.ner.utils import (
    draw_box,
    get_font,
    get_random_color,
    normalize_bbox,
    unnormalize_box,
    visualize_extract_kv,
    visualize_icr,
    visualize_prediction,
)
from marie.logging.logger import MarieLogger
from marie.models.utils import initialize_device_settings

from ...api.docs import BatchableMarieDoc, MarieDoc
from ...helper import batch_iterator
from ...logging.profile import TimeContext
from ...registry.model_registry import ModelRegistry
from ...utils.docs import frames_from_docs
from ...utils.json import load_json_file
from ...utils.utils import ensure_exists
from ..util import scale_bounding_box


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
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.logger = MarieLogger(self.__class__.__name__).logger
        self.logger.info(f"Document indexer : {model_name_or_path}")
        self.show_error = show_error  # show prediction errors
        self.batch_size = batch_size
        self.task = task

        self.logger.info(f"NER Extraction component : {model_name_or_path}")

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

    def predict(
        self,
        documents: DocList[MarieDoc],
        words: Optional[List[List[str]]] = None,
        boxes: Optional[List[List[List[int]]]] = None,
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

        # create a named tuple of (document, words, boxes) for each document

        batchable_docs = DocList(
            BatchableMarieDoc(
                tensor=doc.tensor,
                words=word,
                boxes=box,
            )
            for doc, word, box in zip(documents, words, boxes)
        )

        labels = self.labels
        frames = frames_from_docs(batchable_docs)
        for batchable_doc, frame in zip(batchable_docs, frames):
            t = batchable_doc.tensor
            w = batchable_doc.words
            b = batchable_doc.boxes
            self.logger.info(f"Batchable Doc : {frame.shape}")

            # Perform inference
            self.logger.info(f"Performing inference")
            # convert tensor to PIL image
            z = t.astype(np.uint8)  # .transpose(1, 2, 0)
            frame = Image.fromarray(z)
            r = self.inference(
                image=frame, words=w, boxes=b, labels=labels, threshold=0.5
            )

        results = DocList[MarieDoc]()
        return results

    def inference(
        self,
        image: Any,
        words: List[Any],
        boxes: List[Any],
        labels: List[str],
        threshold: float,
    ) -> Tuple[List, List, List]:
        """Run Inference on the model with given processor"""
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
            probs = (
                nn.softmax(outputs.logits.squeeze(), dim=1).max(dim=1).values.tolist()
            )
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

        assert len(out_prediction) == len(words)
        return out_prediction, out_boxes, out_scores

    def align_predictions(
        self,
        words,
        original_boxes: [],
        out_prediction: [],
        out_boxes: [],
        out_scores: [],
    ):
        """Align predictions with words and boxes"""

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
