import os
import time
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from docarray import DocList
from PIL import Image
from torch.nn import Module
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from marie.constants import __marie_home__, __model_path__
from marie.executor.ner.utils import normalize_bbox, visualize_prediction
from marie.logging_core.logger import MarieLogger
from marie.models.utils import initialize_device_settings

from ...api.docs import BatchableMarieDoc, MarieDoc
from ...ocr import OcrEngine
from ...registry.model_registry import ModelRegistry
from ...utils.docs import convert_frames, frames_from_docs
from ...utils.image_utils import hash_frames_fast
from ...utils.json import load_json_file, store_json_object
from ...utils.utils import ensure_exists
from .base import BaseDocumentIndexer

MAX_SOURCE_LENGTH = 1024
MAX_TARGET_LENGTH = 512


class TransformersSeq2SeqDocumentIndexer(BaseDocumentIndexer):
    """
    Transformer based model for relation extraction. This model is based on the following paper:
    https://arxiv.org/pdf/2305.05003
    """

    def __init__(
        self,
        model_name_or_path: Union[str, os.PathLike],
        model_version: Optional[str] = None,
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
        self.logger.info(f"Document indexer Seq2Seq: {model_name_or_path}")

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

        self.model, self.tokenizer = self.__load_model(
            model_name_or_path, self.labels, self.device.type
        )
        self.model = self.optimize_model(self.model)

    def __load_model(
        self, model_name_or_path: str, labels: list[str], device: str
    ) -> Tuple[Module, Callable]:
        """
        Create token classification model
        """
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)

        tokenizer.model_max_length = MAX_SOURCE_LENGTH
        model.to(device)
        model.eval()

        return model, tokenizer

    def optimize_model(self, model: nn.Module) -> Callable | Module:
        """Optimizes the model for inference. This method is called by the __init__ method."""
        # TODO: Implement model optimization

        return model

    def get_label_info(self, labels: list[str]):
        self.logger.debug(f"Labels : {labels}")

        id2label = {v: k for v, k in enumerate(labels)}
        label2id = {k: v for v, k in enumerate(labels)}

        return labels, id2label, label2id

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
        **kwargs,
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

        texts = []
        if "prompts" in kwargs:
            prompts = kwargs.pop("prompts")
        else:
            # TODO : we should create a prompt text from words and boxes, but for now we will use the provided
            raise ValueError(
                "At this time prompts must be provided for sequence classification"
            )

        frames = frames_from_docs(documents)
        file_hash = hash_frames_fast(frames)
        frames, words, boxes_normal = self.preprocess(frames, words, boxes)

        batchable_docs = DocList(
            BatchableMarieDoc(
                tensor=doc.tensor,
                words=word,
                boxes=box,
                text=text,
            )
            for doc, word, box, text in zip(documents, words, boxes_normal, prompts)
        )
        annotations = []

        for k, (batchable_doc, _image) in enumerate(zip(batchable_docs, frames)):
            _t = batchable_doc.tensor
            w = batchable_doc.words
            b = batchable_doc.boxes
            t = batchable_doc.text

            width = _image.size[0]
            height = _image.size[1]

            true_predictions = self.inference(
                image=_image,
                words=w,
                boxes=b,
                text=t,
                labels=self.labels,
                threshold=0.5,
            )

            annotation = {
                "meta": {"imageSize": {"width": width, "height": height}, "page": k},
                "predictions": true_predictions,
                # "scores": true_scores,
            }
            annotations.append(annotation)

        for k, (document, annotation) in enumerate(zip(documents, annotations)):
            if self.task == "transformers-document-indexer":
                filtered_kv = annotation["predictions"]
                document.tags["indexer"] = {
                    "page": k,
                    "kv": filtered_kv,
                }
            else:
                raise ValueError(f"Unsupported task: {self.task}")
        return documents

    @torch.no_grad()
    def inference(
        self,
        image: Any,
        words: List[Any],
        boxes: List[Any],
        text: str,
        labels: List[str],
        threshold: float,
    ) -> Tuple[List, List, List]:
        """Run Inference

        :param image: The image to be processed. This can be a PIL.Image or numpy.
        :param words: The words to be processed.
        :param boxes: The boxes to be processed, in the format (x, y, w, h). Boxes should be normalized to 1000
        :param text: The text to be processed(prompt for LLM).
        :param labels: The labels to be used for inference.
        :param threshold: The threshold to be used for filtering the results.
        :returns: The predictions, boxes (normalized), and scores.
        """
        self.logger.info(f"Performing inference")
        input_texts = [text]
        num_beams = 6
        max_new_tokens = 512
        print(f"Estimated number of tokens: {max_new_tokens}")

        start_time = time.time()
        predictions = self.generate_predictions(
            input_texts, max_new_tokens=max_new_tokens, num_beams=num_beams
        )
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Time taken: {elapsed_time:.4f} seconds")

        for i, (inp, pred) in enumerate(zip(input_texts, predictions)):
            print(f"Input {i + 1}: {inp}")
            print(f"Prediction {i + 1}: {pred}")
            print("-" * 50)

            parsed_output = self.parse_key_value_pairs(pred)
            for key, value in parsed_output.items():
                print(f"{key:<25} -   {value}")

            return parsed_output

    @torch.no_grad()
    def generate_predictions(self, inputs, max_new_tokens=256, num_beams=5):
        """
        Generate predictions for a list of input texts.

        Args:
            inputs (list): A list of strings (input texts for generation).
            max_new_tokens (int): Maximum number of new tokens to generate.
            num_beams (int): Number of beams for beam search.

        Returns:
            list: A list of model-generated outputs as decoded strings (post-processed).
        """

        model = self.model
        tokenizer = self.tokenizer
        device = self.device

        tokenized_inputs = tokenizer(
            inputs,
            max_length=MAX_SOURCE_LENGTH,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )

        tokenized_inputs = {
            key: value.to(device) for key, value in tokenized_inputs.items()
        }

        with torch.no_grad():
            outputs = model.generate(
                input_ids=tokenized_inputs["input_ids"],
                attention_mask=tokenized_inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,  # Explore more candidate sequences
                length_penalty=2,  # Increase the length penalty to reduce short outputs
                do_sample=False,  # Use deterministic beam search (no random sampling)
                early_stopping=True,  # Stop generation when EOS token is generated
            )

        decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        decoded_preds = [pred.strip() for pred in decoded_preds]

        return decoded_preds

    def parse_key_value_pairs(self, input_text):
        """
        Parses the input text into key-value pairs based on the defined separators.

        Args:
            input_text (str): The input text containing keys and values.

        Returns:
            dict: A dictionary of parsed key-value pairs.
        """
        # Dictionary to store parsed key-value pairs
        parsed_data = {}

        segments = input_text.split('>>>')  # First split by '>>>'
        segments = [
            s.strip() for parts in segments for s in parts.split('>>')
        ]  # Then split remaining parts by '>>'

        for segment in segments:
            if "KEY_FOR" in segment:  # Only process lines with key-value separator
                key, value = segment.split("KEY_FOR", 1)
                parsed_data[key.strip()] = value.strip()
        return parsed_data

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

        for i, (_boxes, _words, annotation, frame) in enumerate(
            zip(boxes, words, annotations, frames)
        ):
            self.logger.info(f"Processing page # {i}")
            frame_box = frame
            if isinstance(frame, Image.Image):
                frame_box = np.array(frame)

        aggregated_meta = []
        aggregated_kv = []
        aggregated_ner = []
        aggregated_groups = []

        results = {
            "meta": aggregated_meta,
            "kv": aggregated_kv,
            "ner": aggregated_ner,
            "groups": aggregated_groups,
        }
        self.logger.debug(f" results : {results}")

        # # visualize results per page
        # if self.debug_visuals and self.debug_visuals_ner:
        #     for k in range(0, len(frames)):
        #         output_filename = f"/tmp/tensors/ner_{file_hash}_{k}.png"
        #         items = []
        #         items.extend([row for row in aggregated_kv if int(row["page"]) == k])
        #         items.extend([row for row in aggregated_ner if int(row["page"]) == k])
        #         items.extend(
        #             [row for row in aggregated_groups if int(row["page"]) == k]
        #         )
        #
        #         visualize_extract_kv(output_filename, frames[k], items)

        return results
