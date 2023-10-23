import os
from typing import Optional, Union, List, Callable, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from marie import DocumentArray
from torch.nn import Module
from tqdm import tqdm
from transformers import (
    pipeline,
    LayoutLMv3Processor,
    LayoutLMv3ImageProcessor,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from marie.constants import __model_path__

from marie.logging.logger import MarieLogger
from marie.models.utils import initialize_device_settings
from .base import BaseDocumentSplitter
from ...helper import batch_iterator
from ...logging.profile import TimeContext
from ...registry.model_registry import ModelRegistry


def scale_bounding_box(
    box: List[int], width_scale: float = 1.0, height_scale: float = 1.0
) -> List[int]:
    return [
        int(box[0] * width_scale),
        int(box[1] * height_scale),
        int(box[2] * width_scale),
        int(box[3] * height_scale),
    ]


class TransformersDocumentSplitter(BaseDocumentSplitter):
    """
    Transformer based model for document splitting using the HuggingFace's transformers framework
    (https://github.com/huggingface/transformers).
    """

    def __init__(
        self,
        model_name_or_path: Union[str, os.PathLike],
        model_version: Optional[str] = None,
        tokenizer: Optional[str] = None,
        use_gpu: bool = True,
        labels: Optional[List[str]] = None,
        batch_size: int = 16,
        use_auth_token: Optional[Union[str, bool]] = None,
        devices: Optional[List[Union[str, "torch.device"]]] = None,
        show_error: Optional[Union[str, bool]] = True,
        **kwargs,
    ):
        """
        Load a document splitting model from ModelRepository or HuggingFace model hub.

        TODO: ADD EXAMPLE AND CODE SNIPPET

        :param model_name_or_path: Directory of a saved model or the name of a public model  from the HuggingFace model hub.
        See https://huggingface.co/models for full list of available models.
        :param model_version: The version of model to use from the HuggingFace model hub. Can be tag name, branch name, or commit hash.
        :param tokenizer: Name of the tokenizer (usually the same as model)
        :param use_gpu: Whether to use GPU (if available).
        :param batch_size: Number of Documents to be processed at a time.
        :param use_auth_token: The API token used to download private models from Huggingface.
                               If this parameter is set to `True`, then the token generated when running
                               `transformers-cli login` (stored in ~/.huggingface) will be used.
                               Additional information can be found here
                               https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained
        :param devices: List of torch devices (e.g. cuda, cpu, mps) to limit inference to specific devices.
                        A list containing torch device objects and/or strings is supported (For example
                        [torch.device('cuda:0'), "mps", "cuda:1"]). When specifying `use_gpu=False` the devices
                        parameter is not used and a single cpu device is used for inference.
        """
        super().__init__(**kwargs)
        self.logger = MarieLogger(self.__class__.__name__).logger
        self.logger.info(f"Document splitter : {model_name_or_path}")
        self.show_error = show_error  # show prediction errors
        self.batch_size = batch_size
        self.labels = labels
        self.progress_bar = False

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

        if tokenizer is None:
            tokenizer = model_name_or_path

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path
        )
        self.model = self.model.eval().to(resolved_devices[0])
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        feature_extractor = LayoutLMv3ImageProcessor(
            apply_ocr=False, do_resize=True, resample=Image.LANCZOS
        )
        self.processor = LayoutLMv3Processor(
            feature_extractor, tokenizer=self.tokenizer
        )

        if False:
            self.model = self.optimize_model(self.model)

    def predict(
        self,
        documents: DocumentArray,
        words: Optional[List[List[str]]] = None,
        boxes: Optional[List[List[List[int]]]] = None,
        batch_size: Optional[int] = None,
    ) -> DocumentArray:

        if batch_size is None:
            batch_size = self.batch_size

        if len(documents) == 0:
            return documents

        assert (
            words is not None and boxes is not None
        ), "words and boxes must be provided for sequence classification"
        assert (
            len(documents) == len(words) == len(boxes)
        ), "documents, words and boxes must have the same length"

        batches = batch_iterator(documents, batch_size)
        predictions = []
        pb = tqdm(
            total=len(documents),
            disable=not self.progress_bar,
            desc="Classifying documents",
        )

        for batch in batches:
            batch_results = []

            for doc, w, b in zip(batch, words, boxes):
                if doc.content_type != "tensor":
                    raise ValueError(
                        f"Document content_type {doc.content_type} is not supported"
                    )
                batch_results.append(
                    self.predict_document_image(
                        doc.tensor, words=w, boxes=b, top_k=self.top_k
                    )
                )

            predictions.extend(batch_results)
            pb.update(len(batch))
        pb.close()

        for document, prediction in zip(documents, predictions):
            formatted_prediction = {
                "label": prediction[0]["label"],
                "score": prediction[0]["score"],
                "details": {el["label"]: el["score"] for el in prediction},
            }
            document.tags["split"] = formatted_prediction
        return documents

    def predict_document_image(
        self,
        image: np.ndarray,
        words: List[List[str]],
        boxes: List[List[int]],
        top_k: int = 1,
    ) -> list[dict[str, Any]]:
        """
        Predicts the label of a document image

        :param image: image to predict
        :param words: words in the image
        :param boxes: bounding boxes of the words
        :param top_k: number of predictions to return
        :return: prediction dictionary with label and score
        """
        id2label = self.model.config.id2label
        width, height = image.shape[1], image.shape[0]
        width_scale = 1000 / width
        height_scale = 1000 / height
        boxes_normalized = [
            scale_bounding_box(box, width_scale, height_scale) for box in boxes
        ]

        encoding = self.processor(
            image,
            words,
            boxes=boxes_normalized,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        with torch.inference_mode():
            output = self.model(
                input_ids=encoding["input_ids"].to(self.device),
                attention_mask=encoding["attention_mask"].to(self.device),
                bbox=encoding["bbox"].to(self.device),
                pixel_values=encoding["pixel_values"].to(self.device),
            )
        # TODO: add top_k support

        logits = output.logits
        predicted_class = logits.argmax(-1)
        probabilities = F.softmax(logits, dim=-1).squeeze().tolist()

        return [
            {
                "label": id2label[predicted_class.item()],
                "score": probabilities[predicted_class.item()],
            }
        ]

    def optimize_model(self, model: nn.Module) -> Callable | Module:
        """Optimizes the model for inference. This method is called by the __init__ method."""
        try:
            with TimeContext("Compiling model", logger=self.logger):
                import torchvision.models as models
                import torch._dynamo as dynamo

                torch._dynamo.config.verbose = False
                torch._dynamo.config.suppress_errors = True
                # torch.backends.cudnn.benchmark = True
                # model = torch.compile(model, backend="inductor", mode="max-autotune")
                # model = torch.compile(model, backend="onnxrt", fullgraph=False)
                model = torch.compile(model)
                return model
        except Exception as err:
            self.logger.warning(f"Model compile not supported: {err}")
            return model
