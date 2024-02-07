import os
import traceback
from typing import Any, Callable, List, Optional, Sequence, Union

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
from marie.logging.logger import MarieLogger
from marie.models.utils import initialize_device_settings

from ...api.docs import BatchableMarieDoc, MarieDoc
from ...helper import batch_iterator
from ...logging.profile import TimeContext
from ...registry.model_registry import ModelRegistry
from ...utils.json import load_json_file
from ...utils.utils import ensure_exists
from ..document_classifier.base import BaseDocumentClassifier
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
        labels: Optional[List[str]] = None,
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

        labels = self.init_configuration["labels"]

        self.model = self.__load_model(model_name_or_path, labels, self.device)
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

        if self.task == "text-classification-multimodal":
            assert (
                words is not None and boxes is not None
            ), "words and boxes must be provided for sequence classification"
            assert (
                len(documents) == len(words) == len(boxes)
            ), "documents, words and boxes must have the same length"

        results = DocList[MarieDoc]()

        return results
