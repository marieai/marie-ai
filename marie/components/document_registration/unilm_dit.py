import argparse
import os
from typing import Any, Callable, List, Optional, Union

import detectron2.data.transforms as T
import numpy as np
import torch
import torch.nn as nn
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.modeling import build_model
from ditod import add_vit_config
from PIL import Image
from torch.nn import Module
from tqdm import tqdm

from marie import DocumentArray
from marie.constants import __config_dir__, __model_path__
from marie.logging.logger import MarieLogger
from marie.models.utils import initialize_device_settings
from marie.utils.types import strtobool

from ...detectron.detector import OptimizedDetectronPredictor
from ...helper import batch_iterator
from ...logging.profile import TimeContext
from ...registry.model_registry import ModelRegistry
from ..util import scale_bounding_box
from .base import BaseDocumentBoundaryRegistration


def setup_cfg(args, device):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_vit_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # set device
    cfg.MODEL.DEVICE = device
    cfg.MODEL.WEIGHTS = os.path.join(__model_path__, cfg.MODEL.WEIGHTS)
    cfg.freeze()
    # default_setup(cfg, args)
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(
        description="DIT Object detection inference script"
    )
    parser.add_argument(
        "--config-file",
        default="./config/zoo/unilm/dit/object_detection/document_boundary/mask_rcnn_dit_base.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )

    return parser


class UnilmDocumentBoundaryRegistration(BaseDocumentBoundaryRegistration):
    """
    Document boundary registration using the Large-scale Self-supervised Pre-training Across Tasks, Languages, and Modalities (Unilm)
    models from Microsoft Research. https://github.com/microsoft/unilm/tree/master/dit/object_detection

    EXAMPLE USAGE

    .. code-block:: python

        from marie.boxes import BoxProcessorUlimDit
        from marie.boxes.box_processor import PSMode

        box = BoxProcessorUlimDit(
            models_dir="../../model_zoo/unilm/dit/text_detection",
            cuda=True,
        )
        (
            boxes,
            fragments,
            lines,
            _,
            lines_bboxes,
        ) = box.extract_bounding_boxes("gradio", "field", image, PSMode.SPARSE)

        bboxes_img = visualize_bboxes(image, boxes, format="xywh")
        lines_img = visualize_bboxes(image, lines_bboxes, format="xywh")
    """

    def __init__(
        self,
        model_name_or_path: Union[str, os.PathLike],
        model_version: Optional[str] = None,
        use_gpu: bool = True,
        batch_size: int = 16,
        devices: Optional[List[Union[str, "torch.device"]]] = None,
        show_error: Optional[Union[str, bool]] = True,
        **kwargs,
    ):
        """
        Load a Unilm model for document boundary registration.

        TODO: ADD EXAMPLE AND CODE SNIPPET

        :param model_name_or_path: Directory of a saved model or the name of a public model from the HuggingFace model hub.
        See https://huggingface.co/models for full list of available models.
        :param model_version: The version of model to use from the HuggingFace model hub. Can be tag name, branch name, or commit hash.
        :param use_gpu: Whether to use GPU (if available).
        :param batch_size: Number of Documents to be processed at a time.
        :param devices: List of torch devices (e.g. cuda, cpu, mps) to limit inference to specific devices.
                        A list containing torch device objects and/or strings is supported (For example
                        [torch.device('cuda:0'), "mps", "cuda:1"]). When specifying `use_gpu=False` the devices
                        parameter is not used and a single cpu device is used for inference.
        """
        super().__init__(**kwargs)

        self.logger = MarieLogger(self.__class__.__name__).logger
        self.logger.info(f"Document registration : {model_name_or_path}")
        self.show_error = show_error  # show prediction errors
        self.batch_size = batch_size
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
        self.logger.info(f"Loading model from config dir {__config_dir__}")

        args = get_parser().parse_args(
            [
                "--config-file",
                os.path.join(
                    __config_dir__,
                    "zoo/unilm/dit/object_detection/document_boundary/prod.yaml",
                ),
            ]
        )

        self.logger.info(f"Loading model from {args.config_file}")

        cfg = setup_cfg(args, self.device)
        self.min_size_test = [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST]
        # self.predictor = DefaultPredictor(cfg)
        self.predictor = OptimizedDetectronPredictor(
            cfg, half_precision=True
        )  # TODO: half_precision=True should be from config
        self.cpu_device = torch.device("cpu")

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

        if words is None or boxes is None:
            words = [None] * len(documents)
            boxes = [None] * len(documents)

        batches = batch_iterator(documents, batch_size)
        predictions = []
        pb = tqdm(
            total=len(documents),
            disable=not self.progress_bar,
            desc="Segmenting documents",
        )

        for batch in batches:
            batch_results = []

            for doc, w, b in zip(batch, words, boxes):
                if doc.content_type != "tensor":
                    raise ValueError(
                        f"Document content_type {doc.content_type} is not supported"
                    )
                batch_results.append(
                    self.predict_document_image(doc.tensor, words=w, boxes=b, top_k=1)
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
        Predicts the label of a document image.

        :param image: image to predict on
        :param words: words in the image
        :param boxes: bounding boxes of the words
        :param top_k: number of predictions to return
        :return: prediction dictionary with label and score
        """
        id2label = self.model.config.id2label
        width, height = image.shape[1], image.shape[0]

        with torch.no_grad():
            image = Image.fromarray(image)
            print(f"Inference on image: {image.size}")
            output = self.predictor(image)["instances"]
            print(f"Output: {output}")
            md = None

        return [
            {
                "label": "XX",
                "score": 0.0,
            }
        ]
