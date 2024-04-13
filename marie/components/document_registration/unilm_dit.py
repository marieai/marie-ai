import argparse
import os
from typing import List, Optional, Union

import cv2
import numpy as np
import torch
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode, Visualizer
from ditod import add_vit_config
from docarray import DocList
from tqdm import tqdm

from marie import DocumentArray
from marie.constants import __config_dir__, __model_path__
from marie.logging.logger import MarieLogger
from marie.models.utils import initialize_device_settings

from ...detectron.detector import OptimizedDetectronPredictor
from ...helper import batch_iterator
from ...utils.image_utils import hash_frames_fast
from .base import BaseDocumentBoundaryRegistration
from .datamodel import DocumentBoundaryPrediction


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
        default="zoo/unilm/dit/object_detection/document_boundary/prod.yaml",
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

        cfg = setup_cfg(args, self.device.type)
        self.min_size_test = [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST]
        # self.predictor = DefaultPredictor(cfg)
        self.predictor = OptimizedDetectronPredictor(
            cfg, half_precision=True
        )  # TODO: half_precision=True should be from config
        self.cpu_device = torch.device("cpu")

    def predict(
        self,
        documents: DocumentArray,
        registration_method: str,  # absolute or fit_to_page
        words: Optional[List[List[str]]] = None,
        boxes: Optional[List[List[List[int]]]] = None,
        batch_size: Optional[int] = None,
    ) -> DocList:

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

        idx = 0
        for batch in batches:
            batch_results = []

            for doc, w, b in zip(batch, words, boxes):
                print(f"Processing document : {idx}")
                batch_results.append(
                    self.predict_document_image(
                        doc.tensor,
                        registration_method,
                        words=w,
                        boxes=b,
                        top_k=1,
                        doc_id=idx,
                    )
                )
                idx += 1

            predictions.extend(batch_results)
            pb.update(len(batch))
        pb.close()

        for document, prediction in zip(documents, predictions):
            if len(prediction) == 0:
                self.logger.warning(f"No segmentation boxes predicted.")
                continue
            document.tags["document_boundary"] = prediction[0]
        return documents

    def predict_document_image(
        self,
        image: np.ndarray,
        registration_mode: str,  # absolute or fit_to_page
        words: List[List[str]],
        boxes: List[List[int]],
        top_k: int = 1,
        doc_id: str = None,
    ) -> list[DocumentBoundaryPrediction]:
        """
        Predicts the label of a document image.

        :param image: image to predict on
        :param words: words in the image
        :param boxes: bounding boxes of the words
        :param registration_mode: absolute or fit_to_page
        :param top_k: number of predictions to return
        :return: prediction dictionary with label and score
        """
        if doc_id is None:
            doc_id = hash_frames_fast([image])

        width, height = image.shape[1], image.shape[0]

        with torch.no_grad():
            self.logger.info(f"Inference on image: {image.shape}")
            rp = self.predictor(image)
            # detach the predictions from GPU to avoid memory leak
            predictions = rp["instances"]
            # Following will hang if we move the predictions from GPU to CPU all at once
            # This is a workaround to avoid the hanging
            # predictions = predictions.to(self.cpu_device)
            boxes = (
                predictions.pred_boxes.to(self.cpu_device)
                if predictions.has("pred_boxes")
                else None
            )
            scores = (
                predictions.scores.to(self.cpu_device)
                if predictions.has("scores")
                else None
            )
            classes = (
                predictions.pred_classes.to(self.cpu_device)
                if predictions.has("pred_classes")
                else None
            )

            md = None
            v = Visualizer(
                image[:, :, ::-1], md, scale=1.0, instance_mode=ColorMode.SEGMENTATION
            )
            result = v.draw_instance_predictions(predictions.to("cpu"))
            result_image = result.get_image()[:, :, ::-1]

            output_filename = f"/tmp/dit/result_visualizer.png"
            cv2.imwrite(output_filename, result_image)

            # delete the predictor to free up memory
            del predictions
            del rp

            # check if boxes are empty, which means no boxes were detected(blank image)
            if boxes is None or scores is None or classes is None:
                self.logger.warning(f"No segmentation boxes predicted.")
                return []

            boxes = boxes.tensor.numpy()
            scores = scores.numpy()
            classes = classes.numpy()

            default_prediction = DocumentBoundaryPrediction(
                label="document",
                detected=False,
                mode=registration_mode,
                aligned_image=None,
                boundary_bbox=[0, 0, 0, 0],
                score=0,
            )

            if len(boxes) == 0:
                self.logger.warning(f"No segmentation boxes predicted.")
                return [default_prediction]

            if len(boxes) > 1:
                self.logger.warning(f"Multiple boxes detected, skipping segmentation.")
                return [default_prediction]

            boundary_bbox = [int(x) for x in boxes[0]]  # xyxy format
            # TODO : add this as a parameter
            # TODO : Add more advanced logic to adjust the boundary_bbox
            x0, y0, x1, y1 = boundary_bbox
            w = x1 - x0
            h = y1 - y0
            x = x0
            y = y0

            margin_width = 5
            margin_height = 5

            p1 = [5, 5]
            p1_x, p1_y = p1

            boundary_bbox = [
                max(0, x - margin_width),
                max(0, y - margin_height),
                min(width, w + margin_width * 2),
                min(height, h + margin_height * 2),
            ]

            print(f"Boundary bbox: {boundary_bbox}")

            score = scores[0]
            label = classes[0]  # there is only one class in this model

            aligned_image = np.ones((height, width, 3), dtype=np.uint8) * 255
            boundary = image[
                boundary_bbox[1] : boundary_bbox[1] + boundary_bbox[3],
                boundary_bbox[0] : boundary_bbox[0] + boundary_bbox[2],
            ]
            cv2.imwrite(f"/tmp/dit/{doc_id}_boundary_image.png", boundary)

            # absolute registration
            if registration_mode == "absolute":
                if p1_x + boundary_bbox[2] > width:
                    self.logger.warning(
                        f"Offset x1 + boundary_bbox width is out of bounds."
                    )
                    return [default_prediction]
                if p1_y + boundary_bbox[3] > height:
                    self.logger.warning(
                        f"Offset y1 + boundary_bbox height is out of bounds."
                    )
                    return [default_prediction]

                aligned_image[
                    p1_y : p1_y + boundary.shape[0], p1_x : p1_x + boundary.shape[1]
                ] = boundary
                cv2.circle(aligned_image, (p1_x, p1_y), 8, (0, 0, 255), -1)

            elif registration_mode == "fit_to_page":
                new_width = width - p1_x * 2
                scale_factor = 1
                resized_boundary = boundary

                if boundary_bbox[3] > boundary_bbox[2]:
                    original_height, original_width = image.shape[:2]
                    scale_factor = boundary_bbox[3] / boundary_bbox[2]
                    new_height = int(original_height * scale_factor)
                    resized_boundary = cv2.resize(
                        boundary, (new_width, new_height), interpolation=cv2.INTER_CUBIC
                    )

                # cv2.imwrite(f"/tmp/dit/{doc_id}_resized_boundary.png", resized_boundary)
                bottom = height - p1_y - h * scale_factor
                bottom = max(0, int(bottom))

                aligned_image = cv2.copyMakeBorder(
                    resized_boundary,
                    p1_y,
                    bottom,
                    p1_x,
                    p1_x,
                    cv2.BORDER_CONSTANT,
                    value=(255, 255, 255),
                )

                cv2.circle(aligned_image, (p1_x, p1_y), 8, (0, 0, 255), -1)
                cv2.circle(aligned_image, (p1_x + new_width, p1_y), 8, (0, 0, 255), -1)

            debug_visualization = True
            if debug_visualization:
                # ensure image and new image have the same width and height before stacking, debugging purposes only
                if (
                    aligned_image.shape[0] != image.shape[0]
                    or aligned_image.shape[1] != image.shape[1]
                ):
                    aligned_image = cv2.resize(
                        aligned_image,
                        (image.shape[1], image.shape[0]),
                        interpolation=cv2.INTER_CUBIC,
                    )

                divider = np.ones((aligned_image.shape[0], 5, 3), dtype=np.uint8) * 150
                stacked = np.hstack([image, divider, aligned_image])
                cv2.imwrite(f"/tmp/dit/{doc_id}_registration.png", aligned_image)
                cv2.imwrite(f"/tmp/dit/stacked_{doc_id}.png", stacked)

        return [
            DocumentBoundaryPrediction(
                label="document",
                detected=True,
                mode=registration_mode,
                aligned_image=aligned_image,
                boundary_bbox=boundary_bbox,
                score=score,
            )
        ]
