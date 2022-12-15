import argparse
import copy
import os
import time

import cv2
import numpy as np
import torch
from PIL import Image

from marie.boxes.box_processor import BoxProcessor, PSMode
from marie.logger import setup_logger
from marie.models.textfusenet.detectron2.config import get_cfg
from marie.models.textfusenet.detectron2.engine.defaults import DefaultPredictor
from marie.models.textfusenet.detectron2.structures import Boxes, RotatedBoxes
from marie.models.textfusenet.detectron2.utils.colormap import random_color
from marie.utils.image_utils import imwrite, paste_fragment
from marie.utils.utils import ensure_exists

logger = setup_logger(__name__)


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    os.environ["OMP_NUM_THREADS"] = str(18)
    os.environ["KMP_BLOCKTIME"] = str(1)

    cfg.MODEL.DEVICE = "cpu"
    cfg.TEST.DETECTIONS_PER_IMAGE = 2000

    print(args.confidence_threshold)
    # Set model
    cfg.MODEL.WEIGHTS = args.weights
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = (
        args.confidence_threshold
    )
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="SynthDetection")
    parser.add_argument(
        "--config-file",
        default="./config/zoo/textfusenet/synthtext_pretrain_101_FPN.yaml",
        metavar="FILE",
        help="path to config file",
    )

    parser.add_argument(
        "--weights",
        default="./model_zoo/textfusenet/model_0153599.pth",
        metavar="pth",
        help="the model used to inference",
    )

    parser.add_argument(
        "--input",
        default="./assets/private/*.*",
        nargs="+",
        help="the folder of totaltext test images",
    )

    parser.add_argument(
        "--output",
        default="./test_synth/",
        help="A file or directory to save output visualizations. If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def _convert_boxes(boxes):
    """
    Convert different format of boxes to an NxB array, where B = 4 or 5 is the box dimension.
    """
    # FIXME : throw  only one element tensors can be converted to Python scalars
    if True:
        return boxes.tensor.numpy()

    if isinstance(boxes, Boxes) or isinstance(boxes, RotatedBoxes):
        return boxes.tensor.numpy()
    else:
        return np.asarray(boxes)


class BoxProcessorTextFuseNet(BoxProcessor):
    """ "TextFuseNet box processor responsible for extracting bounding boxes for given documents"""

    def __init__(
        self,
        work_dir: str = "/tmp/boxes",
        models_dir: str = "./model_zoo/textfusenet",
        cuda: bool = False,
    ):
        super().__init__(work_dir, models_dir, cuda)
        logger.info("Box processor [textfusenet, cuda={}]".format(cuda))
        args = get_parser().parse_args()
        cfg = setup_cfg(args)
        self.predictor = DefaultPredictor(cfg)
        self.cpu_device = torch.device("cpu")

    def extract_bounding_boxes(self, _id, key, src_img, psm=PSMode.SPARSE):

        start_time = time.time()
        debug_dir = ensure_exists(
            os.path.join(self.work_dir, _id, "bounding_boxes", key, "debug")
        )
        crops_dir = ensure_exists(
            os.path.join(self.work_dir, _id, "bounding_boxes", key, "crop")
        )

        # deepcopy image so that original is not altered
        img = copy.deepcopy(src_img)

        predictions = self.predictor(img)
        predictions = predictions["instances"].to(self.cpu_device)
        logger.info(f"Number of prediction : {len(predictions)}")

        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
        scores = predictions.scores if predictions.has("scores") else None
        classes = predictions.pred_classes if predictions.has("pred_classes") else None
        boxes = _convert_boxes(boxes)

        prediction_result = dict()
        prediction_result["bboxes"] = boxes
        prediction_result["polys"] = boxes
        prediction_result["scores"] = scores
        prediction_result["heatmap"] = None
        pil_image = Image.new("RGB", (img.shape[1], img.shape[0]), color=(0, 255, 0, 0))

        rect_from_poly = []
        rect_line_numbers = []
        fragments = []

        max_h = img.shape[0]
        max_w = img.shape[1]

        for i in range(len(boxes)):
            box = np.array(boxes[i]).astype(np.int32)
            x0, y0, x1, y1 = box
            w = x1 - x0
            h = y1 - y0
            box_adj = [x0, y0, w, h]

            # Class 0 == Text
            if classes[i] == 0:
                snippet = img[y0 : y0 + h, x0 : x0 + w :]
                # export cropped region
                file_path = os.path.join("./result", "snippet_%s.jpg" % i)
                cv2.imwrite(file_path, snippet)

                fragments.append(snippet)
                rect_from_poly.append(box_adj)
                rect_line_numbers.append(0)

                # After normalization image is in 0-1 range
                # snippet = (snippet * 255).astype(np.uint8)
                paste_fragment(pil_image, snippet, (x0, y0))

        savepath = os.path.join(debug_dir, "txt_overlay.jpg")
        pil_image.save(savepath, format="JPEG", subsampling=0, quality=100)

        cv_img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        stacked = np.hstack((cv_img, img))
        save_path = os.path.join(debug_dir, "stacked.png")
        imwrite(save_path, stacked)

        stop_time = time.time()
        eval_time = round((stop_time - start_time) * 1000, 2)

        # metrics.add_time('HandlerTime', round(
        #     (stop_time - start_time) * 1000, 2), None, 'ms')

        # we can't return np.array here as t the 'fragments' will throw an error
        # ValueError: could not broadcast input array from shape (42,77,3) into shape (42,)
        return rect_from_poly, fragments, rect_line_numbers, prediction_result
