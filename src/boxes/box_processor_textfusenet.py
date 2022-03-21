
import argparse
import copy
import glob
import os
import time
import cv2
import numpy as np
import torch
from reportlab.lib.utils import ImageReader
import io

from models.textfusenet.detectron2.config import get_cfg
from models.textfusenet.detectron2.structures import Boxes, RotatedBoxes
from models.textfusenet.detectron2.utils.colormap import random_color

from models.textfusenet.detectron2.engine.defaults import DefaultPredictor
from PIL import Image

from boxes.box_processor import BoxProcessor, PSMode


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    cfg.MODEL.DEVICE = 'cpu'
    cfg.TEST.DETECTIONS_PER_IMAGE = 5000

    # Set model
    cfg.MODEL.WEIGHTS = args.weights
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg



def get_parser():
    parser = argparse.ArgumentParser(description="SynthDetection")
    parser.add_argument(
        "--config-file",
        default="./config/ocr/synthtext_pretrain_101_FPN.yaml",
        metavar="FILE",
        help="path to config file",
    )

    parser.add_argument(
        "--weights",
        default="./models/textfusenet/model_0153599.pth",
        metavar="pth",
        help="the model used to inference",
    )

    parser.add_argument(
        "--input",
        default="./assets/private/*.*",
        nargs="+",
        help="the folder of totaltext test images"
    )

    parser.add_argument(
        "--output",
        default="./test_synth/",
        help="A file or directory to save output visualizations. "
             "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.6,
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
        models_dir: str = "./models/textfusenet",
        cuda: bool = False,
    ):
        super().__init__(work_dir, models_dir, cuda)
        print("Box processor [textfusenet, cuda={}]".format(cuda))
        args = get_parser().parse_args()
        cfg = setup_cfg(args)

        self.predictor = DefaultPredictor(cfg)
        self.cpu_device = torch.device("cpu")

    def extract_bounding_boxes(self, _id, key, img, psm=PSMode.SPARSE):

        start_time = time.time()
        predictions = self.predictor(img)
        instances = predictions["instances"].to(self.cpu_device)
        print(f"Number of prediction : {len(instances)}")
        predictions = instances

        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
        scores = predictions.scores if predictions.has("scores") else None
        classes = predictions.pred_classes if predictions.has("pred_classes") else None
        boxes = _convert_boxes(boxes)

        prediction_result = dict()
        prediction_result["bboxes"] = boxes
        prediction_result["polys"] = boxes
        prediction_result["heatmap"] = None

        print(boxes.shape)

        # deepcopy image so that original is not altered
        image = copy.deepcopy(img)
        pil_image = Image.new("RGB", (image.shape[1], image.shape[0]), color=(0, 255, 0, 0))

        rect_from_poly = []
        rect_line_numbers = []
        fragments = []
        ms = int(time.time() * 1000)

        max_h = image.shape[0]
        max_w = image.shape[1]

        for i in range(len(boxes)):
            box = np.array(boxes[i]).astype(np.int32)
            x0, y0, x1, y1 = box
            w = x1 - x0
            h = y1 - y0

            if classes[i] == 0:
                print(box)
                snippet = image[y0:y0+h, x0:x0+w:]
                # export cropped region
                file_path = os.path.join('./result', "snippet_%s.jpg" % (i))
                cv2.imwrite(file_path, snippet)

        # we can't return np.array here as t the 'fragments' will throw an error
        # ValueError: could not broadcast input array from shape (42,77,3) into shape (42,)
        return rect_from_poly, fragments, rect_line_numbers, prediction_result