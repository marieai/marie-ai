import argparse
import copy
import glob
import io
import os
import time

import cv2
import numpy as np
import torch
from models.textfusenet.detectron2.config import get_cfg
from models.textfusenet.detectron2.engine.defaults import DefaultPredictor
from models.textfusenet.detectron2.structures import Boxes, RotatedBoxes
from models.textfusenet.detectron2.utils.colormap import random_color
from PIL import Image
from PyPDF4 import PdfFileReader, PdfFileWriter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    cfg.MODEL.DEVICE = "cpu"
    cfg.TEST.DETECTIONS_PER_IMAGE = 5000

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
        default="./config/ocr/synthtext_pretrain_101_FPN.yaml",
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


def render_boxes(img, boxes, scores, classes):
    """Render boxes"""
    print(f"size : {img.shape}")
    img_h = img.shape[0]
    img_w = img.shape[1]

    boxes = _convert_boxes(boxes)
    num_instances = len(boxes)
    assigned_colors = np.array(
        [random_color(rgb=True, maximum=255) for _ in range(num_instances)]
    )
    print(f"Number of boxes : {num_instances}")
    # Display in largest to smallest order to reduce occlusion.
    areas = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)

    sorted_idxs = np.argsort(-areas).tolist()
    # Re-order overlapped instances in descending order.
    boxes = boxes[sorted_idxs] if boxes is not None else None
    classes = classes[sorted_idxs] if classes is not None else None
    assigned_colors = assigned_colors[sorted_idxs]

    mask = np.zeros(img.shape[:2], np.uint8) * 255
    img_boxes = copy.deepcopy(img)

    # convert OpenCV to Pil
    img_bgr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img_bgr)

    packet = io.BytesIO()
    can = canvas.Canvas(packet, pagesize=(img_w, img_h))
    can.setFontSize(18)
    can.drawImage(ImageReader(im_pil), 0, 0)

    for i in range(num_instances):
        color = assigned_colors[i]
        box = np.array(boxes[i]).astype(np.int32)
        x0, y0, x1, y1 = box
        w = x1 - x0
        h = y1 - y0

        if classes[i] == 0:
            print(box)
            # color = list(np.random.random(size=3) * 256)
            cv2.rectangle(img_boxes, (x0, y0), (x0 + w, y0 + h), color.tolist(), 2)

            # PDF rendering transformation
            # x and y define the lower left corner of the image, so we need to perform some transformations
            px0 = x0
            py0 = img_h - y1
            txt = f"Word : {px0}, {py0}"
            print(txt)
            can.drawString(px0, py0, txt)

    cv2.imwrite(os.path.join("./result", "img_boxes.png"), img_boxes)

    can.save()
    # move to the beginning of the StringIO buffer
    packet.seek(0)

    return can, packet


if __name__ == "__main__":

    args = get_parser().parse_args()
    cfg = setup_cfg(args)
    test_images_path = args.input
    output_path = args.output

    predictor = DefaultPredictor(cfg)
    start_time_all = time.time()
    img_count = 0
    cpu_device = torch.device("cpu")

    pdfwriter = PdfFileWriter()

    for i in glob.glob(test_images_path):
        print(i)
        src_image_path = i
        img_name = os.path.basename(i)
        img_save_path = output_path + img_name.split(".")[0] + ".tif"
        img = cv2.imread(i)
        start_time = time.time()

        predictions = predictor(img)
        instances = predictions["instances"].to(cpu_device)
        print(f"Number of prediction : {len(instances)}")
        predictions = instances

        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
        scores = predictions.scores if predictions.has("scores") else None
        classes = predictions.pred_classes if predictions.has("pred_classes") else None

        canvas, packet = render_boxes(img, boxes, scores, classes)

        new_pdf = PdfFileReader(packet)
        page = new_pdf.getPage(0)

        pdfwriter.addPage(page)

    with open("./result/merged.pdf", "wb") as output:
        pdfwriter.write(output)
