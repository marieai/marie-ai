import argparse
import copy
import os
import time
from typing import Any, Tuple, Union

import PIL
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

from ditod import add_vit_config
from marie.boxes.box_processor import BoxProcessor, PSMode, create_dirs
from marie.boxes.line_processor import find_line_number, line_merge
from marie.constants import __model_path__, __config_dir__
from marie.logging.logger import MarieLogger
from marie.utils.image_utils import imwrite, paste_fragment
from marie.utils.overlap import merge_boxes


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
    # cfg.MODEL.WEIGHTS = "td-syn_dit-b_mrcnn.pth"
    cfg.freeze()
    # default_setup(cfg, args)
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="DIT TextBlock inference script")
    parser.add_argument(
        "--config-file",
        default="./config/zoo/unilm/dit/text_detection/mask_rcnn_dit_base.yaml",
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


def visualize_bboxes(
    image: Union[np.ndarray, PIL.Image.Image], bboxes: np.ndarray, format="xyxy"
) -> PIL.Image:
    """Visualize bounding boxes on the image
    Args:
        image(Union[np.ndarray | PIL.Image.Image]): numpy array of shape (H, W), where H is the image height and W is the image width.
        bboxes(np.ndarray): Bounding boxes for image (xmin,ymin,xmax,ymax)
        format(xyxy|xywh): format of the bboxes, defaults to `xyxy`
    """

    # convert pil to OpenCV
    if type(image) == PIL.Image.Image:
        image = np.array(image)

    viz_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    viz_img = Image.fromarray(viz_img)
    draw = ImageDraw.Draw(viz_img, "RGBA")

    for box in bboxes:
        if format == "xywh":
            box = [box[0], box[1], box[0] + box[2], box[1] + box[3]]

        draw.rectangle(
            box,
            outline="#993300",
            fill=(
                int(np.random.random() * 256),
                int(np.random.random() * 256),
                int(np.random.random() * 256),
                125,
            ),
            width=1,
        )

    # viz_img.show()
    return viz_img


def lines_from_bboxes(image, bboxes):
    """Create lines out of bboxes for given image.
    Args:
        image(ndarray): numpy array of shape (H, W), where H is the image height and W is the image width.
        bboxes: Bounding boxes for image (xmin,ymin,xmax,ymax)

    Returns:
        lines_bboxes: Bounding boxes for the lines in (x,y,w,h)
    """

    if False:
        viz_img = visualize_bboxes(image, bboxes)
        viz_img.save(
            os.path.join("/tmp/fragments", f"line_refiner_initial.png"), format="PNG"
        )

    # create a box overlay with adjusted coordinates
    overlay = np.ones((image.shape[0], image.shape[1], 1), dtype=np.uint8) * 100
    for box in bboxes:
        x1, y1, x2, y2 = box.astype(np.int32)
        w = x2 - x1
        h = (y2 - y1) // 2
        y1_adj = y1 + h // 2
        cv2.rectangle(overlay, (x1, y1_adj), (x1 + w, y1_adj + h), (0, 0, 0), -1)

    ret, link_score = cv2.threshold(overlay, 0, 255, cv2.THRESH_BINARY)

    if False:
        cv2.imwrite(os.path.join("/tmp/fragments", f"overlay_refiner-RAW.PNG"), overlay)
        cv2.imwrite(
            os.path.join("/tmp/fragments", f"overlay_refiner-link_score.PNG"),
            link_score,
        )

    # Create structure element for extracting horizontal lines through morphology operations
    cols = link_score.shape[1]
    horizontal_size = cols // 30
    horizontal_struct = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))

    # Apply morphology operations
    horizontal = np.copy(link_score)
    horizontal = cv2.erode(horizontal, horizontal_struct)
    horizontal = cv2.dilate(horizontal, horizontal_struct)

    if False:
        cv2.imwrite("/tmp/fragments/horizontal.jpg", horizontal)

    if False:
        binary_mask = np.zeros(horizontal.shape)
        binary_mask[horizontal == 255] = 0
        binary_mask[horizontal != 255] = 255
        binary_mask = binary_mask.astype(np.uint8)

    binary_mask = cv2.bitwise_not(horizontal)
    connectivity = 4

    nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask, connectivity, cv2.CV_32S
    )

    line_bboxes = []

    # 0 is background (useless)
    for k in range(1, nLabels):
        size = stats[k, cv2.CC_STAT_AREA]
        x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        # size filtering
        if h < 2 or w < 4:
            continue
        line_bboxes.append([x, y, w, h])

    # the format now will be in xywh
    lines_bboxes = line_merge(binary_mask, line_bboxes)

    if False:
        viz_img = visualize_bboxes(image, lines_bboxes, format="xywh")
        viz_img.save(
            os.path.join("/tmp/fragments", f"line_refiner-final.png"), format="PNG"
        )

    return lines_bboxes


class BoxProcessorUlimDit(BoxProcessor):
    """DiT for Text Detection"""

    def __init__(
        self,
        work_dir: str = "/tmp/boxes",
        models_dir: str = os.path.join(__model_path__, "unilm/dit/text_detection"),
        cuda: bool = False,
    ):
        super().__init__(work_dir, models_dir, cuda)
        self.logger = MarieLogger(self.__class__.__name__)
        self.logger.info("Box processor [dit, cuda={}]".format(cuda))

        args = get_parser().parse_args(
            [
                "--config-file",
                os.path.join(
                    __config_dir__,
                    "zoo/unilm/dit/text_detection/mask_rcnn_dit_base.yaml",
                ),
                "--opts",
                "MODEL.WEIGHTS",
                os.path.join(models_dir, "td-syn_dit-b_mrcnn.pth"),
            ]
        )
        self.strict_box_segmentation = False
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg = setup_cfg(args, device)
        self.predictor = DefaultPredictor(cfg)
        self.cpu_device = torch.device("cpu")

    def psm_word(self, image):
        if self.strict_box_segmentation:
            raise Exception("Not implemented : PSM_WORD")
        return self.psm_sparse(image)

    def psm_sparse(self, image: np.ndarray):
        try:
            self.logger.debug(f"Starting box predictions")
            predictions = self.predictor(image)
            predictions = predictions["instances"]
            # Following will hang if we move the predictions from GPU to CPU all at once
            # This is a workaround to avoid the hang
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
            del predictions

            bboxes = _convert_boxes(boxes)
            bboxes = merge_boxes(bboxes, 0.08)
            bboxes = np.array(bboxes)

            # sort by xy-coordinated
            ind = np.lexsort((bboxes[:, 0], bboxes[:, 1]))
            bboxes = bboxes[ind]
            lines = lines_from_bboxes(image, bboxes)

            return bboxes, classes, scores, lines, classes
        except Exception as e:
            self.logger.error(e)

    def psm_line(self, image):
        if self.strict_box_segmentation:
            raise Exception("Not implemented : PSM_LINE")
        return self.psm_sparse(image)

    def psm_raw_line(self, image):
        if self.strict_box_segmentation:
            raise Exception("Not implemented : PSM_RAW_LINE")
        return self.psm_sparse(image)

    def psm_multiline(self, image):
        if self.strict_box_segmentation:
            raise Exception("Not implemented : PSM_MULTILINE")
        return self.psm_sparse(image)

    def extract_bounding_boxes(
        self, _id, key, img, psm=PSMode.SPARSE
    ) -> Tuple[Any, Any, Any, Any, Any]:
        if img is None:
            raise Exception("Input image can't be empty")

        if type(img) == PIL.Image.Image:  # convert pil to OpenCV
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            self.logger.warning("PIL image received converting to ndarray")

        if not isinstance(img, np.ndarray):
            raise Exception("Expected image in numpy format")

        try:
            crops_dir, debug_dir, lines_dir, mask_dir = create_dirs(
                self.work_dir, _id, key
            )

            start_time = time.time()
            # deepcopy image so that original is not altered
            image = copy.deepcopy(img)
            lines_bboxes = []

            # Page Segmentation Model
            if psm == PSMode.SPARSE:
                bboxes, polys, scores, lines_bboxes, classes = self.psm_sparse(image)
            # elif psm == PSMode.WORD:
            #     bboxes, polys, scores, lines_bboxes, classes = self.psm_word(image_norm)
            elif psm == PSMode.LINE:
                bboxes, polys, scores, lines_bboxes, classes = self.psm_line(image)
            elif psm == PSMode.MULTI_LINE:
                bboxes, polys, scores, lines_bboxes, classes = self.psm_multiline(image)
            elif psm == PSMode.RAW_LINE or psm == PSMode.WORD:
                # NOTE: Semantics have changed and now both RAW_LINE and WORD are same
                # this needs to be handled better, there is no need to have the segmentation for RAW_LINES
                # as we treat the whole line as BBOX
                # bboxes, polys, score_text = self.psm_raw_line(image_norm)
                rect_from_poly = []
                fragments = []
                rect_line_numbers = []
                prediction_result = dict()

                # x, y, w, h = box
                w = image.shape[1]
                h = image.shape[0]
                rect_from_poly.append([0, 0, w, h])
                fragments.append(image)
                rect_line_numbers.append(0)

                return (
                    rect_from_poly,
                    fragments,
                    rect_line_numbers,
                    prediction_result,
                    lines_bboxes,
                )
            else:
                raise Exception(f"PSM mode not supported : {psm}")

            prediction_result = dict()
            prediction_result["bboxes"] = bboxes
            prediction_result["polys"] = bboxes
            prediction_result["scores"] = scores
            prediction_result["heatmap"] = None

            pil_image = Image.new(
                "RGB", (image.shape[1], image.shape[0]), color=(0, 255, 0, 0)
            )

            rect_from_poly = []
            rect_line_numbers = []
            fragments = []

            for i in range(len(bboxes)):
                # Adjust box from (xmin, ymin, xmax, ymax) -> (x, y, w, h)
                box = np.array(bboxes[i]).astype(np.int32)
                x0, y0, x1, y1 = box
                w = x1 - x0
                h = y1 - y0
                box_adj = [x0, y0, w, h]

                # Class 0 == Text
                if classes[i] == 0:
                    snippet = img[y0 : y0 + h, x0 : x0 + w :]
                    line_number = find_line_number(lines_bboxes, box_adj)
                    fragments.append(snippet)
                    rect_from_poly.append(box_adj)
                    rect_line_numbers.append(line_number)

                    # After normalization image is in 0-1 range
                    # snippet = (snippet * 255).astype(np.uint8)
                    paste_fragment(pil_image, snippet, (x0, y0))

            if False:
                savepath = os.path.join(debug_dir, f"{key}_txt_overlay.jpg")
                pil_image.save(savepath, format="JPEG", subsampling=0, quality=100)

                cv_img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                stacked = np.hstack((cv_img, img))

                save_path = os.path.join(debug_dir, f"{key}_stacked.png")
                imwrite(save_path, stacked)

            stop_time = time.time()
            eval_time = round((stop_time - start_time) * 1000, 2)

            # metrics.add_time('HandlerTime', round(
            #     (stop_time - start_time) * 1000, 2), None, 'ms')

            # we can't return np.array here as t the 'fragments' will throw an error
            # ValueError: could not broadcast input array from shape (42,77,3) into shape (42,)
            return (
                rect_from_poly,
                fragments,
                rect_line_numbers,
                prediction_result,
                lines_bboxes,
            )

        except Exception as ex:
            raise ex
