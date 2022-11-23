import argparse
import copy
import os
import time
import cv2
import numpy as np
import torch

from marie.boxes.line_processor import find_line_number, line_merge
from marie.logger import setup_logger
from PIL import Image

from marie.boxes.box_processor import BoxProcessor, PSMode, create_dirs
from marie.utils.image_utils import paste_fragment, imwrite
from marie.utils.utils import ensure_exists

from ditod import add_vit_config
from detectron2.utils.visualizer import ColorMode, Visualizer

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from PIL import Image, ImageDraw

logger = setup_logger(__name__)


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


def lines_from_bboxes(image, bboxes):
    """Create lines out of bboxes for given image.
    Args:
        image(ndarray): numpy array of shape (H, W), where H is the image height and W is the image width.
        bboxes: Bounding boxes for image (xmin,ymin,xmax,ymax)
    """

    print(image.shape)
    _h = image.shape[0]
    _w = image.shape[1]

    overlay = np.ones((_h, _w, 3), dtype=np.uint8) * 255
    img_line = copy.deepcopy(image)
    viz_img = cv2.cvtColor(img_line, cv2.COLOR_BGR2RGB)
    viz_img = Image.fromarray(viz_img)
    draw = ImageDraw.Draw(viz_img, "RGBA")

    for box in bboxes:
        x1, y1, x2, y2 = box.astype(np.int32)
        w = x2 - x1
        h = (y2 - y1) // 2
        y1_adj = y1 + h // 2

        cv2.rectangle(overlay, (x1, y1_adj), (x1 + w, y1_adj + h), (0, 0, 0), -1)

        draw.rectangle(
            [x1, y1, x2, y2],
            outline="#993300",
            fill=(
                int(np.random.random() * 256),
                int(np.random.random() * 256),
                int(np.random.random() * 256),
                125,
            ),
            width=1,
        )

    viz_img.save(
        os.path.join("/tmp/fragments", f"overlay_refiner_boxes.png"), format="PNG"
    )

    cv2.imwrite(os.path.join("/tmp/fragments", f"overlay_refiner-RAW.jpg"), overlay)

    # ret, link_score = cv2.threshold(overlay, link_threshold, 1, 0)
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
    ret, link_score = cv2.threshold(overlay, 0, 255, cv2.THRESH_BINARY)

    # [horiz]
    # Specify size on horizontal axis
    cols = link_score.shape[1]
    horizontal_size = cols // 30
    # Create structure element for extracting horizontal lines through morphology operations
    horizontal_struct = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))

    # Apply morphology operations
    horizontal = np.copy(link_score)
    horizontal = cv2.erode(horizontal, horizontal_struct)
    horizontal = cv2.dilate(horizontal, horizontal_struct)

    if True:
        cv2.imwrite("/tmp/fragments/horizontal.jpg", horizontal)
        # cv2.imwrite("/tmp/fragments/lines-morph.jpg", line_img)
        # cv2.imwrite("/tmp/fragments/lines-morph.jpg", line_img)
        # cv2.imwrite(
        #     os.path.join("/tmp/fragments/", "h-text_score_comb.jpg"),
        #     text_score_comb,
        # )

    # create binary mask for the image
    # cv2.connectedComponents_XXXXXX() considers only the white portion as a component.
    # binary = cv2.bitwise_not(binary)

    binary_mask = np.zeros(horizontal.shape)
    binary_mask[horizontal == 255] = 0
    binary_mask[horizontal != 255] = 255
    binary_mask = binary_mask.astype(np.uint8)
    connectivity = 4

    nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask, connectivity, cv2.CV_32S
    )

    h, w = link_score.shape
    stat_overlay = np.ones((h, w, 3), dtype=np.uint8) * 255
    line_bboxes = []

    # 0 is background (useless)
    for k in range(1, nLabels):
        size = stats[k, cv2.CC_STAT_AREA]
        x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        # size filtering
        # if h < 2:
        #     continue
        box = x, y, w, h
        line_bboxes.append(box)
        color = list(np.random.random(size=3) * 255)
        cv2.rectangle(stat_overlay, (x, y), (x + w, y + h), color, 1)

    cv2.imwrite("/tmp/fragments/stat_overlay.png", stat_overlay)
    cv2.imwrite("/tmp/fragments/binary_mask.png", binary_mask)

    lines_bboxes = line_merge(stat_overlay, line_bboxes)

    # overlay = np.ones((_h, _w, 3), dtype=np.uint8) * 255
    img_line = copy.deepcopy(image)
    viz_img = cv2.cvtColor(img_line, cv2.COLOR_BGR2RGB)
    viz_img = Image.fromarray(viz_img)
    draw = ImageDraw.Draw(viz_img, "RGBA")

    for box in lines_bboxes:
        x, y, w, h = box
        color = list(np.random.random(size=3) * 256)
        # cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 1)
        draw.rectangle(
            [x, y, x + w, y + h],
            outline="#993300",
            fill=(
                int(np.random.random() * 256),
                int(np.random.random() * 256),
                int(np.random.random() * 256),
                125,
            ),
            width=1,
        )

    viz_img.save(
        os.path.join("/tmp/fragments", f"overlay_refiner-final.png"), format="PNG"
    )

    return lines_bboxes


class BoxProcessorUlimDit(BoxProcessor):
    """DiT for Text Detection"""

    def __init__(
        self,
        work_dir: str = "/tmp/boxes",
        models_dir: str = "./model_zoo/unilm/dit/text_detection",
        cuda: bool = False,
    ):
        super().__init__(work_dir, models_dir, cuda)
        logger.info("Box processor [dit, cuda={}]".format(cuda))
        args = get_parser().parse_args(
            [
                "--config-file",
                "./config/zoo/unilm/dit/text_detection/mask_rcnn_dit_large.yaml",
                "--opts",
                "MODEL.WEIGHTS",
                os.path.join(models_dir, "td-syn_dit-l_mrcnn.pth"),
            ]
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg = setup_cfg(args, device)
        self.predictor = DefaultPredictor(cfg)
        self.cpu_device = torch.device("cpu")

    def psm_word(self, image):
        raise Exception("Not implemented")

    def psm_sparse(self, image):
        predictions = self.predictor(image)
        predictions = predictions["instances"].to(self.cpu_device)
        logger.info(f"Number of prediction : {len(predictions)}")

        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
        scores = predictions.scores if predictions.has("scores") else None
        classes = predictions.pred_classes if predictions.has("pred_classes") else None

        bboxes = _convert_boxes(boxes)
        # sort by xy-coordinated
        ind = np.lexsort((bboxes[:, 0], bboxes[:, 1]))
        bboxes = bboxes[ind]
        lines = lines_from_bboxes(image, bboxes)

        return bboxes, classes, scores, lines, classes

    def psm_line(self, image):
        raise Exception("Not implemented")

    def psm_raw_line(self, image):
        raise Exception("Not implemented")

    def psm_multiline(self, image):
        raise Exception("Not implemented")

    def extract_bounding_boxes(self, _id, key, img, psm=PSMode.SPARSE):

        if img is None:
            raise Exception("Input image can't be empty")
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

            debug_dir = "/tmp/fragments"
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
            return (
                rect_from_poly,
                fragments,
                rect_line_numbers,
                prediction_result,
                lines_bboxes,
            )

        except Exception as ex:
            raise ex
