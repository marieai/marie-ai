import argparse
import copy
import os
import time
from typing import Any, Tuple, Union, Optional

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
from marie.models.utils import torch_gc
from marie.utils.image_utils import imwrite, paste_fragment
from marie.utils.overlap import merge_boxes
from marie.utils.resize_image import resize_image


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

    if image is None:
        raise Exception(
            "Input image can't be empty : Ensure  overlay_bboxes is set to TRUE"
        )
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
        lines_bboxes: Bounding boxes for the lines in (x,y,w,h) format
    """
    enable_visualization = False

    if enable_visualization:
        viz_img = visualize_bboxes(image, bboxes)
        viz_img.save(
            os.path.join("/tmp/fragments", f"line_refiner_initial.png"), format="PNG"
        )

    # create a box overlay with adjusted coordinates
    overlay = np.ones((image.shape[0], image.shape[1], 1), dtype=np.uint8) * 100
    for box in bboxes:
        x1, y1, x2, y2 = box.astype(np.int32)
        w = x2 - x1
        q = (y2 - y1) // 8
        h = (y2 - y1) // 2 + q
        y1_adj = y1 + h // 2 - q
        cv2.rectangle(overlay, (x1, y1_adj), (x1 + w, y1_adj + h), (0, 0, 0), -1)

    ret, link_score = cv2.threshold(overlay, 0, 255, cv2.THRESH_BINARY)

    if enable_visualization:
        cv2.imwrite(os.path.join("/tmp/fragments", f"overlay_refiner-RAW.PNG"), overlay)
        cv2.imwrite(
            os.path.join("/tmp/fragments", f"overlay_refiner-link_score.PNG"),
            link_score,
        )

    # Create structure element for extracting horizontal lines through morphology operations
    # select the horizontal size based on the image width  3000 -> 160 1000 -> 80 500 -> 40

    cols = link_score.shape[1]
    half_cols = cols // 2
    stride = cols // min(160, cols)
    horizontal_size = stride if stride > 1 else half_cols
    #
    # print(f"cols: {cols}")
    # print(f"half_cols: {half_cols}")
    # print(f"horizontal_size: {horizontal_size}")
    horizontal_struct = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))

    # Apply morphology operations
    horizontal = np.copy(link_score)
    horizontal = cv2.erode(horizontal, horizontal_struct)
    horizontal = cv2.dilate(horizontal, horizontal_struct)

    if enable_visualization:
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

    if enable_visualization:
        viz_img = visualize_bboxes(image, lines_bboxes, format="xywh")
        viz_img.save(
            os.path.join("/tmp/fragments", f"line_refiner-final.png"), format="PNG"
        )

    return lines_bboxes


def crop_to_content_box(
        frame: np.ndarray, content_aware=False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crop given image to content and return new box with the offset.
    No content is defined as first non background(white) pixel.

    @param frame: the image frame to process
    @param content_aware: if enabled we will apply more aggressive crop method
    @return: the offset box in LTRB format (left, top, right, bottom) and cropped image.
    """

    start = time.time()
    # conversion required, or we will get 'Failure to use adaptiveThreshold: CV_8UC1 in function adaptiveThreshold'
    # frame = np.random.choice([0, 255], size=(32, 32), p=[0.01, 0.99]).astype("uint8")
    # Transform source image to gray if it is not already
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # with content aware we will apply more aggressive crop method to remove the background
    # there is a trade off here, we will lose some content but we will also remove the background
    if content_aware:
        # apply division normalization to preprocess the image
        blur = cv2.GaussianBlur(gray, (5, 5), sigmaX=0, sigmaY=0)
        # divide
        divide = cv2.divide(gray, blur, scale=255)
        thresh = cv2.threshold(divide, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        #
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 3))
        op_frame = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    else:
        op_frame = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    indices = np.array(np.where(op_frame == [0]))
    img_w = op_frame.shape[1]
    img_h = op_frame.shape[0]

    min_x_pad = 1
    min_y_pad = 1
    # cv2.imwrite("/tmp/fragments/crop_to_content_box.jpg", op_frame)
    if len(indices[0]) == 0:
        # no content found, return the whole image
        return [0, 0, 0, 0], frame

    # indices are in y,X format
    if content_aware:
        x = max(0, indices[1].min() - min_x_pad)
        y = max(0, indices[0].min() - min_y_pad)
        h = min(img_h, indices[0].max() - y + min_y_pad)
        w = min(img_w, indices[1].max() - x + min_x_pad)
    else:
        x = indices[1].min()
        y = indices[0].min()
        h = indices[0].max() - y
        w = indices[1].max() - x

    cropped = frame[y: y + h + 1, x: x + w + 1].copy()
    dt = time.time() - start
    # create offset box in LTRB format (left, top, right, bottom) from XYWH format
    offset = [x, y, img_w - w, img_h - h]
    # print("crop_to_content_box took {}s".format(dt))
    # print("offset box: {}".format(offset))
    return offset, cropped


class CompiledDefaultPredictor(DefaultPredictor):
    def __init__(self, cfg):
        super().__init__(cfg)


class BoxProcessorUlimDit(BoxProcessor):
    """
    Document text box processor using DIT model from ULIM.

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
            work_dir: str = "/tmp/boxes",
            models_dir: str = __model_path__,
            cuda: bool = False,
    ):
        super().__init__(work_dir, models_dir, cuda)
        self.logger = MarieLogger(self.__class__.__name__)
        self.logger.info("Box processor [dit, cuda={}]".format(cuda))
        self.logger.info(f"Loading model from config dir {__config_dir__}")

        args = get_parser().parse_args(
            [
                "--config-file",
                os.path.join(
                    __config_dir__,
                    "zoo/unilm/dit/text_detection/mask_rcnn_dit_prod.yaml",
                ),
            ]
        )

        self.logger.info(f"Loading model from {args.config_file}")
        self.strict_box_segmentation = False
        device = "cuda" if cuda else "cpu"

        cfg = setup_cfg(args, device)
        self.min_size_test = [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST]
        self.predictor = DefaultPredictor(cfg)
        self.cpu_device = torch.device("cpu")

    def psm_word(self, image):
        if self.strict_box_segmentation:
            raise Exception("Not implemented : PSM_WORD")
        return self.psm_sparse(image)

    def psm_sparse(
            self,
            image: np.ndarray,
            bbox_optimization: Optional[bool] = False,
            bbox_context_aware: Optional[bool] = True,
            enable_visualization: Optional[bool] = False,
    ):
        try:
            self.logger.debug(f"Starting box predictions : {image.shape}")
            #  this should match Detectron2 model input size
            # this is to ensure that the model works correctly
            # FIXME : This is a hack
            # TODO : Update the model to work with any size image
            adj_x = 0
            adj_y = 0
            orig_image = image

            # Both height and width are smaller than the minimum size then frame the image
            if (
                    image.shape[0] < self.min_size_test[0]
                    or image.shape[1] < self.min_size_test[1]
            ):
                self.logger.debug(
                    f"Image size is too small : {image.shape}, resizing to {self.min_size_test}"
                )
                image, coord = resize_image(
                    image,
                    (self.min_size_test[0], self.min_size_test[1]),
                    keep_max_size=True,
                )
                self.logger.debug(f"Resized image  : {image.shape}, {coord}")
                # cv2.imwrite(f"/tmp/marie/bbox_framed.png", image)
                adj_x = coord[0]
                adj_y = coord[1]

            rp = self.predictor(image)
            image = orig_image
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

            # delete the predictor to free up memory
            del predictions
            del rp

            # check if boxes are empty, which means no boxes were detected(blank image)
            if boxes is None or len(boxes) == 0:
                self.logger.debug(f"No boxes predicted.")
                return [], [], [], [], []

            bboxes = _convert_boxes(boxes)
            self.logger.debug(f"Predicted boxes : {len(boxes)}")
            # adjust the boxes to original image size
            if adj_x != 0 or adj_y != 0:
                bboxes[:, 0] = bboxes[:, 0] - adj_x
                bboxes[:, 1] = bboxes[:, 1] - adj_y
                bboxes[:, 2] = bboxes[:, 2] - adj_x
                bboxes[:, 3] = bboxes[:, 3] - adj_y

                # clip the boxes to image size and 0 to avoid negative values
                bboxes[:, 0] = np.clip(bboxes[:, 0], 0, image.shape[1])
                bboxes[:, 1] = np.clip(bboxes[:, 1], 0, image.shape[0])
                bboxes[:, 2] = np.clip(bboxes[:, 2], 0, image.shape[1])
                bboxes[:, 3] = np.clip(bboxes[:, 3], 0, image.shape[0])

            # for each box check if it has a height and width are in range of (0..2], if so remove it
            # this is a workaround for a bug in the model where it predicts a box with height and width of fraction of a pixel
            len_a = len(bboxes)
            min_height = 2
            min_width = 2
            # bboxes are in (xmin,ymin,xmax,ymax) format
            bboxes = [box for box in bboxes if box[2] - box[0] > min_height]
            bboxes = [box for box in bboxes if box[3] - box[1] > min_width]

            len_b = len(bboxes)
            if len_a != len_b:
                self.logger.debug(
                    f"Removed predicted boxes that did not meet size minimum requirements: {len_a - len_b}"
                )
            if len_b == 0:
                self.logger.debug(f"No boxes found within requirements")
                return [], [], [], [], []

            bboxes = merge_boxes(bboxes, 0.08)
            bboxes = np.array(bboxes)

            if bbox_optimization:
                # extract snippets from the image
                for i, box in enumerate(bboxes):
                    box = np.array(box).astype(np.int32)
                    x0, y0, x1, y1 = box
                    w = x1 - x0
                    h = y1 - y0
                    snippet = image[y0: y0 + h, x0: x0 + w:]
                    offset, cropped = crop_to_content_box(
                        snippet, content_aware=bbox_context_aware
                    )
                    # cv2.imwrite(f"/tmp/fragments/snippet_{i}.png", snippet)
                    # cv2.imwrite(f"/tmp/fragments/snippet_{i}_cropped.png", cropped)
                    adj_box = [
                        box[0] + offset[0],
                        box[1] + offset[1],
                        box[2] - (offset[2] - offset[0]),
                        box[3] - (offset[3] - offset[1]),
                    ]

                    # print(f"Snippet {i} : {box} -> {offset} -> {adj_box}")
                    bboxes[i] = adj_box

            # FIXME : This is a hack
            # TODO : Update the model to correctly predict the orientation of the text and assign the correct class
            # remove vertical boxes
            bb = []
            for box in bboxes:
                h = box[2] - box[0]
                w = box[3] - box[1]
                rat = w / h
                if rat < 2.5:
                    bb.append(box)

            bboxes = np.array(bb)

            if len(bboxes) == 0:
                self.logger.debug(f"No boxes predicted.")
                return [], [], [], [], []

            # sort by xy-coordinated
            ind = np.lexsort((bboxes[:, 0], bboxes[:, 1]))
            bboxes = bboxes[ind]
            lines = lines_from_bboxes(image, bboxes)

            return bboxes, classes, scores, lines, classes
        except Exception as e:
            raise e
        finally:
            torch_gc()

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

    @torch.no_grad()
    def extract_bounding_boxes(
            self,
            _id,
            key,
            img: Union[np.ndarray, PIL.Image.Image],
            psm=PSMode.SPARSE,
            bbox_optimization: Optional[bool] = False,
            bbox_context_aware: Optional[bool] = True,
    ) -> Tuple[Any, Any, Any, Any, Any]:
        if img is None:
            raise Exception("Input image can't be empty")

        if isinstance(img, PIL.Image.Image):  # convert pil to OpenCV
            # convert PIL.TiffImagePlugin.TiffImageFile to numpy array
            self.logger.warning("PIL image received converting to ndarray")
            img = img.convert("RGB")
            converted = np.array(img, dtype=np.uint8)
            img = cv2.cvtColor(converted, cv2.COLOR_RGB2BGR)

        if not isinstance(img, np.ndarray):
            raise Exception("Expected image in numpy format")
        # bbox_optimization = True
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
                bboxes, polys, scores, lines_bboxes, classes = self.psm_sparse(
                    image, bbox_optimization, bbox_context_aware
                )
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

            debug_image = False
            if debug_image:
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

                # self.logger.debug(f" index = {i} box_adj = {box_adj}  : {h} , {w}  > {box}")
                # Class 0 == Text
                if classes[i] == 0:
                    snippet = img[y0: y0 + h, x0: x0 + w:]
                    line_number = find_line_number(lines_bboxes, box_adj)
                    fragments.append(snippet)
                    rect_from_poly.append(box_adj)
                    rect_line_numbers.append(line_number)

                    # After normalization image is in 0-1 range
                    # snippet = (snippet * 255).astype(np.uint8)
                    if debug_image:
                        paste_fragment(pil_image, snippet, (x0, y0))

            if debug_image:
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
