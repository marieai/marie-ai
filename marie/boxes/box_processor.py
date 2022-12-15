import os
from abc import ABC, abstractmethod
from collections import OrderedDict
from enum import Enum
from typing import Any, Tuple

import cv2
import numpy as np
from PIL import Image

from marie.base_handler import BaseHandler
from marie.utils.utils import ensure_exists


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")


def make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if h == oh and w == ow:
        return img

    if False:
        print(
            "The image size needs to be a multiple of 4. "
            "The loaded image size was (%d, %d), so it was adjusted to "
            "(%d, %d). This adjustment will be done to all images "
            "whose sizes are not multiples of 4" % (ow, oh, w, h)
        )

    return img.resize((w, h), method)


def order_points(pts):
    # initialize a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect


def estimate_character_width(src_img, bounding_boxes):
    """
    Estimate Character Width based on the score map.
    """
    src = src_img.copy()
    # print(f"Estimated stroke width detection : {src.shape}")
    # conversion required, or we will get 'Failure to use adaptiveThreshold: CV_8UC1 in function adaptiveThreshold'
    # cv2.imwrite("/tmp/fragments/esw_src.png", src)
    # Transform source image to gray if it is not already
    if len(src.shape) != 2:
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    else:
        gray = src

    gray = gray * 255
    gray = gray.astype("uint8")
    # cv2.imwrite("/tmp/fragments/gray.png", gray)
    # Picked values based on experiments
    thresh = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)[1]
    # cv2.imwrite("/tmp/fragments/thresh.png", thresh)
    # thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # cv2.imwrite('/tmp/fragments/thresh_th2.png', thresh)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    points = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    cv2.imwrite("/tmp/fragments/esw_points.png", points)

    # we should calculat this for each bouding_box separately
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        points.astype(np.uint8), connectivity=4
    )
    total_chars = n_labels - 1
    total_width = 0
    # convert the bounding boxes points (polys) into a box
    for idx, region in enumerate(bounding_boxes):
        region = np.array(region).astype(np.int32).reshape((-1))
        region = region.reshape(-1, 2)
        poly = region.reshape((-1, 1, 2))
        box = cv2.boundingRect(poly)
        box = np.array(box).astype(np.int32)
        x, y, w, h = box
        total_width += w

    # using hardcoded values
    # https://accessibility.psu.edu/legibility/fontsize/
    # https://www.thomasphinney.com/2011/03/point-size/
    # https://www.w3.org/Style/Examples/007/units.en.html#font-size
    #
    # 1 character(X) = 8 pixel(X)
    # 1 pixel(X) = 0.125 character(X)
    # char_width = 8
    if total_chars == 0:
        char_width = 8
    else:
        char_width = total_width // total_chars

    # print(f"total_chars, char_width : {total_chars}, {char_width}")
    return char_width


class PSMode(Enum):
    """ "Page Segmentation Modes"""

    # Treat the image as a single word.
    WORD = "word"
    # Sparse text. Find as much text as possible in no particular order.
    SPARSE = "sparse"
    # Treat the image as a single text line.
    LINE = "line"
    # Raw line. Treat the image as a single text line, NO bounding box detection performed.
    RAW_LINE = "raw_line"
    # Multiline. Treat the image as multiple text lines, NO bounding box detection performed.
    MULTI_LINE = "multiline"

    @staticmethod
    def from_value(value: str):
        if value is None:
            return PSMode.SPARSE
        for data in PSMode:
            # print("{:15} = {}".format(data.name, data.value))
            if data.value == value.lower():
                return data
        return PSMode.SPARSE


def create_dirs(work_dir, _id, key):
    debug_dir = ensure_exists(
        os.path.join(work_dir, _id, "bounding_boxes", key, "debug")
    )
    crops_dir = ensure_exists(
        os.path.join(work_dir, _id, "bounding_boxes", key, "crop")
    )
    lines_dir = ensure_exists(
        os.path.join(work_dir, _id, "bounding_boxes", key, "lines")
    )
    mask_dir = ensure_exists(os.path.join(work_dir, _id, "bounding_boxes", key, "mask"))

    return crops_dir, debug_dir, lines_dir, mask_dir


class BoxProcessor(BaseHandler):
    """Box processor extract bounding boxes"""

    def __init__(
        self,
        work_dir: str = "/tmp/boxes",
        models_dir: str = "./models",
        cuda: bool = False,
        config={},
    ):
        """Initialize

        Args:
            work_dir: Working directory
            models_dir: Models directory
            cuda: Is CUDA processing enabled
            config: Configuration map
        """

        self.cuda = cuda
        self.work_dir = work_dir

    @abstractmethod
    def extract_bounding_boxes(
        self, _id, key, img, psm=PSMode.SPARSE
    ) -> Tuple[Any, Any, Any, Any, Any]:
        """Extract bounding boxes for specific image, try to predict line number representing each bounding box.
        Args:
            _id:  Unique Image ID
            key: Unique image key/Zone
            img: A pre-cropped image containing characters
            psm: Page Segmentation Mode accepts one of following (sparse, word, line)
        Return:
            box array, fragment array, line_number array,  prediction results[bboxes, polys, heatmap]
        """

    @abstractmethod
    def psm_word(self, image):
        """Treat the image as a single word.
        Args:
            image: A pre-cropped image
        Return:
            bboxes array, polys array, score_text array,  lines
        """

    @abstractmethod
    def psm_sparse(self, image):
        """Find as much text as possible (default).
        Args:
            image: A pre-cropped image
        Return:
            bboxes array, polys array, score_text array,  lines
        """

    @abstractmethod
    def psm_line(self, image):
        """
        Treat the image as a single text line.
        """

    @abstractmethod
    def psm_raw_line(self, image):
        """Treat the image as a single text line.
        Args:
            image: A pre-cropped image
        Return:
            bboxes array, polys array, score_text array,  lines
        """

    @abstractmethod
    def psm_multiline(self, image):
        """Treat the image as a single word.
        Args:
            image: A pre-cropped image
        Return:
            bboxes array, polys array, score_text array,  lines
        """
