import os
import sys
import time

import cv2
import numpy as np

from marie.utils.docs import load_image, frames_from_file
from marie.utils.image_utils import crop_to_content
from marie.utils.utils import ensure_exists


if __name__ == '__main__':

    work_dir_boxes = ensure_exists('/tmp/boxes')
    work_dir_icr = ensure_exists('/tmp/icr')
    img_path = './assets/psm/word/0001.png'

    img_path = "~/tmp/wrong-ocr/regions/overlay_image_1_9359800610.png"
    # img_path = "~/tmp/wrong-ocr/regions/overlay_image_1_9359800610_clipped.png"
    img_path = os.path.expanduser(img_path)
    print(img_path)
    if not os.path.exists(img_path):
        raise Exception(f"File not found : {img_path}")

    key = img_path.split("/")[-1]
    frames = frames_from_file(img_path)
    img = frames[0]
    print(len(frames))
    crop_to_content(img)
