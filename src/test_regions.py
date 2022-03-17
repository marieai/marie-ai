import os

import cv2
from utils.utils import ensure_exists


#
# from boxes.box_processor import BoxProcessor, PSMode
# from document.icr_processor import IcrProcessor

if __name__ == '__main__':

    work_dir_boxes = ensure_exists('/tmp/boxes')
    work_dir_icr = ensure_exists('/tmp/icr')
    img_path = './assets/psm/word/0001.png'

    if not os.path.exists(img_path):
        raise Exception(f'File not found : {img_path}')
