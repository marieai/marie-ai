import os

import numpy as np
import tqdm

import cv2

from renderer.pdf_renderer import PdfRenderer
from renderer.text_renderer import TextRenderer
from boxes.box_processor import PSMode
from utils.utils import ensure_exists

from boxes.box_processor_craft import BoxProcessorCraft
from boxes.box_processor_textfusenet import BoxProcessorTextFuseNet
from document.icr_processor import IcrProcessor


if __name__ == "__main__":

    work_dir_boxes = ensure_exists("/tmp/boxes")
    work_dir_icr = ensure_exists("/tmp/icr")
    ensure_exists("/tmp/fragments")

    img_path = "./assets/psm/word/0001.png"
    img_path = "./assets/english/Scanned_documents/Picture_029.tif"
    # img_path = './assets/english/Scanned_documents/t2.tif'
    img_path = "./assets/english/Scanned_documents/Picture_010.tif"
    # img_path = "./assets/english/Lines/002.png"
    # img_path = './assets/english/Lines/001.png'
    # img_path = './assets/english/Lines/003.png'
    # img_path = './assets/english/Lines/005.png'
    # img_path = './assets/english/Lines/004.png'

    # img_path = './assets/private/PID_576_7188_0_149495857_page_0002.tif'
    # img_path = "/home/greg/dataset/medprov/PID/150300431/clean/PID_576_7188_0_150300431_page_0005.tif"

    # cal_mean_std('./assets/english/Scanned_documents/')

    if not os.path.exists(img_path):
        raise Exception(f"File not found : {img_path}")

    if True:
        key = img_path.split("/")[-1]
        image = cv2.imread(img_path)

        box = BoxProcessorCraft(work_dir=work_dir_boxes, models_dir="./models/craft", cuda=False)
        # box = BoxProcessorTextFuseNet(work_dir=work_dir_boxes, models_dir='./models/fusenet', cuda=False)
        icr = IcrProcessor(work_dir=work_dir_icr, cuda=False)

        boxes, img_fragments, lines, _ = box.extract_bounding_boxes(key, "field", image, PSMode.LINE)
        result, overlay_image = icr.recognize(key, "test", image, boxes, img_fragments, lines)

        output_filename = '/tmp/output_filename.pdf'
        print("Testing pdf render")

        renderer = PdfRenderer(config={"preserve_interword_spaces": True})
        renderer.render(image, result, output_filename)
