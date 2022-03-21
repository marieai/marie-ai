import os

import numpy as np
import tqdm
import glob

import cv2

from renderer.pdf_renderer import PdfRenderer
from renderer.text_renderer import TextRenderer
from boxes.box_processor import PSMode
from utils.utils import ensure_exists

from boxes.box_processor_craft import BoxProcessorCraft
from boxes.box_processor_textfusenet import BoxProcessorTextFuseNet
from document.icr_processor import IcrProcessor

from overlay.overlay import OverlayProcessor
from utils.image_utils import imwrite
from utils.utils import ensure_exists

from utils.tiff_ops import burst_tiff, merge_tiff
from utils.pdf_ops import merge_pdf
from utils.utils import ensure_exists

if __name__ == "__main__":

    work_dir_boxes = ensure_exists("/tmp/boxes")
    work_dir_icr = ensure_exists("/tmp/icr")
    work_dir = ensure_exists("/tmp/form-segmentation")

    img_path = "/home/greg/dataset/medprov/PID/150300431/PID_576_7188_0_150300431.tif"

    if not os.path.exists(img_path):
        raise Exception(f"File not found : {img_path}")

    # this is the image working directory
    root_dir = "/home/greg/dataset/medprov/PID/150300431/"

    burst_dir = ensure_exists(os.path.join(root_dir, "burst"))
    stack_dir = ensure_exists(os.path.join(root_dir, "stack"))
    clean_dir = ensure_exists(os.path.join(root_dir, "clean"))
    pdf_dir = ensure_exists(os.path.join(root_dir, "pdf"))

    overlay_processor = OverlayProcessor(work_dir=work_dir)
    box = BoxProcessorCraft(work_dir=work_dir_boxes, models_dir="./models/craft", cuda=False)
    # box = BoxProcessorTextFuseNet(work_dir=work_dir_boxes, models_dir='./models/fusenet', cuda=False)
    icr = IcrProcessor(work_dir=work_dir_icr, cuda=False)

    # prolog
    # burst_tiff(img_path, burst_dir)

    # process each image from the bursts directory
    for i, _path in enumerate(sorted(glob.glob(os.path.join(burst_dir, "*.tif")))):
        try:
            filename = _path.split("/")[-1]
            docId = filename.split("/")[-1].split(".")[0]
            key = docId
            image = cv2.imread(_path)
            print(f"DocumentId : {docId}")

            if not os.path.exists(os.path.join(clean_dir, filename)):
                src_img_path = os.path.join(burst_dir, filename)
                real, fake, blended = overlay_processor.segment(docId, src_img_path)
                # debug image
                if False:
                    stacked = np.hstack((real, fake, blended))
                    save_path = os.path.join(stack_dir, f"{docId}.png")
                    imwrite(save_path, stacked)

                save_path = os.path.join(clean_dir, f"{docId}.tif")  # This will have the .tif extension
                imwrite(save_path, blended)
                print(f"Saving  document : {save_path}")

            pdf_save_path = os.path.join(pdf_dir, f"{docId}.pdf")
            if not os.path.exists(pdf_save_path):
                boxes, img_fragments, lines, _ = box.extract_bounding_boxes(key, "field", image, PSMode.SPARSE)
                result, overlay_image = icr.recognize(key, "test", image, boxes, img_fragments, lines)
                print(f"Rendering PDF document : {pdf_save_path}")
                renderer = PdfRenderer(config={"preserve_interword_spaces": True})
                renderer.render(image, result, pdf_save_path)

            if i == -1:
                break
        except Exception as ident:
            # raise ident
            print(ident)

    merge_pdf(pdf_dir, '/tmp/merged.pdf')

    # # epilog
    # merge_tiff(burst_dir, '/tmp/burst.tif')
    # merge_tiff(clean_dir, '/tmp/clean.tif')
