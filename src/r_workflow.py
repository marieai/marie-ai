import io
import os
import pathlib
from zipfile import ZipFile
from timer import Timer

import multiprocessing

import numpy as np
import torch
import tqdm
import glob
import cv2
from pathlib import Path

from renderer.pdf_renderer import PdfRenderer
from renderer.blob_renderer import BlobRenderer
from renderer.adlib_renderer import AdlibRenderer
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

import json
from document.numpyencoder import NumpyEncoder
from shutil import make_archive


def from_json_file(filename):
    with io.open(filename, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)
        return data


@Timer(text="Creating zip in {:.2f} seconds")
def merge_zip(src_dir, dst_path):
    """Add files from directory to the zipfile without absolute path"""
    from os.path import basename
    with ZipFile(dst_path, "w") as newzip:
        for _path in sorted(glob.glob(os.path.join(src_dir, "*.*"))):
            newzip.write(_path, basename(_path))


if __name__ == "__main__":

    work_dir_boxes = ensure_exists("/tmp/boxes")
    work_dir_icr = ensure_exists("/tmp/icr")
    work_dir = ensure_exists("/tmp/form-segmentation")

    img_path = "/home/greg/dataset/medprov/PID/150300431/PID_576_7188_0_150300431.tif"
    img_path = "/home/gbugaj/datasets/medprov/PID/150300431/PID_576_7188_0_150300431.tif"

    img_path = "/home/gbugaj/datasets/medprov/PID/150459314/PID_576_7188_0_150459314.tif"

    if not os.path.exists(img_path):
        raise Exception(f"File not found : {img_path}")

    # this is the image working directory
    root_dir = "/home/greg/dataset/medprov/PID/150300431/"
    root_dir = pathlib.Path(img_path).parent.absolute()
    fileId = img_path.split("/")[-1].split(".")[0]

    burst_dir = ensure_exists(os.path.join(root_dir, "burst"))
    stack_dir = ensure_exists(os.path.join(root_dir, "stack"))
    clean_dir = ensure_exists(os.path.join(root_dir, "clean"))
    pdf_dir = ensure_exists(os.path.join(root_dir, "pdf"))
    result_dir = ensure_exists(os.path.join(root_dir, "results"))
    assets_dir = ensure_exists(os.path.join(root_dir, "assets"))
    blob_dir = ensure_exists(os.path.join(root_dir, "blobs"))
    adlib_dir = ensure_exists(os.path.join(root_dir, "adlib"))

    overlay_processor = OverlayProcessor(work_dir=work_dir)
    box = BoxProcessorCraft(work_dir=work_dir_boxes, models_dir="./models/craft", cuda=False)
    # box = BoxProcessorTextFuseNet(work_dir=work_dir_boxes, models_dir='./models/fusenet', cuda=False)
    icr = IcrProcessor(work_dir=work_dir_icr, cuda=False)

    # burst_tiff(img_path, burst_dir)

    # os.environ["OMP_NUM_THREADS"] = str(multiprocessing.cpu_count())
    # os.environ["OMP_NUM_THREADS"] = str(1)

    # Improves inference by ~ 13%-18%(verified empirically)
    torch.autograd.set_detect_anomaly(False)
    torch.set_grad_enabled(False)
    torch.autograd.profiler.emit_nvtx(enabled=False)

    # process each image from the bursts directory
    for idx, _path in enumerate(sorted(glob.glob(os.path.join(burst_dir, "*.tif")))):
        try:
            filename = _path.split("/")[-1]
            docId = filename.split("/")[-1].split(".")[0]
            key = docId
            clean_image_path = os.path.join(clean_dir, filename)
            pageIndex = idx + 1
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
            icr_save_path = os.path.join(result_dir, f"{docId}.json")
            image = cv2.imread(clean_image_path)
            result = None

            if not os.path.exists(pdf_save_path):
                boxes, img_fragments, lines, _ = box.extract_bounding_boxes(key, "field", image, PSMode.SPARSE)
                result, overlay_image = icr.recognize(key, "test", image, boxes, img_fragments, lines)

                with open(icr_save_path, 'w') as f:
                    json.dump(result, f, sort_keys=True, separators=(',', ': '), ensure_ascii=False, indent=4,
                              cls=NumpyEncoder)

                print(f"Rendering PDF document : {pdf_save_path}")
                renderer = PdfRenderer(config={"preserve_interword_spaces": True})
                renderer.render(image, result, pdf_save_path)
            else:
                result = from_json_file(icr_save_path)

            blob_save_path = os.path.join(blob_dir, f"{fileId}_{pageIndex}.BLOBS.XML")
            if not os.path.exists(blob_save_path):
                print(f'Rendering blob : {blob_save_path}')
                renderer = BlobRenderer(config={"page_number": pageIndex})
                renderer.render(image, result, blob_save_path)

            adlib_save_path = os.path.join(adlib_dir, f"{fileId}_{pageIndex}.tif.xml")
            if not os.path.exists(adlib_save_path):
                print(f'Rendering adlib : {adlib_save_path}')
                renderer = AdlibRenderer(config={"page_number": pageIndex})
                renderer.render(image, result, adlib_save_path)

            if idx == -1:
                break
        except Exception as ident:
            raise ident
            print(ident)

    # epilog
    merge_zip(blob_dir, os.path.join(assets_dir, f'{fileId}.blobs.xml.zip'))

    merge_pdf(pdf_dir, os.path.join(assets_dir, f'{fileId}.pdf'))
    merge_tiff(clean_dir,  os.path.join(assets_dir, f'{fileId}.tif.clean'))
