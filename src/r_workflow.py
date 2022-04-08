import glob
import io
import json
import multiprocessing
import os
import pathlib
import shutil
from zipfile import ZipFile

import cv2
import torch
import torch.backends.cudnn as cudnn

from boxes.box_processor import PSMode
from boxes.craft_box_processor import BoxProcessorCraft
from numpyencoder import NumpyEncoder
from document.trocr_icr_processor import TrOcrIcrProcessor
from overlay.overlay import OverlayProcessor
from renderer.adlib_renderer import AdlibRenderer
from renderer.blob_renderer import BlobRenderer
from renderer.pdf_renderer import PdfRenderer
from timer import Timer
from utils.image_utils import imwrite
from utils.pdf_ops import merge_pdf
from utils.tiff_ops import merge_tiff
from utils.utils import ensure_exists


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


def __sort_key_files_by_page(name):
    page = name.split("/")[-1].split(".")[0].split("_")[-1]
    return int(page)


def write_adlib_summary(adlib_dir, adlib_summary_filename, file_sorter):
    import xml.etree.ElementTree as gfg

    def _meta(field, val):
        meta = gfg.Element("METADATAELEMENT")
        meta.set("FIELD", str(field))
        meta.set("VALUE", str(val))
        return meta

    root = gfg.Element("OCR")
    metas = gfg.Element("METADATAELEMENTS")
    root.append(metas)
    pages_node = gfg.Element("PAGES")

    for idx, _path in enumerate(sorted(glob.glob(os.path.join(adlib_dir, "*.xml")), key=file_sorter)):
        filename = _path.split("/")[-1]
        filePageIndex = _path.split("/")[-1].split(".")[0].split("_")[-1]
        print(f" {filePageIndex} : {filename}")

        # <PAGE Filename="PID_576_7188_0_150459314_1.tif.xml" NUMBER="1"/>
        node = gfg.Element("PAGE")
        node.set("Filename", str(filename))
        node.set("NUMBER", str(idx + 1))
        pages_node.append(node)
    root.append(pages_node)

    tree = gfg.ElementTree(root)
    with open(adlib_summary_filename, "wb") as files:
        tree.write(files)


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
    root_dir = pathlib.Path(img_path).parent.absolute()
    fileId = img_path.split("/")[-1].split(".")[0]
    # fileIdZeroed = fileId.replace("_7188_", "_0_").replace("_150459314","_0") # RMSOCR QUirk

    burst_dir = ensure_exists(os.path.join(root_dir, "burst"))
    stack_dir = ensure_exists(os.path.join(root_dir, "stack"))
    clean_dir = ensure_exists(os.path.join(root_dir, "clean"))
    pdf_dir = ensure_exists(os.path.join(root_dir, "pdf"))
    result_dir = ensure_exists(os.path.join(root_dir, "results"))
    assets_dir = ensure_exists(os.path.join(root_dir, "assets"))
    blob_dir = ensure_exists(os.path.join(root_dir, "blobs"))
    adlib_dir = ensure_exists(os.path.join(root_dir, "adlib"))
    adlib_final_dir = ensure_exists(os.path.join(root_dir, "adlib_final"))

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda_present = torch.cuda.is_available()

    cudnn.benchmark = True
    cudnn.deterministic = True


    overlay_processor = OverlayProcessor(work_dir=work_dir, cuda=False)
    box = BoxProcessorCraft(work_dir=work_dir_boxes, models_dir="./model_zoo/craft", cuda=False)
    # icr = CraftIcrProcessor(work_dir=work_dir_icr, cuda=False)
    # box = BoxProcessorTextFuseNet(work_dir=work_dir_boxes, models_dir='./model_zoo/textfusenet', cuda=False)
    icr = TrOcrIcrProcessor(work_dir=work_dir_icr, cuda=False)

    # burst_tiff(img_path, burst_dir)

    os.environ["OMP_NUM_THREADS"] = str(multiprocessing.cpu_count())
    # os.environ["OMP_NUM_THREADS"] = str(1)

    # Improves inference by ~ 13%-18%(verified empirically)
    torch.autograd.set_detect_anomaly(False)
    torch.set_grad_enabled(False)
    torch.autograd.profiler.emit_nvtx(enabled=False)

    # process each image from the bursts directory
    for idx, _path in enumerate(sorted(glob.glob(os.path.join(burst_dir, "*.tif")), key=__sort_key_files_by_page)):
        try:
            filename = _path.split("/")[-1]
            docId = filename.split("/")[-1].split(".")[0]
            key = docId
            clean_image_path = os.path.join(clean_dir, filename)
            burs_image_path = os.path.join(burst_dir, filename)
            image_clean = cv2.imread(clean_image_path)
            image_original = cv2.imread(burs_image_path)
            pageIndex = idx + 1
            print(f"DocumentId : {docId}")

            # make sure we have clean image
            if not os.path.exists(os.path.join(clean_dir, filename)):
                src_img_path = os.path.join(burst_dir, filename)
                real, fake, blended = overlay_processor.segment(docId, src_img_path)
                # debug image
                if False:
                    stacked = np.hstack((real, fake, blended))
                    save_path = os.path.join(stack_dir, f"{docId}.png")
                    imwrite(save_path, stacked)

                save_path = os.path.join(clean_dir, f"{docId}.tif")  # This will have the .tif extension
                imwrite(save_path, blended, dpi=(300, 300))
                print(f"Saved clean img : {save_path}")

            pdf_save_path = os.path.join(pdf_dir, f"{docId}.pdf")
            icr_save_path = os.path.join(result_dir, f"{docId}.json")

            result = None
            # require both PDF and OCR results
            if not os.path.exists(pdf_save_path) or not os.path.exists(icr_save_path):
                boxes, img_fragments, lines, _ = box.extract_bounding_boxes(key, "field", image_clean, PSMode.SPARSE)
                result, overlay_image = icr.recognize(key, "test", image_clean, boxes, img_fragments, lines)

                with open(icr_save_path, "w") as f:
                    json.dump(
                        result,
                        f,
                        sort_keys=True,
                        separators=(",", ": "),
                        ensure_ascii=False,
                        indent=4,
                        cls=NumpyEncoder,
                    )

                print(f"Rendering PDF document : {pdf_save_path}")
                renderer = PdfRenderer(config={"preserve_interword_spaces": True})
                renderer.render(image_original, result, pdf_save_path)
            else:
                result = from_json_file(icr_save_path)

            blob_save_path = os.path.join(blob_dir, f"{fileId}_{pageIndex}.BLOBS.XML")
            if not os.path.exists(blob_save_path):
                print(f"Rendering blob : {blob_save_path}")
                renderer = BlobRenderer(config={"page_number": pageIndex})
                renderer.render(image_original, result, blob_save_path)

            adlib_save_path = os.path.join(adlib_dir, f"{fileId}_{pageIndex}.tif.xml")
            if True or not os.path.exists(adlib_save_path):
                print(f"Rendering adlib : {adlib_save_path}")
                renderer = AdlibRenderer(config={"page_number": pageIndex})
                renderer.render(image_original, result, adlib_save_path)

            if idx == -1:
                break
        except Exception as ident:
            raise ident

    # Summary info
    adlib_summary_filename = os.path.join(adlib_final_dir, f"{fileId}.tif.xml")
    write_adlib_summary(adlib_dir, adlib_summary_filename, __sort_key_files_by_page)
    shutil.copytree(adlib_dir, adlib_final_dir, dirs_exist_ok=True)

    # create assets
    merge_zip(adlib_final_dir, os.path.join(assets_dir, f"{fileId}.ocr.zip"))
    merge_zip(blob_dir, os.path.join(assets_dir, f"{fileId}.blobs.xml.zip"))
    merge_pdf(pdf_dir, os.path.join(assets_dir, f"{fileId}.pdf"), __sort_key_files_by_page)
    merge_tiff(clean_dir, os.path.join(assets_dir, f"{fileId}.tif.clean"), __sort_key_files_by_page)
