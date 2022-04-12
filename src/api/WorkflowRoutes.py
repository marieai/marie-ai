from flask import Blueprint, jsonify
from flask_restful import request

import conf
from logger import create_info_logger
from utils.network import get_ip_address

import glob
import io
import json
import os
import pathlib
import shutil
import logging

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from boxes.box_processor import PSMode
from boxes.craft_box_processor import BoxProcessorCraft
from common.file_io import PathManager
from document.craft_icr_processor import CraftIcrProcessor
from numpyencoder import NumpyEncoder
from document.trocr_icr_processor import TrOcrIcrProcessor
from overlay.overlay import OverlayProcessor
from renderer.adlib_renderer import AdlibRenderer
from renderer.blob_renderer import BlobRenderer
from renderer.pdf_renderer import PdfRenderer
from utils.image_utils import imwrite
from utils.pdf_ops import merge_pdf
from utils.tiff_ops import merge_tiff, burst_tiff
from utils.utils import ensure_exists, FileSystem
from utils.zip_ops import merge_zip

logger = create_info_logger(__name__, "marie.log")

# Blueprint Configuration
blueprint = Blueprint(
    name='workflow_bp',
    import_name=__name__,
    url_prefix=conf.API_PREFIX
)

logger.info('Workflow Routes inited')
show_error = True  # show prediction errors


@blueprint.route('/workflow', methods=['GET'])
def status():
    """Get status"""
    host = get_ip_address()

    return jsonify(
        {
            "name": "marie-icr",
            "host": host,
            "component": [
                {
                    "name": "workflow",
                    "version": "1.0.0"
                },
            ],
        }
    ), 200


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

        node = gfg.Element("PAGE")
        node.set("Filename", str(filename))
        node.set("NUMBER", str(idx + 1))
        pages_node.append(node)
    root.append(pages_node)

    tree = gfg.ElementTree(root)
    with open(adlib_summary_filename, "wb") as files:
        tree.write(files)


def from_json_file(filename):
    with io.open(filename, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)
        return data


def __sort_key_files_by_page(name):
    page = name.split("/")[-1].split(".")[0].split("_")[-1]
    return int(page)


def process_workflow(src_file: str, dry_run: bool) -> None:
    logger.info(f"src_file : {src_file}")
    work_dir_boxes = ensure_exists("/tmp/boxes")
    work_dir_icr = ensure_exists("/tmp/icr")
    work_dir = ensure_exists("/tmp/form-segmentation")

    exists = PathManager.exists(src_file)
    img_path = PathManager.get_local_path(src_file)
    logger.info(f"resolved : {img_path}, {exists}")

    if not os.path.exists(img_path):
        raise Exception(f"File not found : {img_path}")

    # this is the image working directory
    src_dir = pathlib.Path(img_path).parent.absolute()
    file_id = img_path.split("/")[-1].split(".")[0].split("_")[-1]
    root_asset_dir = ensure_exists(os.path.join("/tmp", "assets", file_id))

    from datetime import datetime
    backup_time = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    backup_dir = ensure_exists(os.path.join(src_dir, "backup", f"{file_id}_{backup_time}"))
    # backup_dir = ensure_exists(os.path.join(src_dir, "backup"))

    logger.info("Creating snapshot: %s", backup_dir)
    for idx, src_path in enumerate(glob.glob(os.path.join(src_dir, f"*{file_id}*"))):
        try:
            filename = src_path.split("/")[-1]
            dst_path = os.path.join(backup_dir, filename)
            logger.info("snapshot: %s", dst_path)
            shutil.copyfile(src_path, dst_path)
        except Exception as e:
            logger.error("Error in file copy - {}".format(str(e)))

    # return
    # fileIdZeroed = fileId.replace("_7188_", "_0_").replace("_150459314","_0") # RMSOCR QUIRK
    burst_dir = ensure_exists(os.path.join(root_asset_dir, "burst"))
    stack_dir = ensure_exists(os.path.join(root_asset_dir, "stack"))
    clean_dir = ensure_exists(os.path.join(root_asset_dir, "clean"))
    pdf_dir = ensure_exists(os.path.join(root_asset_dir, "pdf"))
    result_dir = ensure_exists(os.path.join(root_asset_dir, "results"))
    assets_dir = ensure_exists(os.path.join(root_asset_dir, "assets"))
    blob_dir = ensure_exists(os.path.join(root_asset_dir, "blobs"))
    adlib_dir = ensure_exists(os.path.join(root_asset_dir, "adlib"))
    adlib_final_dir = ensure_exists(os.path.join(root_asset_dir, "adlib_final"))

    burst_tiff(img_path, burst_dir)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda_present = torch.cuda.is_available()

    cudnn.benchmark = True
    cudnn.deterministic = True

    overlay_processor = OverlayProcessor(work_dir=work_dir, cuda=True)
    box = BoxProcessorCraft(work_dir=work_dir_boxes, models_dir="./model_zoo/craft", cuda=True)
    # icr = CraftIcrProcessor(work_dir=work_dir_icr, cuda=True)
    # box = BoxProcessorTextFuseNet(work_dir=work_dir_boxes, models_dir='./model_zoo/textfusenet', cuda=False)
    icr = TrOcrIcrProcessor(work_dir=work_dir_icr, cuda=True)

    # os.environ["OMP_NUM_THREADS"] = str(multiprocessing.cpu_count())
    # # os.environ["OMP_NUM_THREADS"] = str(1)

    # Improves inference by ~ 13%-18%(verified empirically)
    torch.autograd.set_detect_anomaly(False)
    torch.set_grad_enabled(False)
    torch.autograd.profiler.emit_nvtx(enabled=False)

    # process each image from the bursts directory
    for idx, _path in enumerate(sorted(glob.glob(os.path.join(burst_dir, "*.tif")), key=__sort_key_files_by_page)):
        try:
            filename = _path.split("/")[-1]
            doc_id = filename.split("/")[-1].split(".")[0]
            key = doc_id
            clean_image_path = os.path.join(clean_dir, filename)
            burs_image_path = os.path.join(burst_dir, filename)
            image_clean = cv2.imread(clean_image_path)
            image_original = cv2.imread(burs_image_path)
            page_index = idx + 1
            print(f"DocumentId : {doc_id}")

            # make sure we have clean image
            if not os.path.exists(os.path.join(clean_dir, filename)):
                src_img_path = os.path.join(burst_dir, filename)
                real, fake, blended = overlay_processor.segment(doc_id, src_img_path)
                # debug image
                if True:
                    stacked = np.hstack((real, fake, blended))
                    save_path = os.path.join(stack_dir, f"{doc_id}.png")
                    imwrite(save_path, stacked)

                save_path = os.path.join(clean_dir, f"{doc_id}.tif")  # This will have the .tif extension
                imwrite(save_path, blended, dpi=(300, 300))
                image_clean = blended
                print(f"Saved clean img : {save_path}")

            pdf_save_path = os.path.join(pdf_dir, f"{doc_id}.pdf")
            icr_save_path = os.path.join(result_dir, f"{doc_id}.json")

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

            blob_save_path = os.path.join(blob_dir, f"{file_id}_{page_index}.BLOBS.XML")
            if not os.path.exists(blob_save_path):
                print(f"Rendering blob : {blob_save_path}")
                renderer = BlobRenderer(config={"page_number": page_index})
                renderer.render(image_original, result, blob_save_path)

            adlib_save_path = os.path.join(adlib_dir, f"{file_id}_{page_index}.tif.xml")
            if not os.path.exists(adlib_save_path):
                print(f"Rendering adlib : {adlib_save_path}")
                renderer = AdlibRenderer(config={"page_number": page_index})
                renderer.render(image_original, result, adlib_save_path)

            if idx == -1:
                break
        except Exception as ident:
            raise ident

    # Summary info
    adlib_summary_filename = os.path.join(adlib_final_dir, f"{file_id}.tif.xml")
    write_adlib_summary(adlib_dir, adlib_summary_filename, __sort_key_files_by_page)
    shutil.copytree(adlib_dir, adlib_final_dir, dirs_exist_ok=True)

    # create assets
    merge_zip(adlib_final_dir, os.path.join(assets_dir, f"{file_id}.ocr.zip"))
    merge_zip(blob_dir, os.path.join(assets_dir, f"{file_id}.blobs.xml.zip"))
    merge_pdf(pdf_dir, os.path.join(assets_dir, f"{file_id}.pdf"), __sort_key_files_by_page)
    merge_tiff(clean_dir, os.path.join(assets_dir, f"{file_id}.tif.clean"), __sort_key_files_by_page)

    # copy files from assets back to the asset source
    if dry_run:
        logger.info("Copying final assets[dry_run]: %s", assets_dir)
    else:
        logger.info("Copying final assets: %s", assets_dir)
        for idx, src_path in enumerate(glob.glob(os.path.join(assets_dir, "*.*"))):
            try:
                filename = src_path.split("/")[-1]
                dst_path = os.path.join(src_dir, filename)
                logger.info("Copying asset: %s, %s", src_path, dst_path)
                shutil.copyfile(src_path, dst_path)
            except Exception as e:
                logger.error("Error in file copy - {}".format(str(e)))


@blueprint.route("/workflow/<queue_id>", methods=["POST"])
def workflow(queue_id: str):
    """
    Workflow to process
    Args:
        queue_id: Unique queue to tie the extraction to
    """

    logger.info("Starting Workflow processing request", extra={"session": queue_id})
    try:
        payload = request.json
        print(payload)
        if payload is None:
            return {"error": "empty payload"}, 200

        if 'src' not in payload:
            return {"error": 'src missing'}, 200

        dry_run = True if 'dry-run' not in payload else False
        src = payload['src']
        process_workflow(src, dry_run)
        serialized = src

        return serialized, 200
    except BaseException as error:
        # raise error
        # print(str(error))
        if show_error:
            return {"error": str(error)}, 500
        else:
            return {"error": 'inference exception'}, 500
