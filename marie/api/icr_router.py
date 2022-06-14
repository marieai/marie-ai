import torch
from torch.backends import cudnn

from marie import Executor, requests

import io
import os

import hashlib
import imghdr
import json
import logging
from distutils.util import strtobool as strtobool

import numpy as np

import cv2

import marie.conf
import marie.processors


from flask import Blueprint, jsonify
from flask_restful import Resource, reqparse, request

from marie.logging.logger import MarieLogger
from marie.numpyencoder import NumpyEncoder
from marie.serve.runtimes import monitoring
from marie.utils.utils import FileSystem, ensure_exists, current_milli_time
from marie.boxes.box_processor import PSMode
from marie.utils.base64 import base64StringToBytes, encodeToBase64
from marie.utils.network import find_open_port, get_ip_address

from marie.boxes.box_processor import PSMode
from marie.boxes.craft_box_processor import BoxProcessorCraft
from marie.common.file_io import PathManager
from marie.document.craft_icr_processor import CraftIcrProcessor
from marie.numpyencoder import NumpyEncoder
from marie.document.trocr_icr_processor import TrOcrIcrProcessor

logger = MarieLogger("")

ALLOWED_TYPES = {"png", "jpeg", "tiff"}
TYPES_TO_EXT = {"png": "png", "jpeg": "jpg", "tiff": "tif"}


def load_image(fname, image_type):
    """ "
    Load image, if the image is a TIFF, we will load the image as a multipage tiff, otherwise we return an
    array with image as first element
    """
    import skimage.io as skio

    if fname is None:
        return False, None

    if image_type == "tiff":
        loaded, frames = cv2.imreadmulti(fname, [], cv2.IMREAD_ANYCOLOR)
        if not loaded:
            return False, []
        # each frame needs to be converted to RGB format
        converted = []
        for frame in frames:
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            converted.append(frame)

        return loaded, converted

    img = skio.imread(fname)  # RGB order
    if img.shape[0] == 2:
        img = img[0]
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    img = np.array(img)

    return True, [img]


def store_temp_file(message_bytes, queue_id, file_type, store_raw):
    """Store temp file from decoded payload"""
    m = hashlib.sha256()
    m.update(message_bytes)
    checksum = m.hexdigest()

    upload_dir = ensure_exists(f"/tmp/marie/{queue_id}")
    ext = TYPES_TO_EXT[file_type]

    from datetime import datetime

    current_datetime = datetime.now()
    str_current_datetime = str(current_datetime)
    tmp_file = f"{upload_dir}/{checksum}_{str_current_datetime}.{ext}"

    if store_raw:
        # message read directly from a file
        with open(tmp_file, "wb") as tmp:
            tmp.write(message_bytes)
    else:
        # TODO : This does not handle multipage tiffs
        # convert to numpy array as the message has been passed from base64
        npimg = np.frombuffer(message_bytes, np.uint8)
        # convert numpy array to image
        img = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)
        cv2.imwrite(tmp_file, img)

    return tmp_file, checksum


def extract_payload(payload, queue_id):  # -> tuple[bytes, str]:
    """Extract data from payload"""
    # determine how to extract payload based on the type of the key supplied
    # Possible keys
    # data, srcData, srcFile, srcUrl

    # This indicates that the contents are a file contentstens and need to stored as they appear

    store_raw = False
    if "data" in payload:
        raw_data = payload["data"]
        data = base64StringToBytes(raw_data)
    elif "srcData" in payload:
        raw_data = payload["srcData"]
        data = base64StringToBytes(raw_data)
    elif "srcBase64" in payload:
        raw_data = payload["srcBase64"]
        data = base64StringToBytes(raw_data)
        store_raw = True
    elif "srcFile" in payload:
        img_path = payload["srcFile"]
        # FIXME: relative path resolution is not working as expected
        # FIXME : Use PathManager
        base_dir = FileSystem.get_share_directory()
        path = os.path.abspath(os.path.join(base_dir, img_path))
        logger.info(f"base_dir = {base_dir}")
        logger.info(f"raw_data = {img_path}")
        logger.info(f"resolved path = {path}")
        if not os.path.exists(path):
            raise Exception(f"File not found : {img_path}")
        with open(path, "rb") as file:
            data = file.read()
        store_raw = True
    else:
        raise Exception("Unable to determine datasource in payload")

    with io.BytesIO(data) as memfile:
        file_type = imghdr.what(memfile)

    if file_type not in ALLOWED_TYPES:
        raise Exception(f"Unsupported file type, expected one of : {ALLOWED_TYPES}")

    tmp_file, checksum = store_temp_file(data, queue_id, file_type, store_raw)
    logger.info(f"File info: {file_type}, {tmp_file}")

    return tmp_file, checksum, file_type


class ICRRouter(Executor):
    def __init__(self, app, **kwargs):
        super().__init__(**kwargs)
        if app is None:
            raise RuntimeError("Expected app arguments is null")

        prefix = "/api"
        app.add_url_rule(rule=f"{prefix}/info", endpoint="info", view_func=self.info, methods=["GET"])
        # app.add_url_rule(rule="/status/<queue_id>", endpoint="status", view_func=self.status, methods=["GET"])
        app.add_url_rule(rule=f"/{prefix}/status", endpoint="status", view_func=self.status, methods=["GET"])
        app.add_url_rule(rule=f"/{prefix}", endpoint="api_index", view_func=self.status, methods=["GET"])
        app.add_url_rule(
            rule=f"{prefix}/extract/<queue_id>", endpoint="extract", view_func=self.extract, methods=["POST"]
        )

        self.show_error = True  # show prediction errors

        work_dir_boxes = ensure_exists("/tmp/boxes")
        work_dir_icr = ensure_exists("/tmp/icr")

        # sometimes we have CUDA/GPU support but want to only use CPU
        has_cuda = torch.cuda.is_available()
        if os.environ.get("MARIE_DISABLE_CUDA"):
            has_cuda = False

        if has_cuda:
            cudnn.benchmark = False
            cudnn.deterministic = False

        self.box_processor = BoxProcessorCraft(work_dir=work_dir_boxes, cuda=has_cuda)
        self.icr_processor = TrOcrIcrProcessor(work_dir=work_dir_icr, cuda=has_cuda)

    def status(self):
        """Get application status"""
        import os

        build = {}
        if os.path.exists(".build"):
            with open(".build", "r") as fp:
                build = json.load(fp)
        host = get_ip_address()

        return (
            jsonify(
                {
                    "name": "marie-ai",
                    "host": host,
                    "component": [
                        {"name": "craft", "version": "1.0.0"},
                        {"name": "craft-benchmark", "version": "1.0.0"},
                    ],
                    "build": build,
                }
            ),
            200,
        )

    def info(self):
        logger.info(f"Self : {self}")
        return {"index": "complete"}

    def process_extract_fullpage(self, frames, queue_id, checksum, pms_mode, args, **kwargs):
        """Process full page extraction"""
        # TODO : Implement multipage tiff processing

        page_index = 0
        img = frames[page_index]
        h = img.shape[0]
        w = img.shape[1]
        # allow for small padding around the component
        padding = 4
        overlay = np.ones((h + padding * 2, w + padding * 2, 3), dtype=np.uint8) * 255
        overlay[padding : h + padding, padding : w + padding] = img

        boxes, img_fragments, lines, _ = self.box_processor.extract_bounding_boxes(
            queue_id, checksum, overlay, pms_mode
        )
        result, overlay_image = self.icr_processor.recognize(queue_id, checksum, overlay, boxes, img_fragments, lines)

        cv2.imwrite(f"/tmp/marie/overlay_image_{page_index}.png", overlay_image)
        result["overlay_b64"] = encodeToBase64(overlay_image)

        return result

    def process_extract_regions(self, frames, queue_id, checksum, pms_mode, regions, args):
        """Process region based extract"""
        filter_snippets = bool(strtobool(args["filter_snippets"])) if "filter_snippets" in args else False
        output = []
        extended = []

        for region in regions:
            # validate required fields
            if not all(key in region for key in ("id", "pageIndex", "x", "y", "w", "h")):
                raise Exception(f"Required key missing in region : {region}")

        for region in regions:
            try:
                logger.info(f"Extracting box : {region}")
                rid = region["id"]
                page_index = region["pageIndex"]
                x = region["x"]
                y = region["y"]
                w = region["w"]
                h = region["h"]

                img = frames[page_index]
                img = img[y : y + h, x : x + w].copy()
                # allow for small padding around the component
                padding = 4
                overlay = np.ones((h + padding * 2, w + padding * 2, 3), dtype=np.uint8) * 255
                overlay[padding : h + padding, padding : w + padding] = img
                cv2.imwrite(f"/tmp/marie/overlay_image_{page_index}_{rid}.png", overlay)

                logger.info(f"pms_mode = {pms_mode}")
                boxes, img_fragments, lines, _ = self.box_processor.extract_bounding_boxes(
                    queue_id, checksum, overlay, pms_mode
                )

                result, overlay_image = self.icr_processor.recognize(
                    queue_id, checksum, overlay, boxes, img_fragments, lines
                )

                # cv2.imwrite(f"/tmp/marie/overlay_image_{page_index}_{rid}.png", overlay_image)
                if not filter_snippets:
                    result["overlay_b64"] = encodeToBase64(overlay_image)

                result["id"] = rid

                extended.append(result)

                # TODO : Implement rendering modes
                # 1 - Simple
                # 2 - Full
                # 3 - HOCR

                logger.info(result)
                rendering_mode = "simple"
                region_result = {}
                if rendering_mode == "simple":
                    if "lines" in result:
                        lines = result["lines"]
                        line = lines[0]
                        region_result["id"] = rid
                        region_result["text"] = line["text"]
                        region_result["confidence"] = line["confidence"]
                        output.append(region_result)
            except Exception as ex:
                logger.error(ex)
                raise ex

        # Filter out base 64 encoded fragments(fragment_b64, overlay_b64)
        # This is useful when we like to display or process image in the output but has significant payload overhead

        def filter_base64(node, filters):
            if isinstance(node, (list, tuple, np.ndarray)):
                for v in node:
                    filter_base64(v, filters)
            elif isinstance(node, dict):
                for flt in filters:
                    try:
                        del node[flt]
                    except KeyError:
                        pass
                for key, value in node.items():
                    filter_base64(value, filters)
            else:
                pass
            return node

        if filter_snippets:
            extended = filter_base64(extended, filters=["fragment_b64", "overlay_b64"])

        return {"regions": output, "extended": extended}

    def extract(self, queue_id: str):
        """
        ICR payload to process
            Process image via ICR, this is low level API, to get more usable results call extract_icr.

        Args:
            queue_id: Unique queue to tie the extraction to
        """

        logger.info("Starting ICR processing request", extra={"session": queue_id})
        try:
            payload = request.json
            if payload is None:
                return {"error": "empty payload"}, 200

            pms_mode = PSMode.from_value(payload["mode"] if "mode" in payload else "")
            regions = payload["regions"] if "regions" in payload else []

            # due to compatibility issues with other frameworks we allow passing same arguments in the 'args' object
            args = {}
            if "args" in payload:
                args = payload["args"]
                pms_mode = PSMode.from_value(payload["args"]["mode"] if "mode" in payload["args"] else "")

            tmp_file, checksum, file_type = extract_payload(payload, queue_id)
            loaded, frames = load_image(tmp_file, file_type)

            if not loaded:
                raise Exception(f"Unable to read image : {tmp_file}")

            frame_len = len(frames)
            logger.info(f"frame size, regions size : {frame_len}, {len(regions)}")

            if len(regions) == 0:
                result = self.process_extract_fullpage(frames, queue_id, checksum, pms_mode, args)
            else:
                result = self.process_extract_regions(frames, queue_id, checksum, pms_mode, regions, args)

            serialized = json.dumps(
                result, sort_keys=True, separators=(",", ": "), ensure_ascii=False, indent=2, cls=NumpyEncoder
            )

            return serialized, 200
        except BaseException as error:
            logger.error("Extract error", error)
            if self.show_error:
                return {"error": str(error)}, 500
            else:
                return {"error": "inference exception"}, 500
