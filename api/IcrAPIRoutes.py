from enum import Enum
import enum
from typing import Any
from flask_restful import Resource, reqparse, request
from flask import jsonify
from boxes.box_processor import PSMode
import conf
from logger import create_info_logger
from utils.utils import current_milli_time, ensure_exists
import cv2
import numpy as np
import processors
import json

import hashlib
from numpyencoder import NumpyEncoder
import base64
import imghdr
from skimage import io
from flask import Blueprint

from distutils.util import strtobool as strtobool
from utils.network import get_ip_address, find_open_port

logger = create_info_logger(__name__, "marie.log")

ALLOWED_TYPES = {'png', 'jpeg', 'tiff'}
TYPES_TO_EXT = {'png': 'png', 'jpeg': 'jpg', 'tiff': 'tif'}

def encodeToBase64(img: np.ndarray) -> str:
    """encode image to base64"""
    retval, buffer = cv2.imencode('.png', img)
    png_as_text = base64.b64encode(buffer).decode()
    return png_as_text


def base64StringToBytes(data: str):
    """conver base 64 string to byte"""
    if data is None:
        return ""
    base64_message = data
    base64_bytes = base64_message.encode('utf-8')
    message_bytes = base64.b64decode(base64_bytes)
    return message_bytes


def load_image(fname, image_type):
    """"
        Load image, if the image is a TIFF, we will load the image as a multipage tiff, otherwise we return an
        array with image as first element
    """
    if fname is None:
        return False, None

    if image_type == 'tiff':
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

    img = io.imread(fname)  # RGB order
    if img.shape[0] == 2:
        img = img[0]
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    img = np.array(img)

    return True, [img]


# Blueprint Configuration
blueprint = Blueprint(
    name='icr_bp',
    import_name=__name__,
    url_prefix=conf.API_PREFIX
)

logger.info('IcrAPIRoutes inited')
box_processor = processors.box_processor
icr_processor = processors.icr_processor
show_error = True  # show predition errors


@blueprint.route('/', methods=['GET'])
def status():
    """Get application status"""
    import os
    build = {}
    if os.path.exists('.build'):
        with open('.build', 'r') as fp:
            build = json.load(fp)
    host = get_ip_address()

    return jsonify(
        {
            "name": "marie-icr",
            "host": host,
            "component": [
                {
                    "name": "craft",
                    "version": "1.0.0"
                },
                {
                    "name": "craft-benchmark",
                    "version": "1.0.0"
                }
            ],
            "build": build
        }
    ), 200


def store_temp_file(message_bytes, queue_id, file_type, store_raw):
    """Store temp file from decoded payload"""
    m = hashlib.sha256()
    m.update(message_bytes)
    checksum = m.hexdigest()

    upload_dir = ensure_exists(f'/tmp/marie/{queue_id}')
    ext = TYPES_TO_EXT[file_type]
    tmp_file = f"{upload_dir}/{checksum}.{ext}"

    if store_raw :
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
    import io
    import os
    from utils.utils import ensure_exists, FileSystem

    print(f"Payload info ")
    # determine how to extract payload based on the type of the key supplied
    # Possible keys
    # data, srcData, srcFile, srcUrl

    store_raw = False
    if "data" in payload:
        raw_data = payload["data"]
        data = base64StringToBytes(raw_data)
    elif "srcData" in payload:
        raw_data = payload["srcData"]
        data = base64StringToBytes(raw_data)
    elif "srcFile" in payload:
        img_path = payload["srcFile"]
        base_dir = FileSystem.get_share_directory()
        path = os.path.abspath(os.path.join(base_dir, img_path))
        print(f'raw_data = {img_path}')
        print(f'resolved path = {path}')
        if not os.path.exists(path):
            raise Exception(f"File not found : {img_path}")
        with open(path, 'rb') as file:
            data = file.read()
        store_raw = True
    else:
        raise Exception("Unable to determine datasource in payload")

    with io.BytesIO(data) as memfile:
        file_type = imghdr.what(memfile)

    if file_type not in ALLOWED_TYPES:
        raise Exception(F"Unsupported file type, expected one of : {ALLOWED_TYPES}")

    tmp_file, checksum = store_temp_file(data, queue_id, file_type, store_raw)
    print(f'Filetype : {file_type}')
    print(f'tmp_file : {tmp_file}')

    return tmp_file, checksum, file_type


def process_extract_fullpage(frames, queue_id, checksum, pms_mode, args):
    """Process full page extraction """
    # TODO : Implement multipage tiff processing

    page_index = 0
    img = frames[page_index]
    h = img.shape[0]
    w = img.shape[1]
    # allow for small padding around the component
    padding = 4
    overlay = np.ones((h + padding * 2, w + padding * 2, 3), dtype=np.uint8) * 255
    overlay[padding:h + padding, padding:w + padding] = img

    boxes, img_fragments, lines, _ = box_processor.extract_bounding_boxes(
        queue_id, checksum, overlay, pms_mode)
    result, overlay_image = icr_processor.recognize(
        queue_id, checksum, overlay, boxes, img_fragments, lines)

    cv2.imwrite(f'/tmp/marie/overlay_image_{page_index}.png', overlay_image)
    result['overlay_b64'] = encodeToBase64(overlay_image)

    return result


def process_extract_regions(frames, queue_id, checksum, pms_mode, regions, args):
    """Process region based extract"""
    filter_snippets = bool(strtobool(args['filter_snippets'])) if 'filter_snippets' in args else False
    output = []
    extended = []

    for region in regions:
        # validate required fields
        if not all(key in region for key in ("id", "pageIndex", "x", "y", "w", "h")):
            raise Exception(f"Required key missing in region : {region}")

    for region in regions:
        try:
            rid = region['id']
            page_index = region['pageIndex']
            x = region['x']
            y = region['y']
            w = region['w']
            h = region['h']

            img = frames[page_index]
            img = img[y:y + h, x:x + w].copy()
            # allow for small padding around the component
            padding = 4
            overlay = np.ones((h + padding * 2, w + padding * 2, 3), dtype=np.uint8) * 255
            overlay[padding:h + padding, padding:w + padding] = img

            boxes, img_fragments, lines, _ = box_processor.extract_bounding_boxes(
                queue_id, checksum, overlay, pms_mode)
            result, overlay_image = icr_processor.recognize(
                queue_id, checksum, overlay, boxes, img_fragments, lines)

            cv2.imwrite(f'/tmp/marie/overlay_image_{page_index}_{rid}.png', overlay_image)
            result["overlay_b64"] = encodeToBase64(overlay_image)
            result["id"] = rid

            extended.append(result)

            # TODO : Implement rendering modes
            # 1 - Simple
            # 2 - Full
            # 3 - HOCR

            rendering_mode = 'simple'
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
            print(ex)
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


@blueprint.route("/extract/<queue_id>", methods=["POST"])
def extract(queue_id: str):
    """
    ICR payload to process
        Process image via ICR, this is low level API, to get more usable results call extract_icr.

    Args:
        queue_id: Unique queue to tie the extraction to
    """

    logger.info("Starting ICR processing request", extra={"session": queue_id})
    try:
        print(request)
        payload = request.json
        if payload is None:
            return {"error": "empty payload"}, 200

        pms_mode = PSMode.fromValue(payload["mode"] if 'mode' in payload else '')
        regions = payload["regions"] if "regions" in payload else []

        # due to compatibility issues with other frameworks we allow passing same arguments in the 'args' object
        args = {}
        if 'args' in payload:
            args = payload['args']
            pms_mode = PSMode.fromValue(payload['args']['mode'] if 'mode' in payload['args'] else '')

        tmp_file, checksum, file_type = extract_payload(payload, queue_id)
        loaded, frames = load_image(tmp_file, file_type)

        print(f'Frame size : {len(frames)}')
        if not loaded:
            print(f'Unable to read image : {tmp_file}')
            raise Exception(f'Unable to read image : {tmp_file}')

        frame_len = len(frames)
        print(f'frame_len : {frame_len}')
        print(f'regions_len : {len(regions)}')

        if len(regions) == 0:
            result = process_extract_fullpage(frames, queue_id, checksum, pms_mode, args)
        else:
            result = process_extract_regions(frames, queue_id, checksum, pms_mode, regions, args)

        serialized = json.dumps(result, sort_keys=True, separators=(',', ': '), ensure_ascii=False, indent=2,
                                cls=NumpyEncoder)

        return serialized, 200
    except BaseException as error:
        # raise error
        # print(str(error))
        if show_error:
            return {"error": str(error)}, 500
        else:
            return {"error": 'inference exception'}, 500
