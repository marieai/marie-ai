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

logger = create_info_logger(__name__, "marie.log")

# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
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


def loadImage(fname):
    """"
        Load image, if the image is a TIFF, we will load the image as a multipage tiff, otherwise we return an
        array with image as first element
    """
    if fname is None:
        return False, None

    if fname.lower().endswith(('.tiff', '.tif')):
        loaded, frames = cv2.imreadmulti(fname, [], cv2.IMREAD_ANYCOLOR)
        return loaded, frames

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
    return jsonify(
        {
            "name": "icr",
            "version": "1.0.0",
            "component": [
                {
                    "name": "craft",
                    "version": "1.0.0"
                },
                {
                    "name": "craft-benchmark",
                    "version": "1.0.0"
                }
            ]
        }
    ), 200


def extract_payload(payload):  # -> tuple[bytes, str]:
    """Extract data from payload"""
    import io
    print(f"Payload info ")
    # determine how to extract payload based on the type of the key supplied
    # Possible keys
    # data, srcData, srcFile, srcUrl

    if "data" in payload:
        raw_data = payload["data"]
        data = base64StringToBytes(raw_data)
    elif "srcData" in payload:
        raw_data = payload["srcData"]
        data = base64StringToBytes(raw_data)
    elif "srcFile" in payload:
        raw_data = payload["srcFile"]
        print(f'raw_data = {raw_data}')
    else:
        raise Exception("Unable to determine datasource in payload")

    with io.BytesIO(data) as inmemfile:
        file_type = imghdr.what(inmemfile)
        if file_type not in ALLOWED_TYPES:
            raise Exception(F"Unsupported file type, expected one of : {ALLOWED_TYPES}")

    print(f'Filetype : {file_type}')
    return data, file_type

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
        payload = request.json
        if payload is None:
            return {"error": "empty payload"}, 200

        message_bytes, file_type = extract_payload(payload)
        pms_mode = PSMode.fromValue(payload["mode"] if 'mode' in payload else '')

        m = hashlib.sha256()
        m.update(message_bytes)
        checksum = m.hexdigest()

        upload_dir = ensure_exists(f'/tmp/marie/{queue_id}')
        ext = TYPES_TO_EXT[file_type]

        # convert to numpy array
        npimg = np.fromstring(message_bytes, np.uint8)
        # convert numpy array to image
        img = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)
        tmp_file = f"{upload_dir}/{checksum}.{ext}"
        cv2.imwrite(tmp_file, img)

        loaded, frames = loadImage(tmp_file)
        if not loaded:
            print(f'Unable to read image : {tmp_file}')
            raise Exception(f'Unable to read image : {tmp_file}')

        page_index = 0
        img = frames[page_index]
        h = img.shape[0]
        w = img.shape[1]
        # allow for small padding around the component
        padding = 4
        overlay = np.ones((h + padding * 2, w + padding * 2, 3), dtype=np.uint8) * 255
        overlay[padding:h + padding, padding:w + padding] = img

        # cv2.imwrite('/tmp/marie/overlay.png', overlay)

        boxes, img_fragments, lines, _ = box_processor.extract_bounding_boxes(
            queue_id, checksum, overlay, pms_mode)
        result, overlay_image = icr_processor.recognize(
            queue_id, checksum, overlay, boxes, img_fragments, lines)

        cv2.imwrite(f'/tmp/marie/overlay_image_{page_index}.png', overlay_image)

        result['overlay_b64'] = encodeToBase64(overlay_image)
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
