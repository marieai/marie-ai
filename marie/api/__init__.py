import marie.conf
from marie.api.BoxAPI import BoxAPI, BoxListAPI
from marie.api.MarkAPI import MarkAPI, MarkListAPI
from marie.api.QueueAPI import QueueAPI, QueueListAPI
from marie.api.SegmenterAPI import SegmenterAPI, SegmenterListAPI
from flask_restful import Api


import io
import os

import hashlib
import imghdr
import numpy as np

import cv2

from marie.logging.logger import MarieLogger
from marie.utils.utils import FileSystem, ensure_exists, current_milli_time
from marie.utils.base64 import base64StringToBytes, encodeToBase64

from datetime import datetime

logger = MarieLogger("")

api = Api(
    prefix=marie.conf.API_PREFIX
)  # AttributeError: module 'config' has no attribute 'API_PREFIX

ALLOWED_TYPES = {"png", "jpeg", "tiff"}
TYPES_TO_EXT = {"png": "png", "jpeg": "jpg", "tiff": "tif"}


def store_temp_file(message_bytes, queue_id, file_type, store_raw):
    """Store temp file from decoded payload"""
    m = hashlib.sha256()
    m.update(message_bytes)
    checksum = m.hexdigest()

    upload_dir = ensure_exists(f"/tmp/marie/{queue_id}")
    ext = TYPES_TO_EXT[file_type]

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

    # This indicates that the contents are a file contents and need to stored as they appear

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

    # if we have a tiff file we need to store as RAW otherwise only first page will be converted
    if file_type == "tiff":
        store_raw = True

    tmp_file, checksum = store_temp_file(data, queue_id, file_type, store_raw)
    logger.info(f"File info: {file_type}, {tmp_file}")

    return tmp_file, checksum, file_type
