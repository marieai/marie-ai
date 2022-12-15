import base64

import cv2
import numpy as np


def encodeToBase64(img: np.ndarray) -> str:
    """encode image to base64"""
    retval, buffer = cv2.imencode('.png', img)
    png_as_text = base64.b64encode(buffer).decode()
    return png_as_text


def base64StringToBytes(data: str):
    """convert base64 string to byte array"""
    if data is None:
        return ""
    base64_message = data
    base64_bytes = base64_message.encode('utf-8')
    message_bytes = base64.b64decode(base64_bytes)
    return message_bytes
