import base64
import json

import cv2
import numpy as np


def encodeimg2b64(img: np.ndarray) -> str:
    """encode image to base64"""
    retval, buffer = cv2.imencode(".png", img)
    png_as_text = base64.b64encode(buffer).decode()
    return png_as_text


class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types"""

    def default(self, obj):
        if isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):

            return int(obj)

        elif isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)

        elif isinstance(obj, (np.complex64, np.complex128)):
            return {"real": obj.real, "imag": obj.imag}

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        elif isinstance(obj, np.bool_):
            return bool(obj)

        elif isinstance(obj, np.void):
            return None

        # check if pydantic model and convert to dict
        elif hasattr(obj, "__dict__"):
            return obj.__dict__

        return json.JSONEncoder.default(self, obj)
