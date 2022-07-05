import io
import os
import cv2
import numpy as np
import imghdr
from PIL import Image, ImageDraw, ImageFont

from marie import DocumentArray, Document

ALLOWED_TYPES = {"png", "jpeg", "tiff"}
TYPES_TO_EXT = {"png": "png", "jpeg": "jpg", "tiff": "tif"}


def get_image_type(file_path: str):

    if not os.path.exists(file_path):
        raise Exception(f"File not found : {file_path}")

    with open(file_path, "rb") as file:
        data = file.read()

    with io.BytesIO(data) as memfile:
        file_type = imghdr.what(memfile)

    if file_type not in ALLOWED_TYPES:
        raise Exception(f"Unsupported file type, expected one of : {ALLOWED_TYPES}")

    return file_type


def load_image(fname, format: str = "cv"):
    """
    Load image, if the image is a TIFF, we will load the image as a multipage tiff, otherwise we return an
    array with image as first element

    Args:
        format: cv or pil
    """
    import skimage.io as skio

    if fname is None:
        return False, None

    image_type = get_image_type(fname)

    if image_type == "tiff":
        loaded, frames = cv2.imreadmulti(fname, [], cv2.IMREAD_ANYCOLOR)
        if not loaded:
            return False, []
        # each frame needs to be converted to RGB format
        converted = []
        for frame in frames:
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                # cv or pil
                if format == "pil":
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = Image.fromarray(frame)
            converted.append(frame)
        return loaded, converted

    img = skio.imread(fname)  # RGB order
    if img.shape[0] == 2:
        img = img[0]
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:
        img = img[:, :, :3]

    # cv or pil
    # if we converted twith np.array then the PIl image get converted to ndarray
    if format == "pil":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        return True, [img]

    img = np.array(img)
    return True, [img]


def array_from_docs(docs: DocumentArray) -> np.ndarray:
    """Convert DocumentArray to Numpy Array"""
    frames = []
    for doc in docs:
        tensor = doc.tensor
        frames.append(tensor)

    return np.array(frames)


def docs_from_file(img_path: str) -> DocumentArray:
    """Create DocumentArray from image"""
    if not os.path.exists(img_path):
        raise Exception(f"File not found : {img_path}")

    loaded, frames = load_image(img_path)
    docs = DocumentArray()

    for frame in frames:
        document = Document()
        document.tensor = frame
        docs.append(document)

    return docs

def docs_from_file_raw(img_path: str) -> DocumentArray:
    """Create DocumentArray from image"""
    if not os.path.exists(img_path):
        raise Exception(f"File not found : {img_path}")

    loaded, frames = load_image(img_path)
    docs = DocumentArray()

    for frame in frames:
        document = Document()
        document.tensor = frame
        docs.append(document)

    return docs

def docs_from_image(img) -> DocumentArray:
    """Create DocumentArray from image
    Numpy ndarray or PIl Image ar supported
    """

    if isinstance(img, Image.Image):
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    docs = DocumentArray()
    document = Document()
    document.tensor = img
    docs.append(document)

    return docs
