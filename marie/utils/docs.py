import imghdr
import io
import os
import tempfile
from typing import Any, List, Union

import cv2
import numpy as np
import PyPDF4
import skimage.io as skio
from PIL import Image
from PyPDF4 import PdfFileReader
from PyPDF4.utils import PdfReadError

from marie import Document, DocumentArray

ALLOWED_TYPES = {"png", "jpeg", "tiff", "pdf"}
TYPES_TO_EXT = {"png": "png", "jpeg": "jpg", "tiff": "tif", "pdf": "pdf"}


def get_document_type(file_path: str):
    """Get document type"""

    if not os.path.exists(file_path):
        raise Exception(f"File not found : {file_path}")

    with open(file_path, "rb") as file:
        data = file.read()

    with io.BytesIO(data) as memfile:
        file_type = imghdr.what(memfile)

    if file_type is None:
        try:
            PyPDF4.PdfFileReader(open(file_path, "rb"))
            file_type = "pdf"
        except PdfReadError:
            print("invalid PDF file")
        else:
            pass

    if f"{file_type}".lower() not in ALLOWED_TYPES:
        raise Exception(f"Unsupported file type, expected one of : {ALLOWED_TYPES}")

    return file_type


def _handle_filter_real(x_object, obj, mode, size, data):
    resource_id = obj[1:]

    if "/Filter" in x_object[obj]:
        x_filter = x_object[obj]["/Filter"]
        if x_filter == "/FlateDecode":
            pass
        elif x_filter == "/DCTDecode":
            with open(resource_id + ".jpg", "wb") as img:
                img.write(data)
            return
        elif x_filter == "/JPXDecode":
            with open(resource_id + ".jp2", "wb") as img:
                img.write(data)
            return
        elif x_filter == "/CCITTFaxDecode":
            with open(resource_id + ".tiff", "wb") as img:
                img.write(data)
            return

    img = Image.frombytes(mode, size, data)
    img.save(resource_id + ".png")


def _handle_filter(x_object, obj, mode, size, data):
    resource_id = obj[1:]

    # we are writing the data to temp dir and then read it back in
    fd, path = tempfile.mkstemp()
    try:
        with os.fdopen(fd, "wb") as tmp:
            tmp.write(data)
        image_type = get_document_type(path)
        img = None
        if image_type == "tiff":
            loaded, frames = cv2.imreadmulti(path, [], cv2.IMREAD_ANYCOLOR)
            if loaded:
                img = frames[0]
        else:
            img = skio.imread(path)  # RGB order
            if img.shape[0] == 2:
                img = img[0]
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            if img.shape[2] == 4:
                img = img[:, :, :3]

        return img
    finally:
        os.remove(path)


def load_pdf_frames(pdf_file_path):
    """Load PDF as set of Numpy Images"""
    with open(pdf_file_path, "rb") as f:
        pdf = PdfFileReader(f)
        information = pdf.getDocumentInfo()
        number_of_pages = pdf.getNumPages()

        txt = f"""
        Information about {pdf_file_path}: 
        Producer: {information.producer}
        Number of pages: {number_of_pages}
        """
        print(txt)
        frames = []
        for page_index in range(number_of_pages):
            page = pdf.getPage(page_index)
            x_object = page["/Resources"]["/XObject"].getObject()

            for obj in x_object:
                if x_object[obj]["/Subtype"] == "/Image":
                    size = (x_object[obj]["/Width"], x_object[obj]["/Height"])
                    data = x_object[obj].getData()
                    if x_object[obj]["/ColorSpace"] == "/DeviceRGB":
                        mode = "RGB"
                    else:
                        mode = "P"

                    img = _handle_filter(x_object, obj, mode, size, data)
                    frames.append(img)
                else:
                    blank = np.ones((size[0], size[1], 3), dtype=np.uint8) * 255
                    frames.append(blank)

        return True, frames


def convert_frames(frames, img_format):
    """each frame needs to be converted to RGB format"""
    converted = []
    for frame in frames:
        # cv to pil
        if isinstance(frame, np.ndarray):
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

            if img_format == "pil":
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)

        converted.append(frame)
    return converted


def load_image(img_path, img_format: str = "cv"):
    """
    Load image, if the image is a TIFF, we will load the image as a multipage tiff, otherwise we return an
    array with image as first element. If the document is a PDF we will extract images from the document and
    return them as frames

    Args:
        img_path: source image path
        img_format: cv or pil
    """

    if img_path is None:
        return False, None

    image_type = get_document_type(img_path)
    loaded = False
    frames = []

    if image_type == "pdf":
        loaded, frames = load_pdf_frames(img_path)
        if not loaded:
            return False, []
    elif image_type == "tiff":
        loaded, frames = cv2.imreadmulti(img_path, [], cv2.IMREAD_ANYCOLOR)
        if not loaded:
            return False, []

    # each frame needs to be converted to RGB format to keep proper shape [x,y,c]
    if loaded:
        converted = convert_frames(frames, img_format)
        return loaded, converted

    img = skio.imread(img_path)  # RGB order
    # img = Image.open(img_path).convert('RGB')
    # return True, [img]

    if img.shape[0] == 2:
        img = img[0]
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:
        img = img[:, :, :3]

    # cv or pil
    # if we converted with np.array then the PIl image get converted to ndarray
    if img_format == "pil":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        return True, [img]

    img = np.array(img)
    return True, [img]


def array_from_docs(docs: DocumentArray):
    """Convert DocumentArray to Numpy Array"""
    frames = []
    for doc in docs:
        frames.append(doc.tensor)

    # each tensor can be of different size that is why we are using 'concatenate' instead of 'vstack'
    # concat = np.concatenate(frames, axis=None)
    assert len(frames) == len(docs)
    return frames


def docs_from_file(img_path: str) -> DocumentArray:
    """Create DocumentArray from image"""
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"File not found : {img_path}")

    loaded, frames = load_image(img_path)
    docs = DocumentArray()

    if loaded:
        for frame in frames:
            document = Document(content=frame)
            docs.append(document)

    return docs


def docs_from_file_specific(img_path: str, pages: list) -> DocumentArray:
    """Create DocumentArray from image containing only specific frames"""
    loaded_docs = docs_from_file(img_path)
    docs = DocumentArray()

    if len(loaded_docs) > 0:
        if len(pages) == 0:
            return loaded_docs

        for i, doc in enumerate(loaded_docs):
            if i in pages:
                docs.append(doc)

    return docs


def frames_from_file(img_path: str) -> np.ndarray:
    """Create Numpy frame array from image"""
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"File not found : {img_path}")
    loaded, frames = load_image(img_path)
    if not loaded:
        raise Exception(f"Unable to load image : {img_path}")
    return frames


def is_array_like(obj):
    if hasattr(obj, "__len__") and hasattr(obj, "__getitem__"):
        return True
    return False


def docs_from_image(src: Union[Any, List]) -> DocumentArray:
    """Create DocumentArray from image or array like object
    Numpy ndarray or PIl Image ar supported
    """
    arr = src
    if not is_array_like(src):
        arr = [src]

    docs = DocumentArray()
    for img in arr:
        if isinstance(img, Image.Image):
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        document = Document(content=img)
        docs.append(document)

    return docs
