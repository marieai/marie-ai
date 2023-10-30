import imghdr
import io
import os
import tempfile
from typing import Any, List, Union, Optional, AnyStr

import cv2
import numpy as np
import PyPDF4
import skimage.io as skio
from PIL import Image
from PyPDF4 import PdfFileReader
from PyPDF4.utils import PdfReadError
from docarray import DocList
from docarray.documents import ImageDoc

from marie import Document, DocumentArray
from marie._core.definitions.events import AssetKey
from marie.api.docs import MarieDoc
from marie.common.file_io import StrOrBytesPath
from marie.storage import StorageManager
from marie.utils.utils import ensure_exists

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
                conv = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            else:
                conv = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if img_format == "pil":
                conv = Image.fromarray(frame.copy())
            converted.append(conv)
        else:
            converted.append(frame.copy())
    return converted


def load_image(img_path, img_format: str = "cv") -> (bool, List[np.ndarray]):
    """
    Load image, if the image is a TIFF, we will load the image as a multipage tiff, otherwise we return an
    array with image as first element. If the document is a PDF we will extract images from the document and
    return them as frames

    :param img_path: source image path
    :param img_format: cv or pil
    :return (bool, List[np.ndarray]): loaded, frames
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
        del frames
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
        img = Image.fromarray(img.copy())
        return True, [img]

    # img = np.array(img)
    return True, [img]


def frames_from_docs(
    docs: DocList[MarieDoc], field: Optional[str] = None
) -> List[np.ndarray]:
    """Convert DocList[MarieDoc] to Numpy Array"""
    if docs is None:
        raise ValueError("No documents provided to convert to array")
    frames = []
    if field is None:
        field = 'tensor'

    for doc in docs:
        frames.append(getattr(doc, field))

    # each tensor can be of different size that is why we are using 'concatenate' instead of 'vstack'
    # concat = np.concatenate(frames, axis=None)
    assert len(frames) == len(docs)
    return frames


def docs_from_file(
    path: StrOrBytesPath, pages: Optional[List[int]] = None
) -> DocList[MarieDoc]:
    """
    Create DocumentArray from image file. This will create one document per page in the image file, if the image
    is large and has many pages this can be very memory intensive.

    :param path:  path to image file
    :param pages:  list of pages to extract from document NONE or empty list will extract all pages from document
    :return: DocumentArray with tensor content
    """
    if path is not None:
        path = os.path.expanduser(path)

    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found : {path}")

    loaded, frames = load_image(path)
    # no pages specified, we will use all pages as documents
    if pages is None or len(pages) == 0:
        pages = [i for i in range(len(frames))]

    docs = DocList[MarieDoc]()
    if loaded:
        for idx, frame in enumerate(frames):
            if idx not in pages:
                continue
            doc = MarieDoc(tensor=frame)
            docs.append(doc)
    return docs


def docs_from_asset(
    asset_key: str, pages: Optional[List[int]] = None
) -> DocList[MarieDoc]:
    """
    Create DocumentArray from image file. This will create one document per page in the image file, if the image
    is large and has many pages this can be very memory intensive.

    :param asset_key:  asset key to the resource
    :param pages:  list of pages to extract from document NONE or empty list will extract all pages from document
    :return: DocList[MarieDoc] with tensor content
    """

    uri = asset_key
    import tempfile

    if not StorageManager.can_handle(uri, allow_native=True):
        raise Exception(
            f"Unable to read file from {uri} no suitable storage manager configured"
        )

    # make sure the directory exists
    ensure_exists(f"/tmp/marie")
    # Read remote file to a byte array
    with tempfile.NamedTemporaryFile(dir="/tmp/marie", delete=False) as temp_file_out:
        # with open("/tmp/sample.tiff", "w") as temp_file_out:
        # print(f"Reading file from {uri} to {temp_file_out.name}")
        if not StorageManager.exists(uri):
            raise Exception(f"Remote file does not exist : {uri}")

        StorageManager.read_to_file(uri, temp_file_out, overwrite=True)
        # Read the file to a byte array
        temp_file_out.seek(0)
        data = temp_file_out.read()
        path = temp_file_out.name

        with io.BytesIO(data) as memfile:
            file_type = imghdr.what(memfile)

        if file_type not in ALLOWED_TYPES:
            raise Exception(f"Unsupported file type, expected one of : {ALLOWED_TYPES}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found : {path}")

    loaded, frames = load_image(path)
    # no pages specified, we will use all pages as documents
    if pages is None or len(pages) == 0:
        pages = [i for i in range(len(frames))]

    docs = DocList[MarieDoc]()

    if loaded:
        for idx, frame in enumerate(frames):
            if idx not in pages:
                continue
            doc = MarieDoc(tensor=frame)
            docs.append(doc)
    return docs


def frames_from_file(img_path: StrOrBytesPath) -> List[np.ndarray]:
    """Create Numpy frame array from image"""
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"File not found : {img_path}")
    loaded, frames = load_image(img_path)
    if not loaded:
        raise Exception(f"Unable to load image : {img_path}")
    return frames


def is_array_like(obj: Any) -> bool:
    """Check if object is array like"""
    if hasattr(obj, "__len__") and hasattr(obj, "__getitem__"):
        return True
    return False


def docs_from_image(src: Union[Any, List]) -> DocList[MarieDoc]:
    """Create DocumentArray from image or array like object
    Numpy ndarray or PIl Image ar supported
    """
    arr = src
    if not is_array_like(src):
        arr = [src]

    docs = DocList[MarieDoc]()
    for img in arr:
        if isinstance(img, Image.Image):
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        doc = MarieDoc(tensor=img)
        docs.append(doc)

    return docs
