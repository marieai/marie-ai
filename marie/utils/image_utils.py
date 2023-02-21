import hashlib
import sys
import time
from math import ceil
from typing import Union, Tuple, List

import cv2
import io
import numpy as np
import PIL.Image
from PIL import Image
from rich import print

from marie.timer import Timer


def read_image(image):
    """Read image and convert to OpenCV compatible format"""
    img = None
    if type(image) == str:
        img = cv2.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    elif type(image) == bytes:
        nparr = np.frombuffer(image, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    elif type(image) == np.ndarray:
        if len(image.shape) == 2:  # grayscale
            img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif len(image.shape) == 3 and image.shape[2] == 3:  # BGR
            img = image
        elif len(image.shape) == 3 and image.shape[2] == 4:  # RGBA
            img = image[:, :, :3]
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif type(image) == PIL.Image.Image:  # convert pil to OpenCV
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:
        raise Exception(f"Unhandled image type : {type(image)}")

    return img


def paste_fragment(overlay, fragment, pos=(0, 0)):
    col = list(np.random.choice(range(256), size=3))
    color = [int(col[0]), int(col[1]), int(col[2])]
    fragment = cv2.copyMakeBorder(
        fragment, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=color
    )

    fragment = cv2.cvtColor(fragment, cv2.COLOR_BGR2RGB)
    fragment_pil = Image.fromarray(fragment)
    overlay.paste(fragment_pil, pos)


def viewImage(image, name="Display"):
    cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def imwrite_dpi(output_filename, cv_image, dpi=(300, 300)):

    import PIL.Image

    image = PIL.Image.fromarray(cv_image)
    image.save(output_filename, dpi=dpi)


def imwrite(output_path, img, dpi=None):
    """Save OpenCV image"""
    try:
        if dpi is None:
            cv2.imwrite(output_path, img)
        else:
            """Save OpenCV image with DPI"""
            import PIL.Image

            pil_img = PIL.Image.fromarray(img)
            pil_img.save(output_path, dpi=dpi)
    except Exception as ident:
        raise ident


def hash_file(filename):
    """ "This function returns the SHA-1 hash
    of the file passed into it"""
    # make a hash object
    h = hashlib.sha1()

    # open file for reading in binary mode
    with open(filename, "rb") as file:

        # loop till the end of the file
        chunk = 0
        while chunk != b"":
            # read only 1024 bytes at a time
            chunk = file.read(1024)
            h.update(chunk)

    # return the hex representation of digest
    return h.hexdigest()


def hash_bytes(data) -> str:
    """ "This function returns the SHA-1 hash
    of the file passed into it"""
    h = hashlib.sha1()
    h.update(data)
    # return the hex representation of digest
    return h.hexdigest()


def hash_frames_fast_Z(frames: np.ndarray, max_frame_size=1024) -> str:
    """calculate hash based on the image frame"""
    hash_src = []
    md5 = hashlib.md5()
    for _, frame in enumerate(frames):
        hash_src = np.append(
            hash_src,
            np.ravel(
                frame[
                    0 : max_frame_size
                    if len(frame) > max_frame_size
                    else 0 : len(frame)
                ]
            ),
        )

    md5.update(hash_src)
    return md5.hexdigest()
    # return hash_bytes(hash_src)


# @Timer(text="hashed in {:.4f} seconds")
def hash_frames_fast(frames: List[np.ndarray], blocksize=2**20) -> str:
    """calculate hash based on the image data frame"""
    md5 = hashlib.md5()
    for _, frame in enumerate(frames):
        buf = np.ravel(frame)
        steps = ceil(len(buf) / blocksize)
        for k in range(0, steps):
            s = k * blocksize
            e = (k + 1) * blocksize
            if e > len(buf):
                e = len(buf)
            v = buf[s:e]
            md5.update(v)
    return md5.hexdigest()


def convert_to_bytes(
    frame: Union[np.ndarray, PIL.Image.Image],
    fmt: str = "PNG",
    dpi: Tuple[int, int] = None,
) -> bytes:
    """
    Convert image frame to byte array
    @param frame:
    @param fmt:
    @param dpi:
    @return:
    """

    if isinstance(frame, np.ndarray):
        pil_img = PIL.Image.fromarray(frame)
    elif isinstance(frame, PIL.Image.Image):
        pil_img = frame
    else:
        raise TypeError(f"Unsupported type : {type(frame)}")
    img_byte_arr = io.BytesIO()
    pil_img.save(img_byte_arr, format=fmt, dpi=dpi)
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr


def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened


def crop_to_content(frame: np.ndarray, content_aware=True) -> np.ndarray:
    """
    Crop given image to content
    No content is defined as first non background(white) pixel.

    @param frame: the image frame to process
    @param content_aware: if enabled we will apply more aggressive crop method
    @return: new cropped frame
    """

    start = time.time()
    # conversion required, or we will get 'Failure to use adaptiveThreshold: CV_8UC1 in function adaptiveThreshold'
    # frame = np.random.choice([0, 255], size=(32, 32), p=[0.01, 0.99]).astype("uint8")
    cv2.imwrite("/tmp/fragments/frame-src.png", frame)

    # Transform source image to gray if it is not already
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if content_aware:
        # apply division normalization to preprocess the image
        blur = cv2.GaussianBlur(gray, (5, 5), sigmaX=0, sigmaY=0)
        # divide
        divide = cv2.divide(gray, blur, scale=255)
        thresh = cv2.threshold(divide, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        #
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 3))
        op_frame = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    else:
        op_frame = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    cv2.imwrite("/tmp/fragments/op_frame.png", op_frame)
    indices = np.array(np.where(op_frame == [0]))
    img_w = op_frame.shape[1]
    img_h = op_frame.shape[0]
    min_x_pad = 16  # img_w // 16
    min_y_pad = img_h // 4

    print(min_x_pad)
    print(min_y_pad)

    # indices are in y,X format
    if content_aware:
        x = max(0, indices[1].min() - min_x_pad)
        y = 0  # indices[0].min()
        h = img_h  # indices[0].max() - y
        w = min(img_w, indices[1].max() - x + min_x_pad)
    else:
        x = indices[1].min()
        y = indices[0].min()
        h = indices[0].max() - y
        w = indices[1].max() - x

    print(x, y, w, h)
    cropped = frame[y : y + h + 1, x : x + w + 1].copy()
    cv2.imwrite("/tmp/fragments/cropped.png", cropped)

    dt = time.time() - start
    return cropped
