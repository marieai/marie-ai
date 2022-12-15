import hashlib

import cv2
import numpy as np
import PIL.Image
from PIL import Image


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
    fragment = cv2.copyMakeBorder(fragment, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=color)

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
        print(ident)


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


def hash_bytes(data):
    """ "This function returns the SHA-1 hash
    of the file passed into it"""
    h = hashlib.sha1()
    h.update(data)
    # return the hex representation of digest
    return h.hexdigest()
