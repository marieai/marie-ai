import numpy as np
from PIL import Image

import cv2


def read_image(image):
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
        elif len(image.shape) == 3 and image.shape[2] == 3:  # BGRscale
            img = image
        elif len(image.shape) == 3 and image.shape[2] == 4:  # RGBAscale
            img = image[:, :, :3]
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img


def paste_fragment(overlay, fragment, pos=(0, 0)):
    # You may need to convert the color.
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
