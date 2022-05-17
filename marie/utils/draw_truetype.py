import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def drawTrueTypeTextOnImage(cv2Image, text, xy, size, fillColor):
    """
    Print True Type fonts using PIL and convert image back into OpenCV
    """
    # Pass the image to PIL
    pil_im = Image.fromarray(cv2Image)
    draw = ImageDraw.Draw(pil_im)
    # use a truetype font
    try:
        fontFace = np.random.choice(["FreeMono.ttf", "FreeMonoBold.ttf", "FreeMonoBold.ttf", "FreeSans.ttf"])
        fontPath = os.path.join("./assets/fonts/truetype", "FreeMono.ttf")
        font = ImageFont.truetype(fontPath, size)
    except Exception as ex:
        font = ImageFont.load_default()

    draw.text(xy, text, font=font, fill=fillColor)
    # Make Numpy/OpenCV-compatible version
    cv2Image = np.array(pil_im)
    return cv2Image
