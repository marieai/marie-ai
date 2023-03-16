import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def drawTrueTypeTextOnImage(
    frame: np.ndarray,
    text: str,
    xy: tuple = (0, 0),
    size: int = 16,
    fill: tuple = (0, 0, 0),
):
    """
    Print True Type fonts using PIL and convert image back into OpenCV
    NOTE : This is a slow operation and should be avoided in loops, typically we should only use this for debugging
    """
    # Pass the image to PIL
    pil_im = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_im)
    # use a truetype font
    try:
        fontFace = np.random.choice(
            ["FreeMono.ttf", "FreeMonoBold.ttf", "FreeMonoBold.ttf", "FreeSans.ttf"]
        )
        fontPath = os.path.join("./assets/fonts/truetype", "FreeMono.ttf")
        font = ImageFont.truetype(fontPath, size)
    except Exception as ex:
        font = ImageFont.load_default()

    draw.text(xy, str(text), font=font, fill=fill)
    # Make Numpy/OpenCV-compatible version
    cv2Image = np.array(pil_im)
    return cv2Image


def get_default_font(size: int = 16):
    # use a truetype font
    try:
        fontFace = np.random.choice(
            ["FreeMono.ttf", "FreeMonoBold.ttf", "FreeMonoBold.ttf", "FreeSans.ttf"]
        )
        fontPath = os.path.join("./assets/fonts/truetype", "FreeMono.ttf")
        font = ImageFont.truetype(fontPath, size)
    except Exception as ex:
        font = ImageFont.load_default()
    return font


def determine_font_size(line_height: int) -> int:
    """
    Try to determine font size
    https://i.stack.imgur.com/3r3Ja.png
    https://medium.com/@zkareemz/golden-ratio-62b3b6d4282a
    """
    #  line-height = font-size * 1.42857
    font_size = int(line_height / 1.618)
    return font_size
