
from PIL import ImageFont, ImageDraw, Image  
import numpy as np
import os

def drawTrueTypeTextOnImage(cv2Image, text, xy, size, fillColor):
    """
    Print True Type fonts using PIL and convert image back into OpenCV
    """
    # Pass the image to PIL  
    pil_im = Image.fromarray(cv2Image)  
    draw = ImageDraw.Draw(pil_im)  
    # use a truetype font  
    fontFace = np.random.choice([ "FreeMono.ttf", "FreeMonoBold.ttf", "FreeMonoBold.ttf", "FreeSans.ttf"]) 
    fontPath = os.path.join("./assets/fonts/truetype", "FreeMono.ttf")

    font = ImageFont.truetype(fontPath, size)
    draw.text(xy, text, font=font, fill=fillColor)  
    # Make Numpy/OpenCV-compatible version
    cv2Image = np.array(pil_im)
    return cv2Image