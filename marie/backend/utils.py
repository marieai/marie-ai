import numpy as np
from PIL import Image, ImageChops


def crop_whitespace(img: np.ndarray, padding: int = 0) -> np.ndarray:
    """Crop white borders from a rendered image."""
    pil_img = Image.fromarray(img)
    bg_color = pil_img.getpixel((0, 0))
    bg = Image.new(pil_img.mode, pil_img.size, bg_color)
    diff = ImageChops.difference(pil_img, bg)
    bbox = diff.getbbox()
    if bbox:
        left, upper, right, lower = bbox
        left = max(0, left - padding)
        upper = max(0, upper - padding)
        right = min(pil_img.width, right + padding)
        lower = min(pil_img.height, lower + padding)
        pil_img = pil_img.crop((left, upper, right, lower))
    return np.array(pil_img, dtype=np.uint8)
