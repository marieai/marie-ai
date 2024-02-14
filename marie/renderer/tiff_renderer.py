import json
from os import PathLike
from typing import Any, Callable, Dict, Optional, Union

import cv2
import numpy as np
import tifffile
from PIL import Image, ImageDraw, ImageFont

from marie.logging.predefined import default_logger
from marie.renderer.renderer import ResultRenderer
from marie.utils.draw_truetype import determine_font_size

logger = default_logger


class TiffRenderer(ResultRenderer):
    def __init__(self, config=None):
        super().__init__(config)

    @property
    def name(self):
        return "MARIE-AI-TiffRenderer"

    def __render_page(
        self,
        image: np.ndarray,
        result: Dict[str, Any],
        page_index: int,
        overlay: bool = True,
    ) -> Image.Image:
        """
        Render individual page as `Image`

        :return: PIL Image object
        """
        try:
            self.logger.debug(f"Rendering overlay = {overlay} page #: {page_index}")

            if image is None:
                raise Exception("Image or list of images expected")

            self.check_format_xywh(result, True)
            meta = result.get("meta", [])
            words = result.get("words", [])
            lines = result.get("lines", [])

            word2line = {wid: line for line in lines for wid in line["wordids"]}

            img_h = image.shape[0]
            img_w = image.shape[1]

            # only generate tiff with cleaned image
            if overlay:
                img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(img_pil)
            else:
                img_pil = Image.new('RGB', (img_w, img_h), 'white')
                draw = ImageDraw.Draw(img_pil)

            if len(words) == 0:
                draw.text((0, 0), "", fill=(0, 0, 0))
            else:
                for idx, word in enumerate(words):
                    wid = word["id"]
                    box = word["box"]
                    text = word["text"]
                    x, y, w, h = box

                    if len(text) == 0:
                        logger.warning(f"Empty text: {text} for box: {box}")
                        continue

                    ratio = w / h
                    cpl = len(text)
                    average_char_width = w / cpl
                    mu = 2.27
                    min_content_width = cpl * mu
                    if len(text) > 2:
                        if ratio < 0.4:
                            text = "".join([c + "\n" for c in text])
                            continue

                    # TIFF  rendering
                    left_pad = 5
                    px0 = x
                    py0 = y
                    lh = h

                    # Find baseline for the word
                    if wid in word2line:
                        line = word2line[wid]
                        line_bbox = line["bbox"]
                        ly = line_bbox[1]
                        lh = line_bbox[3]
                        py0 = ly + lh * 0.2  # best fit
                        # py0 = ly
                        # py0 = img_h - ly - lh * 0.80
                        # py0 = img_h - y # + (h / 2)

                    font_size = determine_font_size(lh)
                    try:
                        font = ImageFont.truetype("Helvetica.ttf", font_size)
                    except IOError:
                        font = ImageFont.load_default()

                    draw.text((px0 + left_pad, py0), text, font=font, fill=(0, 0, 0))

            return img_pil
        except Exception as indent:
            raise indent

    def render(
        self,
        frames: np.ndarray,
        results: list[Dict[str, Any]],
        output_filename: Union[str, PathLike],
        filename_generator: Optional[Callable[[int], str]] = None,
        **kwargs: Any,
    ) -> None:
        """Renders the results into a multi-page TIFF file."""

        image_overlay = kwargs.get("overlay", True)

        self.logger.debug(f'pages:  {len(frames)}')
        self.logger.debug(f"Rendering TIFF [{image_overlay}]: {output_filename}")
        images = [
            self.__render_page(image, result, page_index, image_overlay)
            for page_index, (image, result) in enumerate(zip(frames, results))
        ]

        metadata = {"Producer": self.name, "Number of Pages": len(images)}
        description = json.dumps(metadata)
        # Save as a multi-page TIFF
        with tifffile.TiffWriter(output_filename, bigtiff=True) as t:
            for img in images:
                t.write(data=np.array(img), description=description, compression=8)

        # check the annotated document has correct tag:
        with tifffile.TiffFile(output_filename) as t:
            page = t.pages[0]
            des = page.tags["ImageDescription"].value
            tags = json.loads(des)
            self.logger.info(f'Tiff tag: {tags["Producer"]} {tags["Number of Pages"]}')
