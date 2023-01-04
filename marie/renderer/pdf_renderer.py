import contextlib
import io
from os import PathLike
from typing import Any, Dict, Union

import cv2
import numpy as np
from PIL import Image
from PyPDF4 import PdfFileReader, PdfFileWriter
from PyPDF4.pdf import PageObject
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from marie.logging.logger import MarieLogger
from marie.logging.predefined import default_logger
from marie.renderer.renderer import ResultRenderer
from marie.utils.docs import is_array_like

logger = default_logger

# https://github.com/JonathanLink/PDFLayoutTextStripper
# https://github.com/JonathanLink/PDFLayoutTextStripper/blob/master/src/main/java/io/github/jonathanlink/PDFLayoutTextStripper.java


def determine_font_size(box):
    """
    Try to determine font size
    https://i.stack.imgur.com/3r3Ja.png
    """
    x, y, w, h = box
    fmap = {
        8: 6,
        9: 7,
        10: 7.5,
        11: 8,
        12: 9,
        13: 10,
        14: 10.5,
        15: 11,
        16: 12,
        17: 13,
        18: 13,
        19: 13,
        20: 13,
        21: 13,
        22: 13,
        23: 13,
        24: 13,
        25: 13,
        26: 13,
        27: 13,
        28: 13,
        29: 13,
        30: 13,
        31: 13,
        32: 13,
        33: 13,
        34: 13,
        35: 13,
        36: 13,
        37: 13,
        38: 13,
        39: 13,
        40: 13,
        41: 13,
        42: 13,
        43: 13,
        44: 13,
        45: 13,
        46: 32,
        47: 34,
        48: 36,
    }
    return 32


class PdfRenderer(ResultRenderer):
    def __init__(self, config=None):
        super().__init__(config)
        if config is None:
            config = {}

    @property
    def name(self):
        return "PdfRenderer"

    def __render_page(
        self, image: np.array, result: Dict[str, Any], page_index: int
    ) -> PageObject:
        """
        Render individual page as `PageObject`

        :return: the page from PdfFileReader
        """
        try:
            self.logger.info("Rendering ...")
            if image is None:
                raise Exception("Image or list of images expected")
            self.check_format_xywh(result, True)
            # ['meta', 'words', 'lines']
            meta = result["meta"]
            words = result["words"]
            lines = result["lines"]

            word2line = dict()
            for line in lines:
                for wid in line["wordids"]:
                    word2line[wid] = line

            img_h = image.shape[0]
            img_w = image.shape[1]

            draw_image = False

            # convert OpenCV to Pil
            img_bgr = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            im_pil = Image.fromarray(img_bgr)

            packet = io.BytesIO()
            # with contextlib.closing(io.BytesIO()) as packet:
            can = canvas.Canvas(packet, pagesize=(img_w, img_h))

            if draw_image:
                can.drawImage(ImageReader(im_pil), 0, 0)

            if len(words) == 0:
                can.setFont("Helvetica", 0)
                can.setFontSize(0)
                can.drawString(0, 0, "")
            else:
                for idx, word in enumerate(words):
                    wid = word["id"]
                    box = word["box"]
                    text = word["text"]
                    x, y, w, h = box
                    # PDF rendering transformation
                    # x and y define the lower left corner of the image, so we need to perform some transformations
                    left_pad = 5  # By observation
                    px0 = x
                    py0 = img_h - y - h * 0.70  # + (h / 2)

                    # Find baseline for the word
                    if wid in word2line:
                        line = word2line[wid]
                        line_bbox = line["bbox"]
                        ly = line_bbox[1]
                        lh = line_bbox[3]
                        # py0 = ly + lh * 0.70
                        py0 = img_h - ly - lh * 0.70
                        # py0 = img_h - y # + (h / 2)
                    font_size = determine_font_size(box)
                    # print(can.getAvailableFonts())
                    font_size = 12  # 24  # h * .75
                    # print(f'font_size = {font_size}  : {box}')
                    # ['Courier', 'Courier-Bold', 'Courier-BoldOblique', 'Courier-Oblique', 'Helvetica', 'Helvetica-Bold', 'Helvetica-BoldOblique', 'Helvetica-Oblique', 'Symbol', 'Times-Bold', 'Times-BoldItalic', 'Times-Italic', 'Times-Roman', 'ZapfDingbats']
                    can.setFont("Helvetica", font_size)
                    can.setFontSize(font_size)
                    can.drawString(px0 + left_pad, py0, text)

            can.save()
            # move to the beginning of the BytesIO buffer
            # the steam will be closed when the GC is run
            packet.seek(0)
            new_pdf = PdfFileReader(packet)
            page = new_pdf.getPage(0)

            return page

        except Exception as ident:
            raise ident

    def render(
        self,
        frames: [np.array],
        results: [Dict[str, Any]],
        output_filename: Union[str, PathLike],
    ) -> None:
        """Renders results into PDF output stream
        Results parameter "format" is expected to be in "XYWH" conversion will be performed to accommodate this
        """
        self.logger.info(f"Render PDF : {output_filename}")
        # The underlying ByteIO buffer will be closed when we write the file out
        writer = PdfFileWriter()
        for page_index, (image, result) in enumerate(zip(frames, results)):
            try:
                page = self.__render_page(image, result, page_index)
                writer.addPage(page)
            except Exception as e:
                logger.error(e, stack_info=True, exc_info=True)

        with open(output_filename, "wb") as output:
            writer.write(output)
