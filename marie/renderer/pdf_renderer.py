import io
from os import PathLike
from typing import Any, Dict, Union, Optional, Callable

import cv2
import numpy as np
from PIL import Image
from PyPDF4 import PdfFileReader, PdfFileWriter
from PyPDF4.pdf import PageObject
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from marie.logging.predefined import default_logger
from marie.renderer.renderer import ResultRenderer
from marie.utils.draw_truetype import determine_font_size

logger = default_logger

# https://github.com/JonathanLink/PDFLayoutTextStripper
# https://github.com/JonathanLink/PDFLayoutTextStripper/blob/master/src/main/java/io/github/jonathanlink/PDFLayoutTextStripper.java


class PdfRenderer(ResultRenderer):
    def __init__(self, config=None):
        super().__init__(config)

    @property
    def name(self):
        return "PdfRenderer"

    def __render_page(
        self,
        image: np.ndarray,
        result: Dict[str, Any],
        page_index: int,
        overlay: bool = True,
    ) -> PageObject:
        """
        Render individual page as `PageObject`

        :return: the page from PdfFileReader
        """
        try:
            self.logger.debug(f"Rendering overlay = {overlay} page #: {page_index}")

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

            packet = io.BytesIO()
            # with contextlib.closing(io.BytesIO()) as packet:
            can = canvas.Canvas(packet, pagesize=(img_w, img_h))

            if overlay:
                # convert OpenCV to Pil
                img_bgr = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                im_pil = Image.fromarray(img_bgr)
                can.drawImage(ImageReader(im_pil), 0, 0)
            else:
                # set canvas text to black
                can.setFillColorRGB(1, 1, 1)
                can.rect(0, 0, img_w, img_h, fill=1)
                can.setFillColorRGB(0, 0, 0)

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
                    py0 = img_h - y - h * 0.80  # + (h / 2)
                    lh = h

                    # Find baseline for the word
                    rat = h / w
                    if wid in word2line:
                        line = word2line[wid]
                        line_bbox = line["bbox"]
                        ly = line_bbox[1]
                        lh = line_bbox[3]
                        # py0 = ly + lh * 0.70
                        py0 = img_h - ly - lh * 0.80
                        # py0 = img_h - y # + (h / 2)

                    # this is a hack to get the font size and text for vertical text
                    # this needs to be done in text detection and recognition
                    font_size = determine_font_size(lh)
                    if rat > 4.0:
                        font_size = 14

                    # print(f"font_size = {font_size}  : {box} :{rat} : {text}")
                    # ['Courier', 'Courier-Bold', 'Courier-BoldOblique', 'Courier-Oblique', 'Helvetica', 'Helvetica-Bold', 'Helvetica-BoldOblique', 'Helvetica-Oblique', 'Symbol', 'Times-Bold', 'Times-BoldItalic', 'Times-Italic', 'Times-Roman', 'ZapfDingbats']

                    can.setFont("Courier", font_size)
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
        frames: np.ndarray,
        results: [Dict[str, Any]],
        output_filename: Union[str, PathLike],
        filename_generator: Optional[Callable[[int], str]] = None,
        **kwargs: Any,
    ) -> None:
        """Renders results into PDF output stream
        Results parameter "format" is expected to be in "XYWH" conversion will be performed to accommodate this
        """
        image_overlay = kwargs.get("overlay", True)

        self.logger.info(f"Render PDF [{image_overlay}]: {output_filename}")

        # The underlying ByteIO buffer will be closed when we write the file out
        writer = PdfFileWriter()
        for page_index, (image, result) in enumerate(zip(frames, results)):
            try:
                page = self.__render_page(image, result, page_index, image_overlay)
                writer.addPage(page)
            except Exception as e:
                logger.error(e, stack_info=True, exc_info=True)

        with open(output_filename, "wb") as output:
            writer.write(output)
