import io

import cv2
from PIL import Image
from PyPDF4 import PdfFileWriter, PdfFileReader
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from renderer.renderer import ResultRenderer


class AdlibRenderer(ResultRenderer):
    def __init__(self, config=None):
        super().__init__(config)
        if config is None:
            config = {}
        print(f"AdlibRenderer base : {config}")

        self.page_number = -1
        if "page_number" in config:
            self.page_number = str(config["page_number"])

    @property
    def name(self):
        return "AdlibRenderer"

    def render(self, img, result, output_filename):
        import xml.etree.ElementTree as gfg
        try:
            meta = result["meta"]
            words = result["words"]
            lines = result["lines"]

            print(meta)
            im_h = meta['imageSize']['height'] / 72.0
            im_w = meta['imageSize']['width'] / 72.0

            root = gfg.Element("PAGE")
            root.set("HEIGHT", str(im_h))
            root.set("WIDTH", str(im_w))
            root.set("ImageType", "Unknown")
            root.set("NUMBER", self.page_number)
            root.set("OCREndTime", "0")
            root.set("OCRStartTime", "0")
            root.set("Producer", "marie")
            root.set("XRESOLUTION", "300")
            root.set("YRESOLUTION", "300")

            # Add dummy TEXT element
            root.append(gfg.Element("TEXT"))

            for idx, word in enumerate(words):
                x1, y1, w1, h1 = word["box"]
                txt = word["text"]
                x = x1 / 72.0
                y = y1 / 72.0
                w = w1 / 72.0
                h = h1 / 72.0
                left = x
                right = x + w
                top = y - h
                bottom = y + h
                consecutive = False

                m1 = gfg.Element("TEXTSTRING")
                m1.set("CONSECUTIVE", "FALSE")
                m1.set("FONTNAME", "Courier")
                m1.set("FONTSIZE", "32")
                m1.set("NoLocation", "FALSE")
                m1.set("PageNumber", str(self.page_number))

                m1.set("LEFT", str(left))
                m1.set("RIGHT", str(right))
                m1.set("TOP", str(top))
                m1.set("BOTTOM", str(bottom))
                m1.set("WORD", str(txt))

                root.append(m1)

            tree = gfg.ElementTree(root)
            with open(output_filename, "wb") as files:
                tree.write(files)

        except Exception as ident:
            raise ident
