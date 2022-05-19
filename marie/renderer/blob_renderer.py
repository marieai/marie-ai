import io

import cv2
from PIL import Image
from PyPDF4 import PdfFileWriter, PdfFileReader
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from xml.sax.saxutils import escape, quoteattr
from marie.renderer.renderer import ResultRenderer


class BlobRenderer(ResultRenderer):
    def __init__(self, config=None):
        super().__init__(config)
        if config is None:
            config = {}
        print(f"BlobRenderer base : {config}")

        self.page_number = -1
        if "page_number" in config:
            self.page_number = str(config["page_number"])

    @property
    def name(self):
        return "BlobRenderer"

    def render(self, img, result, output_filename):
        import xml.etree.ElementTree as gfg

        root = gfg.Element("blobs")
        root.set("angle", "0.0")
        root.set("yres", "300")
        root.set("xres", "300")
        root.set("page", self.page_number)

        try:
            meta = result["meta"]
            words = result["words"]
            lines = result["lines"]

            # self.page_number
            for idx, word in enumerate(words):
                text = word["text"]
                x, y, w, h = word["box"]
                m1 = gfg.Element("blob")
                m1.set("x", str(x))
                m1.set("y", str(y))
                m1.set("w", str(w))
                m1.set("h", str(h))
                m1.set("text", escape(text))

                b1 = gfg.SubElement(m1, "page")
                b1.text = str(self.page_number)

                root.append(m1)

            tree = gfg.ElementTree(root)
            with open(output_filename, "wb") as files:
                tree.write(files)

        except Exception as ident:
            raise ident
