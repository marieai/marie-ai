import io

import cv2
from PIL import Image
from PyPDF4 import PdfFileWriter, PdfFileReader
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from renderer.renderer import ResultRenderer


class PdfRenderer(ResultRenderer):
    def __init__(self, config=None):
        super().__init__(config)
        if config is None:
            config = {}
        print(f"PdfRenderer base : {config}")

    @property
    def name(self):
        return "PdfRenderer"

    def render(self, img, result, output_filename):
        try:
            print("Rendering ...")
            meta = result["meta"]
            words = result["words"]
            lines = result["lines"]

            img_h = img.shape[0]
            img_w = img.shape[1]

            # convert OpenCV to Pil
            img_bgr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            im_pil = Image.fromarray(img_bgr)

            packet = io.BytesIO()
            can = canvas.Canvas(packet, pagesize=(img_w, img_h))
            can.setFontSize(32)
            can.drawImage(ImageReader(im_pil), 0, 0)

            for idx, word in enumerate(words):
                box = word["box"]
                text = word["text"]
                x, y, w, h = box
                # PDF rendering transformation
                # x and y define the lower left corner of the image, so we need to perform some transformations
                px0 = x
                py0 = img_h - y - h * 0.70  # + (h / 2)
                can.drawString(px0, py0, text)

            can.save()
            # move to the beginning of the StringIO buffer
            packet.seek(0)
            new_pdf = PdfFileReader(packet)
            page = new_pdf.getPage(0)

            pdfwriter = PdfFileWriter()
            pdfwriter.addPage(page)

            with open(output_filename, "wb") as output:
                pdfwriter.write(output)

        except Exception as ident:
            raise  ident
