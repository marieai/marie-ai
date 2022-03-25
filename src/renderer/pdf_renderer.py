import io

import cv2
from PIL import Image
from PyPDF4 import PdfFileWriter, PdfFileReader
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from renderer.renderer import ResultRenderer


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
            # can.drawImage(ImageReader(im_pil), 0, 0)

            for idx, word in enumerate(words):
                box = word["box"]
                text = word["text"]
                x, y, w, h = box
                # PDF rendering transformation
                # x and y define the lower left corner of the image, so we need to perform some transformations
                left_pad = 5  # By observation
                px0 = x
                py0 = img_h - y - h * 0.70  # + (h / 2)
                font_size = determine_font_size(box)
                # print(can.getAvailableFonts())
                font_size = 24  # h * .75
                # print(f'font_size = {font_size}  : {box}')
                # ['Courier', 'Courier-Bold', 'Courier-BoldOblique', 'Courier-Oblique', 'Helvetica', 'Helvetica-Bold', 'Helvetica-BoldOblique', 'Helvetica-Oblique', 'Symbol', 'Times-Bold', 'Times-BoldItalic', 'Times-Italic', 'Times-Roman', 'ZapfDingbats']
                can.setFont("Helvetica", font_size)
                can.setFontSize(font_size)
                can.drawString(px0 + left_pad, py0, text)

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
            raise ident
