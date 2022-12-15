import glob
import os

from PyPDF4 import PdfFileReader, PdfFileWriter


def merge_pdf(src_dir, dst_pdf_path, sort_key=None):
    """Merge individual PDF frames into a multipage PDF"""
    pdf_writer = PdfFileWriter()
    print(f"Merging PDF document : {dst_pdf_path}")

    for _path in sorted(glob.glob(os.path.join(src_dir, "*.pdf")), key=sort_key):
        try:
            print(f"Adding PDF document : {_path}")
            pdf_stream = open(_path, "rb")
            new_pdf = PdfFileReader(pdf_stream)
            page = new_pdf.getPage(0)
            pdf_writer.addPage(page)
        except Exception as ident:
            raise ident

    with open(dst_pdf_path, "wb") as output:
        pdf_writer.write(output)
