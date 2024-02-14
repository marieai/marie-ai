import io
import json
import os

import cv2
import numpy
from PIL import Image

from marie.boxes import BoxProcessorUlimDit
from marie.boxes.box_processor import PSMode
from marie.document import TrOcrProcessor
from marie.document.craft_ocr_processor import CraftOcrProcessor
from marie.numpyencoder import NumpyEncoder
from marie.renderer.tiff_renderer import TiffRenderer
from marie.utils.utils import ensure_exists


def __scale_width(src, target_size, crop_size, method=Image.BICUBIC):
    img = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    ow, oh = img.size
    if ow == target_size and oh >= crop_size:
        return img
    w = target_size
    h = int(max(target_size * oh / ow, crop_size))

    pil_image = img.resize((w, h), method)
    open_cv_image = numpy.array(pil_image)
    # Convert RGB to BGR
    open_cv_image = open_cv_image[:, :, ::-1]
    return open_cv_image


# https://stackoverflow.com/questions/23853632/which-kind-of-interpolation-best-for-resizing-image
def __scale_height(img, target_size, crop_size, method=Image.LANCZOS):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)

    ow, oh = img.size
    scale = oh / target_size
    print(scale)
    w = ow / scale
    h = target_size  # int(max(oh / scale, crop_size))
    return img.resize((int(w), int(h)), method)


def from_json_file(filename):
    with io.open(filename, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)
        return data


if __name__ == "__main__":
    import faulthandler

    faulthandler.enable()

    work_dir_boxes = ensure_exists("/tmp/boxes")
    work_dir_icr = ensure_exists("/tmp/icr")
    ensure_exists("/tmp/fragments")

    img_path = "./assets/english/Lines/005.png"
    img_path = "~/tmp/analysis/marie-issues/107/195668453-0004.png"
    img_path = "~/tmp/analysis/marie-issues/106/195664193-0006.png"
    img_path = "~/Desktop/11302023_28082_5_452_.tif"
    img_path = os.path.expanduser(img_path)

    if not os.path.exists(img_path):
        raise Exception(f"File not found : {img_path}")

    if True:
        key = img_path.split("/")[-1]
        image = cv2.imread(img_path)

        box = BoxProcessorUlimDit(work_dir=work_dir_boxes, cuda=True)
        icr = TrOcrProcessor(work_dir=work_dir_icr, cuda=True)
        # icr = TesseractOcrProcessor(work_dir=work_dir_icr, cuda=True)
        # icr = CraftOcrProcessor(work_dir=work_dir_icr, cuda=True)

        (
            boxes,
            fragments,
            lines,
            _,
            lines_bboxes,
        ) = box.extract_bounding_boxes(key, "field", image, PSMode.SPARSE)

        print(lines)
        result, overlay_image = icr.recognize(
            key, "test", image, boxes, fragments, lines, return_overlay=True
        )

        print("Results -----------------")
        print(result['meta'])
        print(result['words'])
        print(result['lines'])
        cv2.imwrite("/tmp/fragments/overlay.png", overlay_image)
        json_path = os.path.join("/tmp/fragments", "results.json")

        with open(json_path, "w") as json_file:
            json.dump(
                result,
                json_file,
                sort_keys=True,
                separators=(",", ": "),
                ensure_ascii=False,
                indent=2,
                cls=NumpyEncoder,
            )

        # output_filename = "/tmp/fragments/result.pdf"
        # print("Testing pdf render")
        #
        # renderer = PdfRenderer(config={"preserve_interword_spaces": True})
        # renderer.render(image, result, output_filename)

    if True:
        key = img_path.split("/")[-1]
        image = cv2.imread(img_path)
        print(f'images {image}')

        output_filename = "/tmp/fragments/result.tif"
        print("Testing tiff render")
        results = from_json_file("/tmp/fragments/results.json")

        # renderer = PdfRenderer(config={"preserve_interword_spaces": True})
        # renderer.render(image, results, output_filename)

        renderer = TiffRenderer(config={"preserve_interword_spaces": True})
        renderer.render(
            frames=[image],
            results=[results],
            output_filename=output_filename,
            **{
                "overlay": True,
            }
        )
        renderer.render(
            frames=[image],
            results=[results],
            output_filename=output_filename.replace(".tif", "_clean.tif"),
            **{
                "overlay": False,
            }
        )