import argparse
import glob
import hashlib
import json
import os

import cv2
import numpy as np

from tools import from_json_file, ensure_exists, __tmp_path__
from marie.boxes import BoxProcessorUlimDit, PSMode
from marie.document.trocr_ocr_processor import TrOcrProcessor
from marie.boxes.line_processor import find_line_number
from marie.numpyencoder import NumpyEncoder
import logging

# Setup default logger for this file
logger = logging.getLogger(__name__)


def extract_icr(
    image, label: str, boxp, icrp, debug_fragments: bool = False
) -> (list, dict):
    """Preform ICR on an image"""
    if not isinstance(image, np.ndarray):
        raise Exception("Expected image in numpy format")

    msg_bytes = image.tobytes(order="C")
    m = hashlib.sha256()
    m.update(msg_bytes)
    checksum = m.hexdigest()

    ensure_exists(f"{__tmp_path__}/icr/{checksum}/")
    print(f"checksum = {checksum}")
    json_file = f"{__tmp_path__}/icr/{checksum}/{checksum}.json"

    # Have we extracted this image before?
    if os.path.exists(json_file):
        print(f"From JSONFILE : {json_file}")
        json_data = from_json_file(json_file)

        results = json_data["result"]
        print(f"words  = {len(results['words'])}")
        if len(results["words"]) == 0:
            print(f"Empty result for {checksum}")
            return None, None
        return json_data["boxes"], json_data["result"]

    key = checksum
    # NOTE: For now we assume there are no internal boxes to be discovered
    boxes, img_fragments, lines = [], [], [1]
    # TODO: Model needs to be trained to Extract sub-boxes from snippets
    # # Extract sub-boxes
    # if label == "other":
    #     boxes, img_fragments, lines, _, line_bboxes = boxp.extract_bounding_boxes(
    #         key, "field", image, PSMode.SPARSE)

    boxes, img_fragments, lines, _, line_bboxes = boxp.extract_bounding_boxes(
        key, "field", image, PSMode.SPARSE
    )

    # we found no boxes, so we will creat only one box and wrap a whole image as that
    if boxes is None or len(boxes) == 0:
        print(f"No internal boxes for : {checksum}")
        if debug_fragments:
            file_path = os.path.join(
                ensure_exists(f"{__tmp_path__}/icr/{checksum}/"), f"{checksum}.png"
            )
            cv2.imwrite(file_path, image)

        h = image.shape[0]
        w = image.shape[1]
        boxes = [[0, 0, w, h]]
        img_fragments = [image]
        lines = [1]

    # filter out fragments that are too small to be processed
    fragments_tmp = []
    boxes_tmp = []
    lines_tmp = []

    for idx, img in enumerate(img_fragments):
        if img.shape[0] > 10 and img.shape[1] > 10:
            fragments_tmp.append(img)
            boxes_tmp.append(boxes[idx])
            lines_tmp.append(lines[idx])
        else:
            print(f"Skipping fragment of size: {img.shape}, {boxes[idx]}")

    boxes = boxes_tmp
    img_fragments = fragments_tmp
    lines = lines_tmp

    print(
        f"AFTER  boxes = {len(boxes)} frag = {len(img_fragments)} lines = {len(lines)}"
    )
    result = {"words": []}
    if (
        label != "other"
    ):  # NOTE: 'other' Box is the entire document. Currently, no need to parse it.
        result, overlay_image = icrp.recognize(
            key, label, image, boxes, img_fragments, lines
        )

    data = {"boxes": boxes, "result": result}
    with open(json_file, "w") as f:
        json.dump(
            data,
            f,
            sort_keys=True,
            separators=(",", ": "),
            ensure_ascii=False,
            indent=4,
            cls=NumpyEncoder,
        )

    return boxes, result


def load_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)

    image = cv2.imread(image_path)
    h, w = image.shape[0], image.shape[1]
    return image, (w, h)


def __decorate_funsd(
    data: dict,
    filename: str,
    output_ann_dir: str,
    img_dir: str,
    boxp: BoxProcessorUlimDit,
    icrp: TrOcrProcessor,
    debug_fragments: bool = False,
) -> None:
    """'Decorate' a FUNSD file with ICR extracted text from the corresponding image"""
    image_path = os.path.join(img_dir, filename + ".png")
    image, size = load_image(image_path)

    print(f"Extracting line numbers with Box Processor for {filename}")

    # line_numbers : line number associated with bounding box
    # lines : raw line boxes that can be used for further processing
    _, _, line_numbers, _, line_bboxes = boxp.extract_bounding_boxes(
        filename, "lines", image
    )  # TODO: Need to investigate speed issue

    for item in data["form"]:
        # Boxes are in stored in x0,y0,x1,y1 where x0,y0 is upper left corner and x1,y1 if bottom/right
        x0, y0, x1, y1 = item["box"]
        _id = item["id"]

        snippet = image[y0:y1, x0:x1, :]
        line_number = find_line_number(line_bboxes, [x0, y0, x1 - x0, y1 - y0])
        debug_fragments = True
        # each snippet could be on multiple lines
        print(f"line_number = {line_number}")
        # export cropped region
        if debug_fragments:
            file_path = os.path.join(
                f"{__tmp_path__}/snippet", f"{filename}-snippet_{_id}.png"
            )
            cv2.imwrite(file_path, snippet)

        boxes, results = extract_icr(
            snippet, item["label"], boxp, icrp, debug_fragments
        )

        if (
            results is None
            or len(results) == 0
            or "lines" not in results
            or len(results["lines"]) == 0
        ):
            print(f"*No results in {filename} for id:{_id}")
            continue

        results.pop("meta", None)

        words = []
        text = " ".join([line["text"] for line in results["lines"]])
        for word in results["words"]:
            # result word boxes are in x,y,w,h local position .
            x, y, w, h = word["box"]
            # Converting to relative position to account for the offset of the snippet box.
            word_box = [x0 + x, y0 + y, x0 + x + w, y0 + y + h]
            adj_word = {"text": word["text"], "box": word_box}
            words.append(adj_word)

        item["words"] = words
        item["text"] = text
        item["line_number"] = line_number
        print("-------------------------------")
        print(f"id: {_id}, Label: {item['label']}, text: {text}")

    # create masked image for OTHER label
    image_masked = image.copy()
    for item in data["form"]:
        # format : x0,y0,x1,y1
        x0, y0, x1, y1 = item["box"]
        image_masked = cv2.rectangle(
            image_masked, (x0, y0), (x1, y1), (255, 255, 255), thickness=-1
        )

    if debug_fragments:
        file_path = os.path.join(f"{__tmp_path__}/snippet", f"{filename}-masked.png")
        cv2.imwrite(file_path, image_masked)

    # masked boxes will be same as the original ones
    boxes_masked, results_masked = extract_icr(image_masked, "other", boxp, icrp)

    print("-------- MASKED ----------")
    current_max_index = data["form"][-1]["id"]
    # Add masked boxes to the end of the list of annotations, with the same line number as the original box
    # some of the boxes may be empty, so we will filter them out
    if results_masked is not None and len(results_masked["words"]) > 0:
        for i, word in enumerate(results_masked["words"]):
            x, y, w, h = word["box"]
            line_number = find_line_number(line_bboxes, [x, y, w, h])
            word_box = [x, y, x + w, y + h]

            item = {
                "id": current_max_index + i,
                "text": word["text"],
                "box": word_box,
                "line_number": line_number,
                "linking": [],  # TODO: Not in use.
                "label": "other",
                "words": [{"text": word["text"], "box": word_box}],
            }

            data["form"].append(item)

    # Find all annotations by line number
    items_by_line = {}
    form_data = []
    for item in data["form"]:
        print(f"item = {item}")
        if "line_number" not in item:
            print(f"skipping item = {item}")
            continue
        if item["line_number"] not in items_by_line:
            items_by_line[item["line_number"]] = []
        items_by_line[item["line_number"]].append(item)
        form_data.append(item)

    data["form"] = form_data
    # Order by line number
    unique_line_numbers = list(items_by_line.keys())
    unique_line_numbers.sort()
    items_by_line = {
        line: np.array(items_by_line[line]) for line in unique_line_numbers
    }

    word_index = 0
    data_form_sorted = []
    # Order annotations by X value (left to right) per line
    for line_number, items_on_line in items_by_line.items():
        boxes_on_line = np.array([item["box"] for item in items_on_line])
        items_on_line = items_on_line[np.argsort(boxes_on_line[:, 0])]

        for item in items_on_line:
            item["word_index"] = word_index
            data_form_sorted.append(item)
            word_index += 1

    data["form"] = data_form_sorted

    json_path = os.path.join(output_ann_dir, filename + ".json")
    with open(json_path, "w") as json_file:
        json.dump(
            data,
            json_file,
            sort_keys=False,
            separators=(",", ": "),
            ensure_ascii=False,
            indent=2,
            cls=NumpyEncoder,
        )


def decorate_funsd(
    src_dir: str, overwrite: bool = False, debug_fragments: bool = False
) -> None:
    """'Decorate' FUNSD annotation files with ICR-ed contents from the source images."""
    src_dir = os.path.expanduser(src_dir)
    work_dir_boxes = ensure_exists(f"{__tmp_path__}/boxes")
    work_dir_icr = ensure_exists(f"{__tmp_path__}/icr")
    output_ann_dir = ensure_exists(os.path.join(src_dir, "annotations"))
    # debug_fragments = True
    if debug_fragments:
        ensure_exists(f"{__tmp_path__}/snippet")

    logger.info("⏳ Decorating examples from = %s", src_dir)
    ann_dir = os.path.join(src_dir, "annotations_tmp")
    img_dir = os.path.join(src_dir, "images")

    boxp = BoxProcessorUlimDit(
        work_dir=work_dir_boxes,
        # models_dir="./model_zoo/unilm/dit/text_detection",
        cuda=True,
    )

    icrp = TrOcrProcessor(work_dir=work_dir_icr, cuda=True)

    items = glob.glob(os.path.join(ann_dir, "*.json"))
    if len(items) == 0:
        raise Exception(f"No annotations to process in : {ann_dir}")

    for i, FUNSD_file_path in enumerate(items):
        print("*" * 60)
        filename = FUNSD_file_path.split("/")[-1]
        print(f"Processing annotation : {filename}")
        try:
            if os.path.isfile(os.path.join(output_ann_dir, filename)) and not overwrite:
                print(
                    f"File {filename} already decorated and Overwrite is disabled. Continuing to next file."
                )
                continue

            # if filename[:-5] != "179579324_2":
            #     print(f"Skipping {filename}")
            #     continue

            with open(FUNSD_file_path, "r", encoding="utf8") as f:
                data = json.load(f)
            if len(data["form"]) == 0:
                print(f"File: {filename}, has no annotations. Skipping decorate.")
                continue
            __decorate_funsd(
                data,
                filename[:-5],
                output_ann_dir,
                img_dir,
                boxp,
                icrp,
                debug_fragments,
            )
        except Exception as e:
            raise e
    logger.info("Decorate Done!")


def default_decorate(args: object):
    print("Default decorate")
    print(args)
    print("*" * 180)

    # This should be our dataset folder
    mode = args.mode
    src_dir = os.path.join(args.dir, f"{mode}")
    decorate_funsd(src_dir, debug_fragments=False)


def get_decorate_parser(subparsers=None) -> argparse.ArgumentParser:
    """
    Argument parser

    PYTHONPATH="$PWD" python decorate.py --mode test --dir ~/datasets/CORR/output/dataset/

    :param subparsers: If left None, a new ArgumentParser will be made. Otherwise pass the object generated from
                       argparse.ArgumentParser(...).add_subparsers(...) to add this as a subparser.
    :return: an ArgumentParser either independent or attached as a subparser
    """
    if subparsers is not None:
        decorate_parser = subparsers.add_parser(
            "decorate", help="Decorate documents(Box detection, ICR)"
        )
    else:
        decorate_parser = argparse.ArgumentParser(
            prog="decorate", description="Fills converted FUNSD with ICR data"
        )

    decorate_parser.set_defaults(func=default_decorate)

    decorate_parser.add_argument(
        "--mode",
        required=True,
        type=str,
        default="train",
        help="Conversion mode : train/test/validate/etc",
    )

    decorate_parser.add_argument(
        "--dir",
        required=True,
        type=str,
        help="Base dataset directory where the document for decorating resize",
    )

    return decorate_parser


if __name__ == "__main__":
    args = get_decorate_parser().parse_args()
    logger = logging.getLogger(__name__)
    print("-" * 120)
    print(args)
    print("-" * 120)
    args.func(args)
