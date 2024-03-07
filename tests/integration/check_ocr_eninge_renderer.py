import glob
import os
from typing import Dict

import cv2
import torch

from marie.boxes.box_processor import PSMode
from marie.ocr import CoordinateFormat, DefaultOcrEngine, OcrEngine
from marie.ocr.util import get_words_and_boxes
from marie.renderer import PdfRenderer
from marie.renderer.text_renderer import TextRenderer
from marie.timer import Timer
from marie.utils.docs import frames_from_file
from marie.utils.image_utils import crop_to_content
from marie.utils.json import load_json_file, store_json_object
from marie.utils.utils import ensure_exists


def process_dir(
        ocr_engine: OcrEngine,
        image_dir: str,
):
    import random
    items = glob.glob(os.path.join(image_dir, "*.*"))
    random.shuffle(items)

    for idx, img_path in enumerate(items):
        try:
            process_file(ocr_engine, img_path)
        except Exception as e:
            print(e)
            # raise e


@Timer(text="Process time {:.4f} seconds")
def process_file(ocr_engine: OcrEngine, img_path: str):
    try:
        print("Processing", img_path)
        img_path = os.path.expanduser(img_path)
        if not os.path.exists(img_path):
            raise Exception(f"File not found : {img_path}")

        key = img_path.split("/")[-1]
        frames = frames_from_file(img_path)

        results = ocr_engine.extract(frames, PSMode.SPARSE, CoordinateFormat.XYWH)

        print("Testing text renderer")
        json_path = os.path.join("/tmp/fragments", f"results-{key}.json")
        store_json_object(results, json_path)
        # results = load_json_file(os.path.join("/tmp/fragments", f"results-{key}.json"))

        # frames = [crop_to_content(frame, True) for frame in frames]
        extract_bouding_boxes(img_path, json_path, ngram=2)


        renderer = PdfRenderer(config={"preserve_interword_spaces": True})
        renderer.render(
            frames,
            results,
            output_filename=os.path.join(work_dir_icr, f"results-{key}.pdf"),
        )

        if False:
            renderer = TextRenderer(config={"preserve_interword_spaces": True})
            renderer.render(
                frames,
                results,
                output_filename=os.path.join(work_dir_icr, f"results-{key}.txt"),
            )
    except Exception as e:
        print("Error processing", img_path)
        raise e


def normalize_label(label: str):
    return label.replace("%", "_PERCENT_").replace("\"", "_QUOTE_").replace("'", "_SINGLE_QUOTE_").replace(" ",
                                                                                                           "_").replace(
        "(", "_OPEN_BRACKET_").replace(")", "_CLOSE_BRACKET_").replace("&", "_AND_").replace(".", "_DOT_").replace(",",
                                                                                                                   "_COMA_").replace(
        "/", "_SLASH_").replace("\\", "_SLASH_").replace(":", "_SEMI_").replace("*", "_STAR_").replace("?",
                                                                                                       "_QUESTION_").replace(
        "\"", "_SLASH_").replace("<", "_SIGN_LT_").replace(">", "_SIGN_GT_").replace("|", "_PIPE_")


def extract_bouding_boxes(img_path: str, metadata_path: str, ngram: int = 2):
    img_path = os.path.expanduser(img_path)
    if not os.path.exists(img_path):
        raise Exception(f"File not found : {img_path}")

    key = img_path.split("/")[-1]
    frames = frames_from_file(img_path)
    ocr_results = load_json_file(metadata_path)

    words = []
    boxes = []

    for page_idx in range(len(frames)):
        page_words, page_boxes = get_words_and_boxes(ocr_results, page_idx)
        words.append(page_words)
        boxes.append(page_boxes)

    from marie.utils.overlap import find_overlap_horizontal, merge_bboxes_as_block

    fname  = os.path.basename(img_path).split(".")[0]
    # create n-gram for each page and store it in a separate file
    k = 0
    for page_idx in range(len(frames)):
        page_words = words[page_idx]
        page_boxes = boxes[page_idx]

        for i in range(len(page_words) - ngram + 1):
            ngram_words = page_words[i:i + ngram]
            ngram_boxes = page_boxes[i:i + ngram]
            print(ngram_words, ngram_boxes)
            # create a new frame with the ngram
            # create a new ocr result with the ngram
            # store the new frame and the new ocr result in a new file

            # convert ngram_words to a key
            key = "_".join(ngram_words)
            key = normalize_label(key)
            print(key)
            # create ngram snippet
            ngram_frame = frames[page_idx]
            box = merge_bboxes_as_block(ngram_boxes)
            x, y, w, h = box
            snippet = ngram_frame[y: y + h, x: x + w:]
            # store the snippet
            k += 1
            ensure_exists(f"/tmp/fragments/converted/{key}")
            cv2.imwrite(f"/tmp/fragments/converted/{key}/{fname}_{k}.png", snippet)


if __name__ == "__main__":
    work_dir_boxes = ensure_exists("/tmp/boxes")
    work_dir_icr = ensure_exists("/tmp/icr")
    ensure_exists("/tmp/fragments")

    img_path = "~/tmp/4007/176073139.tif"
    img_path = "~/tmp/demo/159581778_1.png"
    img_path = os.path.expanduser("~/tmp/demo")

    use_cuda = torch.cuda.is_available()
    ocr_engine = DefaultOcrEngine(cuda=use_cuda)

    # check if we can process a single file or a directory
    if os.path.isdir(img_path):
        process_dir(ocr_engine, img_path)
    else:
        process_file(ocr_engine, img_path)
