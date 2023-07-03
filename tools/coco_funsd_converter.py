import argparse
import concurrent.futures
import distutils.util
import glob
import hashlib
import io
import json
import logging
from rich.logging import RichHandler
import multiprocessing as mp
import os
import random
import shutil
import string
import sys
import time
import uuid
from concurrent.futures.thread import ThreadPoolExecutor
from functools import lru_cache
from multiprocessing import Pool
# import rstr
import cv2
import numpy as np
from faker import Faker
from faker.providers import BaseProvider
from PIL import Image, ImageDraw, ImageFont

from marie.boxes import BoxProcessorUlimDit
from marie.boxes.box_processor import PSMode
from marie.boxes.craft_box_processor import BoxProcessorCraft
from marie.boxes.line_processor import find_line_number
from marie.document.trocr_icr_processor import TrOcrIcrProcessor
from marie.numpyencoder import NumpyEncoder
from marie.timer import Timer
from marie.utils.utils import ensure_exists

# FUNSD format can be found here
# https://guillaumejaume.github.io/FUNSD/description/

# Globals
logging.basicConfig(level=logging.DEBUG, handlers=[RichHandler(enable_link_path=True)])
logger = logging.getLogger(__name__)
_tmp_path = "/tmp/marie"

# setup data aug
fake = Faker("en_US")
fake_names_only = Faker(["it_IT", "en_US", "es_MX", "en_IN"])  # 'de_DE',

def from_json_file(filename):
    with io.open(filename, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)
        return data


def __scale_height(img, target_size, method=Image.LANCZOS):
    ow, oh = img.size
    scale = oh / target_size
    w = ow / scale
    h = target_size  # int(max(oh / scale, crop_size))
    resized = img.resize((int(w), int(h)), method)
    return resized, resized.size


def __validate_qa_balance(category_counts: dict, link_map: dict, id_to_name: dict) -> dict:
    """ Validate the question/answer balance given a set of categories with their total
        presence in a set of annotations.
        :param category_counts: a dict of {str label: number of occurrences}
        :param link_map: {str label: [category id, category id]} for all QA pairs.
                         {str label: [-1]} for all non-QA categories.
        :param id_to_name: {category id: str label}
    """
    category_imbalance = {}
    for category, count in category_counts.items():
        q_a_link = link_map[category]
        if len(q_a_link) <= 1:  # Not a question/answer pair
            continue
        question_id, answer_id = q_a_link[0], q_a_link[1]
        question, answer = id_to_name[question_id], id_to_name[answer_id]
        question_count, answer_count = category_counts[question], category_counts[answer]

        if question_count != answer_count:
            category_imbalance[(question, answer)] = (question_count, answer_count)

    return category_imbalance


def __convert_coco_to_funsd(
        src_dir: str,
        output_path: str,
        annotations_filename: str,
        config: object,
        strip_file_path: bool,
) -> None:
    """
    Convert CVAT annotated COCO dataset into FUNSD compatible format for finetuning models.
    """
    print("******* Conversion info ***********")
    print(f"src_dir     : {src_dir}")
    print(f"output_path : {output_path}")
    print(f"annotations : {annotations_filename}")
    print(f"strip_file_path : {strip_file_path}")

    # VALIDATE CONFIG Contents
    if "question_answer_map" not in config:
        raise Exception("Expected key missing from config : question_answer_map")
    if "id_map" not in config:
        raise Exception("Expected key missing from config : id_map")
    if "link_map" not in config:
        raise Exception("Expected key missing from config : link_map")

    link_map = config["link_map"]
    name_to_config_id = config["id_map"]
    config_id_to_name = {_id: name for name, _id in name_to_config_id.items()}

    coco = from_json_file(annotations_filename)

    # CONFIG <-> COCO Consistency
    unknown_categories = [category["name"] for category in coco["categories"] if
                          not (category["name"] in config["link_map"])]
    if len(unknown_categories) > 0:
        raise Exception(f"COCO file has categories not found in your config: {unknown_categories}")

    # Initialize Image based contexts
    annotations_by_image = {}
    images_by_id = {}
    for image in coco["images"]:
        annotations_by_image[image["id"]] = []
        images_by_id[image["id"]] = image["file_name"].split("/")[-1] if strip_file_path else image["file_name"]

    # Initialize Category based contexts
    cat_by_id = {}
    category_counts = {}
    duplicate_categories = []
    for category in coco["categories"]:
        # Identify duplicate categories mappings
        if category["name"] in category_counts:
            duplicate_categories.append(category["name"])
        category_counts[category["name"]] = 0
        cat_by_id[int(category["id"])] = category["name"]

    # VALIDATE that we don't have duplicate categories mappings
    if len(duplicate_categories) > 0:
        raise Exception(f"COCO file has duplicate categories: {duplicate_categories}")

    annotation_category_counts = category_counts.copy()
    for annotation in coco["annotations"]:
        # Group annotations by image_id as their key
        annotations_by_image[annotation["image_id"]].append(annotation)
        # Calculate Category counts over all annotations
        annotation_category_counts[cat_by_id[annotation["category_id"]]] += 1

    # VALIDATE question/answer balance in annotations
    category_imbalance = __validate_qa_balance(annotation_category_counts, link_map, config_id_to_name)

    if len(category_imbalance) > 0:
        print(f"WARNING: {len(category_imbalance)} Question/Answer pairs have an imbalanced number of annotations!")
        print("Proceeding to Identify individual offenders image offenders.")
        # Identify question/answer imbalance by file
        images_with_cat_imbalance = {}
        for image_id, annotations in annotations_by_image.items():
            image_category_counts = category_counts.copy()
            for annotation in annotations:
                image_category_counts[cat_by_id[annotation["category_id"]]] += 1
            imbalance = __validate_qa_balance(image_category_counts, link_map, config_id_to_name)
            if len(imbalance) > 0:
                images_with_cat_imbalance[image_id] = imbalance
        print(f"Number of Offenders: {len(images_with_cat_imbalance)}")
        # images_with_cat_imbalance = {images_by_id[_id]: details for _id, details in images_with_cat_imbalance.items()}
        # print(f"Offenders Details: {images_with_cat_imbalance}")

    # Cleanup Validation variables
    del annotation_category_counts
    del category_imbalance
    del category_counts

    # Start Conversion
    ensure_exists(os.path.join(output_path, "annotations_tmp"))
    ensure_exists(os.path.join(output_path, "images"))
    for image_id, image_annotations in annotations_by_image.items():

        form_dict = {"form": []}
        for i, annotation in enumerate(image_annotations):
            # Convert form XYWH -> xmin,ymin,xmax,ymax
            bbox = [int(x) for x in annotation["bbox"]]
            bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]

            category_name = cat_by_id[annotation["category_id"]]
            label = category_name

            form_dict["form"].append({
                "id": i,
                "text": "POPULATE_VIA_ICR",
                "box": bbox,
                "linking": [link_map[category_name]],  # TODO: Not in use. Will need refactor to use.
                "label": label,
                "words": [
                    {"text": "POPULATE_VIA_ICR_WORD", "box": [0, 0, 0, 0]},
                ],
            })

        filename = images_by_id[image_id].split("/")[-1].split(".")[0]
        src_img_path = os.path.join(src_dir, "images", f"{filename}.png")
        json_path = os.path.join(output_path, "annotations_tmp", f"{filename}.json")
        dst_img_path = os.path.join(output_path, "images", f"{filename}.png")
        # Save tmp state FUNSD JSON
        with open(json_path, "w") as json_file:
            json.dump(form_dict, json_file, indent=4)
        # Copy respective image with the above annotations
        shutil.copyfile(src_img_path, dst_img_path)


def convert_coco_to_funsd(
        src_dir: str, output_path: str, config: object, strip_file_name_path: bool
) -> None:
    """
    Convert CVAT annotated COCO 1.0 dataset into FUNSD compatible format for finetuning models.
    source: "FUNSD: A Dataset for Form Understanding in Noisy Scanned Documents" https://arxiv.org/pdf/1905.13538.pdf
    """
    # instances_default.json
    items = glob.glob(os.path.join(src_dir, "annotations/*.json"))
    if len(items) == 0:
        raise Exception(f"No annotations to process in : {src_dir}")

    os.makedirs(output_path, exist_ok=True)

    for idx, annotations_filename in enumerate(items):
        try:
            print(f"Processing annotation : {annotations_filename}")
            __convert_coco_to_funsd(
                src_dir, output_path, annotations_filename, config, strip_file_name_path
            )
        except Exception as e:
            raise e


# def image_to_byte_array(image: Image) -> bytes:
#     imgByteArr = io.BytesIO()
#     image.save(imgByteArr, format=image.format)
#     imgByteArr = imgByteArr.getvalue()
#     return imgByteArr


def extract_icr(image, label:str, boxp, icrp, debug_fragments: bool = False):
    """
    """
    if not isinstance(image, np.ndarray):
        raise Exception("Expected image in numpy format")

    msg_bytes = image.tobytes(order="C")
    m = hashlib.sha256()
    m.update(msg_bytes)
    checksum = m.hexdigest()

    ensure_exists(f"{_tmp_path}/icr")
    print(f"checksum = {checksum}")
    json_file = f"{_tmp_path}/icr/{checksum}/{checksum}.json"

    # Have we extracted this image before?
    if os.path.exists(json_file):
        print(f"From JSONFILE : {json_file}")
        json_data = from_json_file(json_file)
        return json_data["boxes"], json_data["result"]

    # TODO: Model needs to be trained to Extract sub-boxes from snippets
    # # Extract sub-boxes
    key = checksum
    # boxes, img_fragments, lines, _, line_bboxes = boxp.extract_bounding_boxes(
    #     key, "field", image, PSMode.SPARSE)
    # NOTE: For now we assume there are no internal boxes to be discovered
    boxes, img_fragments, lines = [], [], [1]

    # we found no boxes, so we will creat only one box and wrap a whole image as that
    if boxes is None or len(boxes) == 0:
        print(f"No internal boxes for : {checksum}")
        if debug_fragments:
            file_path = os.path.join(ensure_exists(f"{_tmp_path}/icr/{checksum}/"), f"{checksum}.png")
            cv2.imwrite(file_path, image)

        h = image.shape[0]
        w = image.shape[1]
        boxes = [[0, 0, w, h]]
        img_fragments = [image]
        lines = [1]

    result, overlay_image = icrp.recognize(
        key, label, image, boxes, img_fragments, lines)

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
        data: dict, filename: str, output_ann_dir: str, img_dir: str,
        boxp: BoxProcessorUlimDit, icrp: TrOcrIcrProcessor, debug_fragments: bool = False
) -> None:
    """ 'Decorate' a FUNSD file with ICR extracted text from the corresponding image
    """
    image_path = os.path.join(img_dir, filename+".png")
    image, size = load_image(image_path)

    print(f"Extracting line numbers with Box Processor for {filename}")
    # line_numbers : line number associated with bounding box
    # lines : raw line boxes that can be used for further processing
    _, _, line_numbers, _, line_bboxes = boxp.extract_bounding_boxes(
        filename, "lines", image, PSMode.MULTI_LINE
    )

    for item in data["form"]:
        # Boxes are in stored in x0,y0,x1,y1 where x0,y0 is upper left corner and x1,y1 if bottom/right
        x0, y0, x1, y1 = item["box"]
        _id = item["id"]

        snippet = image[y0:y1, x0:x1, :]
        line_number = find_line_number(line_bboxes, [x0, y0, x1 - x0, y1 - y0])

        # each snippet could be on multiple lines
        print(f"line_number = {line_number}")
        # export cropped region
        if debug_fragments:
            file_path = os.path.join(f"{_tmp_path}/snippet", f"{filename}-snippet_{_id}.png")
            cv2.imwrite(file_path, snippet)

        boxes, results = extract_icr(snippet, item["label"], boxp, icrp, debug_fragments)
        results.pop("meta", None)

        if (
            results is None
            or len(results) == 0
            or results["lines"] is None
            or len(results["lines"]) == 0
        ):
            print(f"*No results in {filename} for id:{_id}")
            continue

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
        image_masked = cv2.rectangle(image_masked, (x0, y0), (x1, y1), (255, 255, 255), thickness=-1)

    if debug_fragments:
        file_path = os.path.join(f"{_tmp_path}/snippet", f"{filename}-masked.png")
        cv2.imwrite(file_path, image_masked)

    # masked boxes will be same as the original ones
    boxes_masked, results_masked = extract_icr(image_masked, "other", boxp, icrp)

    print("-------- MASKED ----------")
    current_max_index = data["form"][-1]["id"]
    for i, word in enumerate(results_masked["words"]):
        x, y, w, h = word["box"]
        line_number = find_line_number(line_bboxes, [x, y, w, h])
        word_box = [x, y, x + w, y + h]

        item = {
            "id": current_max_index + i,
            "text": word["text"],
            "box": word_box,
            "line_number": line_number,
            "linking": [],              # TODO: Not in use.
            "label": "other",
            "words": [{"text": word["text"], "box": word_box}],
        }

        data["form"].append(item)

    # Find all annotations by line number
    items_by_line = {}
    for item in data["form"]:
        if item["line_number"] not in items_by_line:
            items_by_line[item["line_number"]] = []
        items_by_line[item["line_number"]].append(item)

    # Order by line number
    unique_line_numbers = list(items_by_line.keys())
    unique_line_numbers.sort()
    items_by_line = {line: np.array(items_by_line[line]) for line in unique_line_numbers}

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

    json_path = os.path.join(output_ann_dir, filename+".json")
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


def decorate_funsd(src_dir: str, overwrite: bool = False, debug_fragments: bool = False) -> None:
    """'Decorate' FUNSD annotation files with ICR-ed contents from the source images."""
    work_dir_boxes = ensure_exists(f"{_tmp_path}/boxes")
    work_dir_icr = ensure_exists(f"{_tmp_path}/icr")
    output_ann_dir = ensure_exists(os.path.join(src_dir, "annotations"))
    # debug_fragments = True
    if debug_fragments:
        ensure_exists(f"{_tmp_path}/snippet")

    logger.info("⏳ Decorating examples from = %s", src_dir)
    ann_dir = os.path.join(src_dir, "annotations_tmp")
    img_dir = os.path.join(src_dir, "images")

    boxp = BoxProcessorUlimDit(
        work_dir=work_dir_boxes,
        # models_dir="./model_zoo/unilm/dit/text_detection",
        cuda=True)

    icrp = TrOcrIcrProcessor(work_dir=work_dir_icr, cuda=True)

    items = glob.glob(os.path.join(ann_dir, "*.json"))
    if len(items) == 0:
        raise Exception(f"No annotations to process in : {ann_dir}")

    for i, FUNSD_file_path in enumerate(items):
        print("*" * 60)
        filename = FUNSD_file_path.split('/')[-1]
        print(f"Processing annotation : {filename}")
        try:
            if os.path.isfile(os.path.join(output_ann_dir, filename)) and not overwrite:
                print(f"File {filename} already decorated and Overwrite is disabled. Continuing to next file.")
                continue

            with open(FUNSD_file_path, "r", encoding="utf8") as f:
                data = json.load(f)
            if len(data["form"]) == 0:
                print(f"File: {filename}, has no annotations. Skipping decorate.")
                continue
            __decorate_funsd(data, filename[:-5], output_ann_dir, img_dir, boxp, icrp, debug_fragments)
        except Exception as e:
            raise e


def generate_pan(num_char):
    import string

    letters = string.ascii_letters.lower()
    if np.random.choice([0, 1], p=[0.5, 0.5]):
        letters = string.digits
    if np.random.choice([0, 1], p=[0.5, 0.5]):
        letters = letters.upper()
    prefix = "".join(random.choice(letters) for i in range(num_char))
    return prefix


@lru_cache(maxsize=20)
def get_cached_font(font_path, font_size):
    # return ImageFont.truetype(font_path, font_size, layout_engine=ImageFont.Layout.BASIC)
    return ImageFont.truetype(font_path, font_size)


def generate_date() -> str:
    # https://datatest.readthedocs.io/en/stable/how-to/date-time-str.html
    patterns = [
        "%Y%m%d",
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%d.%m.%Y",
        "%d %B %Y",
        "%b %d, %Y",
    ]
    return fake.date(pattern=random.choice(patterns))


def generate_money() -> str:
    label_text = fake.pricetag()
    if np.random.choice([0, 1], p=[0.5, 0.5]):
        label_text = label_text.replace("$", "")
    return label_text


def generate_name(original: str) -> str:
    split = original.split(" ")
    name = [fake_names_only.first_name()]
    if len(split) > 1:
        name += [fake_names_only.last_name() for _ in range(1, len(split))]
    return ' '.join(name)


def generate_address() -> str:
    return fake.address()


def generate_alpha_numeric(original: str, alpha: bool = True, numeric: bool = True) -> str:
    return fake.password(length=len(original), digits=numeric, upper_case=alpha,
                         special_chars=False,  lower_case=False)


def generate_text(original: str, mask_type: str) -> str:
    """Generate text for specific type of label"""
    if mask_type == "money":
        return generate_money()
    elif mask_type == "date":
        return generate_date()
    elif mask_type == "name":
        return generate_name(original)
    elif mask_type == "address":
        return generate_address()
    elif mask_type == "numeric":
        return generate_alpha_numeric(original, alpha=False)
    else:  # Alpha-Numeric
        return generate_alpha_numeric(original)


def generate_annotation_data(text: str, width: int, height: int, font_path: str, dec: int = 2):
    """ Generate FUNSD annotations from a given text, using a given font, in a given area.

    :param text: any string with standard '\n' newline boundaries (if applicable)
    :param width: max pixel width of area.
    :param height: max pixel height of area.
    :param font_path: Path to font used for size reference.
    :param dec: font point decrement rate

    :return: 'words' field annotation, line hieghts, and font size(pt)
    """

    lines = text.splitlines()
    line_count = len(lines)
    # TODO: PROPERLY CALCULATE HERE
    font_size = int((height / line_count) * 0.75)  # 72pt/96px = 0.75 point to pixel ratio
    font = get_cached_font(font_path, font_size)
    # Reference image area to calculate font size values
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    space_w, _ = draw.textsize(" ", font=font)

    # Calculate FUNSD format annotations
    index = 0
    text_height = 0
    word_annotations = []

    while index < len(lines):
        text_width, text_height = draw.textsize(lines[index], font=font)
        # Can this line be contained?
        if text_width > width:  # If not reduce the size of the font and start over
            font_size = font_size - dec
            font = get_cached_font(font_path, font_size)
            space_w, _ = draw.textsize(" ", font=font)
            word_annotations = []
            index = 0
            continue

        start_x = 0
        start_y = min(index * text_height, height)
        padding = space_w // 2
        words = lines[index].split(" ")
        for word in words:
            word_width, _ = draw.textsize(word, font=font)
            end_x = min(start_x + word_width + padding, width)
            box = [start_x, start_y, end_x, start_y+text_height]  # x0, y0, x1, y1
            word_annotations.append({"text": word, "box": box})
            start_x += word_width + space_w
        index += 1

    line_heights = [text_height for _ in lines]

    return word_annotations, line_heights, font_size


def load_image_pil(image_path):
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    return image, (w, h)


# @Timer(text="Aug in {:.4f} seconds")
def __augment_decorated_process(
        guid: int, count: int,
        file_path: str, image_path: str,
        dest_annotation_dir: str, dest_image_dir: str,
        mask_config: dict
) -> None:
    """ Generate a number of new FUNSD annotation files and images for a given FUNSD annotation file and
        corresponding image that augment and mask the fields given in the mask_config.

        :param count: number of augmentations
        :param file_path: path to source FUNSD annotation JSON file
        :param image_path: path to source image PNG file
        :param dest_annotation_dir: path to output location for augmented annotations files
        :param dest_image_dir: path to output location for augmented image files
        :param mask_config: A dictionary for how and what fields to augment
            {
                'prefixes':         ['prefix to labels', ... ],
                'fonts':            ['font file name', ... ],
                'masks_by_type':    {'type': ['annotation label(no prefix)', ... ], ... }
            }
    """

    Faker.seed(0)

    filename = file_path.split("/")[-1].split(".")[0]
    prefixes = mask_config['prefixes']
    fonts = mask_config['fonts']
    print(f"File: {file_path}")
    with open(file_path, "r", encoding="utf8") as f:
        data = json.load(f)

    # Subset of annotations we don't intend to mask
    data_constant = {"form": []}
    for i in range(len(data["form"])):
        item = data["form"][i]
        label = item["label"][2:] if prefixes is not None else item["label"]
        # Add mask type to annotation item
        for mask_type in mask_config['masks_by_type']:
            if label in mask_config['masks_by_type'][mask_type]:
                data["form"][i]['mask_type'] = mask_type
                break
        # Remove annotations we don't intend to mask from 'data'
        if 'mask_type' not in data["form"][i]:
            data_constant["form"].append(data["form"].pop(i))


    for k in range(count):
        print(f"Iter : {guid} , {k} of {count} ; {filename} ")
        font_face = np.random.choice(fonts)
        font_path = os.path.join("./assets/fonts", font_face)

        data_aug = {"form": []}

        image_masked, size = load_image_pil(image_path)
        draw = ImageDraw.Draw(image_masked)

        for item in data["form"]:
            # box format : x0,y0,x1,y1
            x0, y0, x1, y1 = np.array(item["box"]).astype(np.int32)
            w = x1 - x0
            h = y1 - y0
            xoffset = 5
            yoffset = 0
            aug_text = generate_text(item['text'], item['mask_type'])
            words_annotations, line_heights, font_size = generate_annotation_data(aug_text, w, h, font_path)

            # Generate text inside image
            # font_size, label_text, segments_lines, line_heights = generate_text(
            #     label, w, h, font_path
            # )

            assert len(line_heights) != 0
            font = get_cached_font(font_path, font_size)

            # x0, y0, x1, y1 = xy
            # Yellow with outline for debug
            # draw.rectangle(
            #     ((x0, y0), (x1, y1)), fill="#FFFFCC", outline="#FF0000", width=1
            # )

            # clear region
            draw.rectangle(((x0, y0), (x1, y1)), fill="#FFFFFF")

            dup_item = {
                'id': item['id'],
                "text": aug_text,
                "words": words_annotations,
                "line_number": item['line_number'],
                "word_index": item['word_index'],
                "linking": [],
            }
            # words = []
            # TODO: CALCULATE HEIGHT IN THE generate_annotation_data(...) FUNCTION
            # total_text_height = 0
            # for th in line_heights:
            #     total_text_height += th
            #
            # space = h - total_text_height
            # line_offset = 0
            # baseline_spacing = max(4, space // len(line_heights))
            #
            # for line_idx, segments in enumerate(segments_lines):
            #     for seg in segments:
            #         seg_text = seg["text"]
            #         sx0, sy0, sx1, sy1 = seg["box"]
            #         sw = sx1 - sx0
            #         sh = sy1 - sy0
            #         adj_box = [
            #             x0 + sx0,
            #             y0 + line_offset,
            #             x0 + sx0 + sw,
            #             y0 + sh + line_offset,
            #         ]
            #         word = {"text": seg_text, "box": adj_box}
            #         words.append(word)
            #         # debug box
            #         # draw.rectangle(
            #         #     ((adj_box[0], adj_box[1]), (adj_box[2], adj_box[3])),
            #         #     outline="#FF0000",
            #         #     width=1,
            #         # )
            #     line_offset += line_heights[line_idx] + baseline_spacing
            #
            # dup_item["words"] = words
            #
            line_offset = 0

            for line_idx, text_line in enumerate(aug_text.split("\n")):
                draw.text(
                    (x0 + xoffset, y0 + line_offset),
                    text=text_line,
                    fill="#000000",
                    font=font,
                    stroke_fill=1,
                )
                line_offset += line_heights[line_idx] + baseline_spacing
            data_copy["form"].append(dup_item)

        # Save items
        out_name_prefix = f"{filename}_{guid}_{k}"

        json_path = os.path.join(dest_annotation_dir, f"{out_name_prefix}.json")
        dst_img_path = os.path.join(dest_image_dir, f"{out_name_prefix}.png")

        print(f'Writing : {json_path}')
        with open(json_path, "w") as json_file:
            json.dump(
                data_copy,
                json_file,
                # sort_keys=True,
                separators=(",", ": "),
                ensure_ascii=False,
                indent=2,
                cls=NumpyEncoder,
            )

        # saving in JPG format as it is substantially faster than PNG
        # image_masked.save(
        #     os.path.join("/tmp/snippet", f"{out_name_prefix}.jpg"), quality=100
        # )  # 100 disables compression
        #
        # image_masked.save(os.path.join("/tmp/snippet", f"{out_name_prefix}.png"), compress_level=1)
        image_masked.save(dst_img_path, compress_level=2)

        del draw


def augment_decorated_annotation(count: int, src_dir: str, dest_dir: str):

    mask_config = {
        'prefixes': {'r.', 'd.', 's.', 'g.'},  # Implicit location of field on the image
        'fonts': [
            "FreeSansOblique.ttf",
            "FreeSans.ttf",
            "OpenSans-Light.ttf",
            "FreeMono.ttf",
            "vpscourt.ttf",
        ],
        'masks_by_type': {
            'address': ['address',],
            'money': [
                'allowed_amount_answer',
                'allowed_amount_total_answer',
                'billed_amount_answer',
                'billed_amount_total_answer',
                'check_amount_answer',
                'check_number_answer',
                'cob_answer',
                'cob_total_answer',
                'coinsurance_answer',
                'coinsurance_total_answer',
                'copay_answer',
                'copay_total_answer',
                'deductible_answer',
                'deductible_total_answer',
                'disallowed_answer',
                'disallowed_total_answer',
                'discount_answer',
                'discount_total_answer',
                'drg_amount_answer',
                'drg_amount_total_answer',
                "higher_allowable_answer",
                'ineligible_amount_member_answer',
                'interest_answer',
                'interest_total_answer',
                'medicare_allowed_answer',
                'medicare_paid_answer',
                'mem_liability_answer',
                'mem_liabilty_total_answer',
                'money',
                'money_answer',
                'other_adjustment_answer',
                'over_rnc_answer',
                'over_rnc_total_answer',
                'overpayments_recovery_answer',
                'paid_amount_answer',
                'paid_amount_total_answer',
                'partial_denial_answer',
                'patient_responsibility_answer',
                'patient_responsibility_total_answer',
                'plan_coverage_answer',
                'prepaid_answer',
                'prepaid_total_answer',
                'total',
                'withholding_answer',
                'withholding_total_answer',
                'writeoff_answer',
                'writeoff_total_answer',
            ],
            'name': [
                "member_name_answer",
                "patient_name_answer",
                "provider_answer",
            ],
            'date': [
                "check_date_answer",
                "begin_date_of_service_answer",  # NOTE: Short date
                "end_date_of_service_answer",
                "birthdate_answer",
                "date",                          # TODO: Verify this field is needed
                "letter_date",
            ],
            'numeric': [
                "claim_number_answer",
                "document_control_number",
                "member_number_answer",
                "patient_account_number_answer",
                "line_number_answer",
                "quantity_answer",
                "tooth_number_answer",
            ],
            'alpha-numeric': [
                "remark_code_answer",
                "code_answer",
                "code_modifier_answer",
                "procedure_code_answer",
                "remark_code_answer",
                "mem_liability_code_answer",
                "non-chargeable_amount_code_answer",
                "payment_code_answer",
                "procedure_code_answer",
                "procedure_code_modifier_answer",
                "remark_code_answer",
                "revenue_code_answer",
                "tooth_surface_answer",
            ],
        },
    }

    ann_dir = ensure_exists(os.path.join(src_dir, "annotations"))
    img_dir = ensure_exists(os.path.join(src_dir, "images"))
    dest_aug_annotations_dir = ensure_exists(os.path.join(dest_dir, "annotations"))
    dest_aug_images_dir = ensure_exists(os.path.join(dest_dir, "images"))

    aug_args = []
    for guid, file in enumerate(sorted(os.listdir(ann_dir))):
        file_path = os.path.join(ann_dir, file)
        img_path = os.path.join(img_dir, file.replace("json", "png"))
        __args = (guid, count, file_path, img_path, dest_aug_annotations_dir, dest_aug_images_dir, mask_config)
        aug_args.append(__args)

    start = time.time()
    print("\nPool Executor:")
    print("Time elapsed: %s" % (time.time() - start))

    pool = Pool(processes=int(mp.cpu_count() * 0.95))
    pool_results = pool.starmap(__augment_decorated_process, aug_args)

    pool.close()
    pool.join()

    print("Time elapsed[submitted]: %s" % (time.time() - start))
    for r in pool_results:
        print("Time elapsed[result]: %s  , %s" % (time.time() - start, r))
    print("Time elapsed[all]: %s" % (time.time() - start))


def visualize_funsd(src_dir: str, dst_dir: str, config: dict):
    ann_dir = os.path.join(src_dir, "annotations")
    img_dir = os.path.join(src_dir, "images")

    os.makedirs(dst_dir, exist_ok=True)

    for guid, file in enumerate(sorted(os.listdir(ann_dir))):
        file_path = os.path.join(ann_dir, file)

        print(f"file_path : {file_path}")
        with open(file_path, "r", encoding="utf8") as f:
            data = json.load(f)

        filename = file.split("/")[-1].split(".")[0]
        image_path = os.path.join(img_dir, file)
        image_path = image_path.replace("json", "png")
        print(f"image_path : {image_path}")
        print(f"filename : {filename}")
        image, size = load_image_pil(image_path)

        # draw predictions over the image
        draw = ImageDraw.Draw(image, "RGBA")
        font = ImageFont.load_default()
        # https://stackoverflow.com/questions/54165439/what-are-the-exact-color-names-available-in-pils-imagedraw
        label2color = config["label2color"]

        for i, item in enumerate(data["form"]):
            predicted_label = item["label"].lower()
            color = label2color[predicted_label]

            for word in item["words"]:
                box = word["box"]
                draw.rectangle(box, outline=color, width=1)

            box = item["box"]

            if predicted_label != "other":
                draw.rectangle(
                    box,
                    outline=color,
                    width=1,
                )  # fill=(0, 180, 0, 50)
            else:
                draw.rectangle(box, outline=color, width=1)

            predicted_label = f"{i} - {predicted_label}"
            draw.text(
                (box[0] + 10, box[1] - 10),
                text=predicted_label,
                fill=color,
                font=font,
                stroke_width=0,
            )

        image.save(os.path.join(dst_dir, f"viz_{filename}.png"))


@lru_cache(maxsize=10)
def resize_align_bbox(bbox, orig_w, orig_h, target_w, target_h):
    clip_to_y = 1000

    x_scale = target_w / orig_w
    y_scale = target_h / orig_h
    orig_left, orig_top, orig_right, orig_bottom = bbox
    x = int(np.round(orig_left * x_scale))
    y = int(np.round(orig_top * y_scale))
    xmax = int(np.round(orig_right * x_scale))
    ymax = int(np.round(orig_bottom * y_scale))
    return [x, y, xmax, min(ymax, clip_to_y)]


def rescale_annotation_frame(src_json_path: str, src_image_path: str):
    print(
        f"Recalling annotation : {src_json_path.split('/')[-1]}, {src_image_path.split('/')[-1]}"
    )

    filename = src_image_path.split("/")[-1].split(".")[0]
    image, orig_size = load_image_pil(src_image_path)
    resized, target_size = __scale_height(image, 1000)
    resized.save(ensure_exists(f"{_tmp_path}/snippet") + "/resized_{filename}.png")

    # print(f"orig_size, target_size   = {orig_size} : {target_size}")
    orig_w, orig_h = orig_size
    target_w, target_h = target_size
    data = from_json_file(src_json_path)

    try:
        for i, item in enumerate(data["form"]):
            bbox = tuple(item["box"])  # np.array(item["box"]).astype(np.int32)
            item["box"] = resize_align_bbox(bbox, orig_w, orig_h, target_w, target_h)
            for word in item["words"]:
                bbox = tuple(word["box"])  # np.array(item["box"]).astype(np.int32)
                word["box"] = resize_align_bbox(
                    bbox, orig_w, orig_h, target_w, target_h
                )
    except Exception as ex:
        print(src_json_path)
        print(ex)
        raise ex

    return data, resized


def __rescale_annotate_frames(
        ann_dir_dest, img_dir_dest, filename, json_path, image_path
):
    # 152630220_3  152618378_2 152618400  152624795_3
    print(f"filename : {filename}")
    print(f"json_path : {json_path}")
    print(f"image_path : {image_path}")

    data, image = rescale_annotation_frame(json_path, image_path)

    # Figure out how to handle this
    # if the width > 1000 SKIP for now
    if max(image.size) > 1000:
        pass
        # print(f"Skipping image due to size[{image.size}] : {filename}")
        # raise Exception(f"Skipping image due to size[{image.size}] : {filename}")
        # return

    json_path_dest = os.path.join(ann_dir_dest, f"{filename}.json")
    image_path_dest = os.path.join(img_dir_dest, f"{filename}.png")
    # save image and json data
    image.save(image_path_dest)

    with open(json_path_dest, "w") as json_file:
        json.dump(
            data,
            json_file,
            sort_keys=True,
            separators=(",", ": "),
            ensure_ascii=False,
            indent=2,
            cls=NumpyEncoder,
        )


def rescale_annotate_frames(src_dir: str, dest_dir: str):
    ann_dir = os.path.join(src_dir, "annotations")
    img_dir = os.path.join(src_dir, "images")

    ann_dir_dest = ensure_exists(os.path.join(dest_dir, "annotations"))
    img_dir_dest = ensure_exists(os.path.join(dest_dir, "images"))

    if False:
        for guid, file in enumerate(sorted(os.listdir(ann_dir))):
            json_path = os.path.join(ann_dir, file)
            filename = file.split("/")[-1].split(".")[0]
            image_path = os.path.join(img_dir, file).replace("json", "png")
            __rescale_annotate_frames(
                ann_dir_dest, img_dir_dest, filename, json_path, image_path
            )

    if True:
        args = []
        for guid, file in enumerate(sorted(os.listdir(ann_dir))):
            json_path = os.path.join(ann_dir, file)
            filename = file.split("/")[-1].split(".")[0]
            image_path = os.path.join(img_dir, file).replace("json", "png")
            __args = (ann_dir_dest, img_dir_dest, filename, json_path, image_path)
            args.append(__args)

        results = []
        start = time.time()
        print("\nPool Executor:")
        print("Time elapsed: %s" % (time.time() - start))

        pool = Pool(processes=mp.cpu_count())
        # pool = Pool(processes=1)
        pool_results = pool.starmap(__rescale_annotate_frames, args)

        pool.close()
        pool.join()

        print("Time elapsed[submitted]: %s" % (time.time() - start))
        for r in pool_results:
            print("Time elapsed[result]: %s  , %s" % (time.time() - start, r))
        print("Time elapsed[all]: %s" % (time.time() - start))


def split_dataset(src_dir, output_path, split_percentage):
    """
    Split CODO dataset for training and test.

    ARGS:
        src_dir  : input folder for COCO dataset, expected structure to have 'annotations' and 'image' folder
        output_path : output folder where new train/test directory will be created
        split_percentage      : split ration ex: .8  this will create 80/20 Train/Test split
    """

    print("Split dataset")
    print("src_dir     = {}".format(src_dir))
    print("output_path = {}".format(output_path))
    print("split       = {}".format(split_percentage))

    ann_dir = os.path.join(src_dir, "annotations")
    img_dir = os.path.join(src_dir, "images")

    if not os.path.exists(ann_dir):
        raise Exception("Source directory missing expected 'annotations' sub-directory")

    if not os.path.exists(img_dir):
        raise Exception("Source directory missing expected 'images' sub-directory")

    file_set = []
    for guid, file in enumerate(sorted(os.listdir(ann_dir))):
        json_path = os.path.join(ann_dir, file)
        image_path = os.path.join(img_dir, file).replace("json", "png")
        filename = image_path.split("/")[-1].split(".")[0]
        item = {"annotation": json_path, "image": image_path, "filename": filename}
        file_set.append(item)

    total_count = len(file_set)
    sample_count = int(total_count * split_percentage)
    print(f"split_percentage = {split_percentage}")
    print(f"total_count      = {total_count}")
    print(f"sample_count     = {sample_count}")
    print(f"output_path      = {output_path}")

    ann_dir_out_train = os.path.join(output_path, "train", "annotations")
    img_dir_out_train = os.path.join(output_path, "train", "images")

    ann_dir_out_test = os.path.join(output_path, "test", "annotations")
    img_dir_out_test = os.path.join(output_path, "test", "images")

    if os.path.exists(os.path.join(output_path, "train")) or os.path.exists(
            os.path.join(output_path, "test")
    ):
        raise Exception(
            "Output directory not empty, manually remove test/train directories."
        )

    os.makedirs(ann_dir_out_train, exist_ok=True)
    os.makedirs(img_dir_out_train, exist_ok=True)
    os.makedirs(ann_dir_out_test, exist_ok=True)
    os.makedirs(img_dir_out_test, exist_ok=True)

    np.random.shuffle(file_set)
    train_set = file_set[0:sample_count]
    test_set = file_set[sample_count:-1]

    print(f"Train size : {len(train_set)}")
    print(f"Test size : {len(test_set)}")

    splits = [
        {
            "name": "train",
            "files": train_set,
            "ann_dir_out": ann_dir_out_train,
            "img_dir_out": img_dir_out_train,
        },
        {
            "name": "test",
            "files": test_set,
            "ann_dir_out": ann_dir_out_test,
            "img_dir_out": img_dir_out_test,
        },
    ]

    for split in splits:
        fileset = split["files"]
        ann_dir_out = split["ann_dir_out"]
        img_dir_out = split["img_dir_out"]

        for item in fileset:
            ann = item["annotation"]
            img = item["image"]
            filename = item["filename"]
            shutil.copyfile(ann, os.path.join(ann_dir_out, f"{filename}.json"))
            shutil.copyfile(img, os.path.join(img_dir_out, f"{filename}.png"))


def default_decorate(args: object):
    print("Default decorate")
    print(args)
    print("*" * 180)

    # This should be our dataset folder
    mode = args.mode
    src_dir = os.path.join(args.dir, f"{mode}")
    decorate_funsd(src_dir, debug_fragments=False)


def default_augment(args: object):
    print("Default augment")
    print(args)
    print("*" * 180)

    # This should be our dataset folder
    mode = args.mode
    aug_count = args.count
    root_dir = args.dir
    src_dir = os.path.join(args.dir, f"{mode}")
    dst_dir = (
        args.dir_output
        if args.dir_output != "./augmented"
        else os.path.abspath(os.path.join(root_dir, f"{mode}-augmented"))
    )

    print(f"mode      = {mode}")
    print(f"aug_count = {aug_count}")
    print(f"src_dir   = {src_dir}")
    print(f"dst_dir   = {dst_dir}")

    augment_decorated_annotation(count=aug_count, src_dir=src_dir, dest_dir=dst_dir)


def default_rescale(args: object):
    print("Default rescale")
    print(args)
    print("*" * 180)

    # This should be our dataset folder
    mode = args.mode
    suffix = args.suffix
    root_dir = args.dir
    src_dir = os.path.join(args.dir, f"{mode}{suffix}")

    dst_dir = (
        args.dir_output
        if args.dir_output != "./rescaled"
        else os.path.abspath(os.path.join(root_dir, f"{mode}-rescaled"))
    )

    print(f"mode    = {mode}")
    print(f"suffix  = {suffix}")
    print(f"src_dir = {src_dir}")
    print(f"dst_dir = {dst_dir}")

    rescale_annotate_frames(src_dir=src_dir, dest_dir=dst_dir)


def default_visualize(args: object):
    print("Default visualize")
    print(args)
    print("*" * 180)

    src_dir = args.dir
    dst_dir = args.dir_output

    print(f"src_dir   = {src_dir}")
    print(f"dst_dir   = {dst_dir}")
    print(f"config    = {args.config}")
    # load config file
    config = from_json_file(args.config)

    visualize_funsd(src_dir, dst_dir, config)


def default_convert(args: object):
    print("Default convert")
    print(args)
    print("*" * 180)
    mode = args.mode
    suffix = args.mode_suffix
    strip_file_name_path = args.strip_file_name_path
    src_dir = os.path.join(args.dir, f"{mode}{suffix}")

    dst_path = (
        args.dir_converted
        if args.dir_converted != "./converted"
        else os.path.join(args.dir, "output", "dataset", f"{mode}")
    )

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"File not found : {args.config}")

    # load config file
    config = from_json_file(args.config)

    print(f"mode       = {mode}")
    print(f"suffix     = {suffix}")
    print(f"src_dir    = {src_dir}")
    print(f"dst_path   = {dst_path}")

    convert_coco_to_funsd(src_dir, dst_path, config, strip_file_name_path)


def default_all_steps(args: object):
    from argparse import Namespace

    print("Default all_steps")
    print(args)
    print("*" * 180)

    root_dir = args.dir
    aug_count = args.aug_count
    dataset_dir = os.path.join(root_dir, "output", "dataset")

    # clone and remove  unused values
    args_1 = vars(args).copy()
    args_1["func"] = None
    args_1["command"] = "convert"
    args_1["dir_converted"] = "./converted"

    args_2 = vars(args).copy()
    args_2["func"] = None
    args_2["command"] = "decorate"
    args_2["dir"] = dataset_dir

    args_3 = vars(args).copy()
    args_3["func"] = None
    args_3["command"] = "augment"
    args_3["dir"] = dataset_dir
    args_3["count"] = aug_count
    args_3["dir_output"] = "./augmented"

    args_4 = vars(args).copy()
    args_4["func"] = None
    args_4["command"] = "rescale"
    args_4["dir"] = dataset_dir
    args_4["dir_output"] = "./rescaled"
    args_4["suffix"] = "-augmented"

    # execute each step
    # default_convert(Namespace(**args_1))
    default_decorate(Namespace(**args_2))
    # default_augment(Namespace(**args_3))
    # default_rescale(Namespace(**args_4))


def default_split(args: object):
    print("Default split")
    print(args)
    print("*" * 180)

    src_dir = args.dir
    dst_dir = args.dir_output
    ratio = args.ratio

    print(f"src_dir = {src_dir}")
    print(f"dst_dir = {dst_dir}")
    print(f"ratio   = {ratio}")

    split_dataset(src_dir, dst_dir, ratio)


def extract_args(args=None) -> object:
    """
    Argument parser

    PYTHONPATH="$PWD" python ./marie/coco_funsd_converter.py --mode test --step augment --strip_file_name_path true --dir ~/dataset/private/corr-indexer --config ~/dataset/private/corr-indexer/config.json --aug-count 2
    """
    parser = argparse.ArgumentParser(
        prog="coco_funsd_converter", description="COCO to FUNSD conversion utility"
    )

    subparsers = parser.add_subparsers(
        dest='command', help='Commands to run', required=True
    )

    convert_parser = subparsers.add_parser(
        "convert", help="Convert documents from COCO to FUNSD-Like intermediate format"
    )

    convert_parser.set_defaults(func=default_convert)

    convert_parser.add_argument(
        "--mode",
        required=True,
        type=str,
        default="train",
        help="Conversion mode : train/test/validate/etc",
    )

    convert_parser.add_argument(
        "--mode-suffix",
        required=False,
        type=str,
        default="-deck-raw",
        help="Suffix for the mode",
    )

    convert_parser.add_argument(
        "--strip_file_name_path",
        required=True,
        # type=bool,
        # action='store_true',
        type=lambda x: bool(distutils.util.strtobool(x)),
        default=False,
        help="Should full image paths be striped from annotations file",
    )

    convert_parser.add_argument(
        "--dir",
        required=True,
        type=str,
        default="~/dataset/ds-001/indexer",
        help="Base data directory",
    )

    convert_parser.add_argument(
        "--dir-converted",
        required=False,
        type=str,
        default="./converted",
        help="Converted data directory",
    )

    convert_parser.add_argument(
        "--config",
        required=True,
        type=str,
        default='./config.json',
        help="Configuration file used for conversion",
    )

    decorate_parser = subparsers.add_parser(
        "decorate", help="Decorate documents(Box detection, ICR)"
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

    rescale_parser = subparsers.add_parser(
        "rescale", help="Rescale/Normalize documents to be used by UNILM"
    )
    rescale_parser.set_defaults(func=default_rescale)

    rescale_parser.add_argument(
        "--mode",
        required=True,
        type=str,
        help="Conversion mode : train/test/validate/etc",
    )

    rescale_parser.add_argument(
        "--dir",
        required=True,
        type=str,
        help="Source directory",
    )

    rescale_parser.add_argument(
        "--dir-output",
        default="./rescaled",
        type=str,
        help="Destination directory",
    )

    rescale_parser.add_argument(
        "--suffix",
        default="-augmented",
        type=str,
        help="Suffix to append to the source directory",
    )

    visualize_parser = subparsers.add_parser("visualize", help="Visualize documents")
    visualize_parser.set_defaults(func=default_visualize)

    visualize_parser.add_argument(
        "--dir",
        required=True,
        type=str,
        help="Source directory",
    )

    visualize_parser.add_argument(
        "--dir-output",
        default=f"{_tmp_path}/visualize",
        type=str,
        help="Destination directory",
    )

    visualize_parser.add_argument(
        "--config",
        type=str,
        default='./visualize-config.json',
        help="Configuration file used for conversion",
    )

    split_parser = subparsers.add_parser(
        "split", help="Split COCO dataset into train/test"
    )
    split_parser.set_defaults(func=default_split)

    split_parser.add_argument(
        "--dir",
        required=True,
        type=str,
        help="Source directory",
    )

    split_parser.add_argument(
        "--dir-output",
        default=f"{_tmp_path}/split",
        type=str,
        help="Destination directory",
    )

    split_parser.add_argument(
        "--ratio",
        default=0.8,
        type=float,
        help="Destination directory",
    )

    convert_all_parser = subparsers.add_parser(
        "convert-all",
        help="Run all conversion phases[convert,decorate,augment,rescale] using most defaults.",
    )

    convert_all_parser.set_defaults(func=default_all_steps)

    convert_all_parser.add_argument(
        "--mode",
        required=True,
        type=str,
        default="train",
        help="Conversion mode : train/test/validate/etc",
    )

    convert_all_parser.add_argument(
        "--mode-suffix",
        required=False,
        type=str,
        default="-deck-raw",
        help="Suffix for the mode",
    )

    convert_all_parser.add_argument(
        "--strip_file_name_path",
        required=True,
        # type=bool,
        # action='store_true',
        type=lambda x: bool(distutils.util.strtobool(x)),
        default=False,
        help="Should full image paths be striped from annotations file",
    )

    convert_all_parser.add_argument(
        "--aug-count",
        required=True,
        type=int,
        help="Number of augmentations per annotation",
    )

    convert_all_parser.add_argument(
        "--dir",
        required=True,
        type=str,
        default="~/dataset/ds-001/indexer",
        help="Base data directory",
    )

    convert_all_parser.add_argument(
        "--config",
        required=True,
        type=str,
        default='./config.json',
        help="Configuration file used for conversion",
    )

    try:
        return parser.parse_args(args) if args else parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    args = extract_args()
    print("-" * 120)
    print(args)
    print("-" * 120)
    # logger.setLevel(logging.DEBUG)
    args.func(args)
