import copy
import io
import json
import logging
import os
import random
import shutil
import uuid
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from marie.boxes.box_processor import PSMode
from marie.boxes.craft_box_processor import BoxProcessorCraft
from marie.document.trocr_icr_processor import TrOcrIcrProcessor
from marie.numpyencoder import NumpyEncoder
from marie.timer import Timer
from marie.utils.utils import ensure_exists

from faker.providers import BaseProvider
from faker import Faker
import rstr

import multiprocessing as mp
from concurrent.futures.thread import ThreadPoolExecutor

import concurrent.futures
import time
from multiprocessing import Pool


# FUNSD format can be found here
# https://guillaumejaume.github.io/FUNSD/description/

logger = logging.getLogger(__name__)

# setup data aug

fake = Faker()
# create new provider class


class MemberProvider(BaseProvider):
    def __init__(self, regex):
        self.possible_regexes = [
            "^C\d{5}$",
            "^0\d{7}$",
            "^5\d{9}$",
            "^C[0-9]*",
            "^H[0-9]*",
            "\d{6}-\d+",
            "^(\d{7})$",
            "^A\d{14}$",
            "^[0-9]{9}$",
            "^\d{10}WF$",
            "^1N[0-9]+$",
            "^HRP\d{8}$",
            "^(0{0}|0{3})1\d{8}$",
            "^(33)[0-9A-Z]{2,4}$",
            "^P1.+$|^PZ[0-9]{6}$",
            "^C\d{1}[A-Z0-9]{6}[A-Z]-{2}$",
            "^[0-9]{6}\\-C[0-9]{6}$",
            "^(?!W)[A-Z]{3}[0-9]{5}$",
            "^[0-9]{5,8}-[0-9]{5,6}P",
            "(^100\d{1}\.\d{4}$)|(^\d{4}\/\d{4,6}-\d{4,6}$)|(^\d{5}-\d{4,6}$)|(^1\d{7}$)|(^0\d{7,13}$)|^[0-9]{4,6}[-/\\][0-9]{4,6}$",
            "^(\d{10}(YN))$|^\d{6}[A-Z]{1}\d{3}(YN)$",
        ]

    def member_id(self) -> str:
        # print(rstr.xeger(r'[A-Z]\d[A-Z] \d[A-Z]\d'))
        sel_reg = random.choice(self.possible_regexes)
        return rstr.xeger(sel_reg)


fake.add_provider(MemberProvider)


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


def __scale_heightXXXX(img, target_size=1000, method=Image.LANCZOS):
    ow, oh = img.size
    old_size = (ow, oh)

    # paste the image if the width or height is smaller than the requested target size
    if max((ow, oh)) < target_size:
        new_im = Image.new("RGB", (min(ow, target_size), target_size), color=(255, 255, 255))
        new_im.paste(img)
        return new_im, new_im.size

    ratio = float(target_size) / max((ow, oh))
    new_size = tuple([int(x * ratio) for x in old_size])
    resized = img.resize(new_size, method)

    # if resized height is less than target then we pad it
    rw, rh = resized.size
    if rh < target_size:
        new_im = Image.new("RGB", (min(rw, target_size), target_size), color=(255, 255, 255))
        new_im.paste(resized)
        return new_im, new_im.size

    return resized, resized.size


def __scale_heightZZZ(img, target_size=1000, method=Image.LANCZOS):
    ow, oh = img.size
    old_size = (ow, oh)

    # paste the image if the width or height is smaller than the requested target size
    if max((ow, oh)) < target_size:
        new_im = Image.new("RGB", (min(ow, target_size), target_size), color=(255, 255, 255))
        new_im.paste(img)
        return new_im, new_im.size

    ratio = float(target_size) / max((ow, oh))
    new_size = tuple([int(x * ratio) for x in old_size])
    resized = img.resize(new_size, method)

    # if resized height is less than target then we pad it
    rw, rh = resized.size
    if rh < target_size:
        new_im = Image.new("RGB", (min(rw, target_size), target_size), color=(255, 255, 255))
        new_im.paste(resized)
        return new_im, new_im.size

    return resized, resized.size


def load_image(image_path):
    image = cv2.imread(image_path)
    h, w = image.shape[0], image.shape[1]
    return image, (w, h)


def convert_coco_to_funsd(src_dir: str, output_path: str) -> None:
    """
    Convert CVAT annotated COCO dataset into FUNSD compatible format for finetuning models.
    """
    src_file = os.path.join(src_dir, "annotations/instances_default.json")
    print(f"src_dir : {src_dir}")
    print(f"output_path : {output_path}")
    print(f"src_file : {src_file}")

    data = from_json_file(src_file)
    categories = data["categories"]
    images = data["images"]
    annotations = data["annotations"]

    images_by_id = {}
    for img in images:
        images_by_id[int(img["id"])] = img

    print(categories)
    print(annotations)

    cat_id_name = {}
    cat_name_id = {}

    # Categories / Answers should be generalized
    # Expected group mapping that will get translated into specific linking
    question_answer_map = {
        "member_name": "member_name_answer",
        "member_number": "member_number_answer",
        "pan": "pan_answer",
        "dos": "dos_answer",
        "patient_name": "patient_name_answer",
    }

    id_map = {
        "member_name": 0,
        "member_name_answer": 1,
        "member_number": 2,
        "member_number_answer": 3,
        "pan": 4,
        "pan_answer": 5,
        "dos": 6,
        "dos_answer": 7,
        "patient_name": 8,
        "patient_name_answer": 9,
    }

    link_map = {
        "member_name": [id_map["member_name"], id_map["member_name_answer"]],
        "member_name_answer": [id_map["member_name"], id_map["member_name_answer"]],
        "member_number": [id_map["member_number"], id_map["member_number_answer"]],
        "member_number_answer": [id_map["member_number"], id_map["member_number_answer"]],
        "pan": [id_map["pan"], id_map["pan_answer"]],
        "pan_answer": [id_map["pan"], id_map["pan_answer"]],
        "dos": [id_map["dos"], id_map["dos_answer"]],
        "dos_answer": [id_map["dos"], id_map["dos_answer"]],
        "patient_name": [id_map["patient_name"], id_map["patient_name_answer"]],
        "patient_name_answer": [id_map["patient_name"], id_map["patient_name_answer"]],
    }

    for category in categories:
        cat_id_name[category["id"]] = category["name"]
        cat_name_id[category["name"]] = category["id"]

    # Generate NER Tags
    ner_tags = []
    for question, answer in question_answer_map.items():
        ner_tags.append("B-" + question.upper())
        ner_tags.append("I-" + question.upper())

        ner_tags.append("B-" + answer.upper())
        ner_tags.append("I-" + answer.upper())

    print("Converted ner_tags =>")
    print(ner_tags)

    ano_groups = {}
    # Group annotations by image_id as their key
    for ano in annotations:
        if ano["image_id"] not in ano_groups:
            ano_groups[ano["image_id"]] = []
        ano_groups[ano["image_id"]].append(ano)

    errors = []

    for group_id in ano_groups:
        grouping = ano_groups[group_id]
        # Validate that each annotation has associated question/answer pair
        found_cat_id = []
        img_data = images_by_id[group_id]
        file_name = img_data["file_name"]
        filename = file_name.split("/")[-1].split(".")[0]

        category_counts = {}
        for ano in grouping:
            found_cat_id.append(ano["category_id"])
            # validate that we don't have duplicate question/answer mappings, we might change this down the road
            cat_name = cat_id_name[ano["category_id"]]
            category_counts[cat_name] = 1 if cat_name not in category_counts else category_counts[cat_name] + 1
            count = category_counts[cat_name]
            if False and count > 1:
                msg = f"Duplicate pair found for image_id[{group_id}] : {cat_name}, {count}, {filename}]"
                print(msg)
                errors.append(msg)

        # if we have any missing mapping we will abort and fix the labeling data before continuing
        for question, answer in question_answer_map.items():
            qid = cat_name_id[question]
            aid = cat_name_id[answer]
            # we only have question but no answer
            if qid in found_cat_id and aid not in found_cat_id:
                msg = f"Pair notfound for image_id[{group_id}] : {question} [{qid}] MISSING -> {answer} [{aid}]"
                print(msg)
                errors.append(msg)
            else:
                print(f"Pair found : {question} [{qid}] -> {answer} [{aid}]")

        if len(errors) > 0:
            payload = "\n".join(errors)
            raise Exception(f"Missing mapping \n {payload}")

        # start conversion
        img_data = images_by_id[group_id]
        file_name = img_data["file_name"]
        filename = file_name.split("/")[-1].split(".")[0]
        src_img_path = os.path.join(src_dir, "images", file_name)
        src_img, size = load_image(src_img_path)

        print(f"Image size : {size}")
        form_dict = {"form": []}

        for ano in grouping:
            category_id = ano["category_id"]
            # Convert form XYWH -> xmin,ymin,xmax,ymax
            bbox = [int(x) for x in ano["bbox"]]
            bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            category_name = cat_id_name[category_id]

            print(f"category_name => {category_name}")
            label = category_name

            item = {
                "id": id_map[category_name],
                "text": "POPULATE_VIA_ICR",
                "box": bbox,
                "linking": [link_map[category_name]],
                "label": label,
                "words": [
                    {"text": "POPULATE_VIA_ICR", "box": [0, 0, 0, 0]},
                ],
            }

            form_dict["form"].append(item)

        os.makedirs(os.path.join(output_path, "annotations_tmp"), exist_ok=True)
        os.makedirs(os.path.join(output_path, "images"), exist_ok=True)

        json_path = os.path.join(output_path, "annotations_tmp", f"{filename}.json")
        dst_img_path = os.path.join(output_path, "images", f"{filename}.png")

        if True:
            with open(json_path, "w") as json_file:
                json.dump(form_dict, json_file, indent=4)

            # copy and resize to 1000 H
            shutil.copyfile(src_img_path, dst_img_path)
            print(form_dict)


def normalize_bbox(bbox, size):
    return [
        int(1000 * bbox[0] / size[0]),
        int(1000 * bbox[1] / size[1]),
        int(1000 * bbox[2] / size[0]),
        int(1000 * bbox[3] / size[1]),
    ]


def extract_icr(image, boxp, icrp):
    key = "coco"
    boxes, img_fragments, lines, _ = boxp.extract_bounding_boxes(key, "field", image, PSMode.SPARSE)
    if boxes is None or len(boxes) == 0:
        print("Empty boxes")
        return [], []
    result, overlay_image = icrp.recognize(key, "test", image, boxes, img_fragments, lines)

    return boxes, result


def decorate_funsd(src_dir: str):
    work_dir_boxes = ensure_exists("/tmp/boxes")
    work_dir_icr = ensure_exists("/tmp/icr")
    output_ann_dir = ensure_exists(os.path.join(src_dir, "annotations"))

    logger.info("⏳ Decorating examples from = %s", src_dir)
    ann_dir = os.path.join(src_dir, "annotations_tmp")
    img_dir = os.path.join(src_dir, "images")

    boxp = BoxProcessorCraft(work_dir=work_dir_boxes, models_dir="./model_zoo/craft", cuda=True)
    icrp = TrOcrIcrProcessor(work_dir=work_dir_icr, cuda=True)

    for guid, file in enumerate(sorted(os.listdir(ann_dir))):
        print(f"guid = {guid}")
        if guid == 1:
            break

        file_path = os.path.join(ann_dir, file)
        with open(file_path, "r", encoding="utf8") as f:
            data = json.load(f)

        image_path = os.path.join(img_dir, file)
        image_path = image_path.replace("json", "png")
        image, size = load_image(image_path)

        for i, item in enumerate(data["form"]):
            # format : x0,y0,x1,y1
            box = np.array(item["box"]).astype(np.int32)
            x0, y0, x1, y1 = box
            snippet = image[y0:y1, x0:x1, :]

            # export cropped region
            if True:
                file_path = os.path.join("/tmp/snippet", f"{guid}-snippet_{i}.png")
                cv2.imwrite(file_path, snippet)

            boxes, results = extract_icr(snippet, boxp, icrp)

            print(boxes)
            print(results)

            if results is None or len(results) == 0 or results["lines"] is None or len(results["lines"]) == 0:
                print(f"No results for : {guid}-{i}")
                continue

            if True:
                file_path = os.path.join("/tmp/snippet", f"{guid}-snippet_{i}.png")
                cv2.imwrite(file_path, snippet)

            continue

            words = []
            text = ""
            try:
                text = " ".join([line["text"] for line in results["lines"]])
            except Exception as ex:
                # raise ex
                print(ex)
                # pass

            # boxes are in stored in x0,y0,x1,y1 where x0,y0 is upper left corner and x1,y1 if bottom/right
            # we need to account for offset from the snippet box
            # results["word"] are in a xywh format in local position and need to be converted to relative position
            print("-------------------------------")
            print(results["words"])
            for word in results["words"]:
                w_text = word["text"]
                x, y, w, h = word["box"]
                w_box = [x0 + x, y0 + y, x0 + x + w, y0 + y + h]
                adj_word = {"text": w_text, "box": w_box}
                words.append(adj_word)

            item["words"] = words
            item["text"] = text

        # create masked image for OTHER label
        image_masked, _ = load_image(image_path)
        index = 0
        for i, item in enumerate(data["form"]):
            # format : x0,y0,x1,y1
            box = np.array(item["box"]).astype(np.int32)
            x0, y0, x1, y1 = box
            cv2.rectangle(image_masked, (x0, y0), (x1, y1), (255, 255, 255), thickness=-1)
            index = i + 1

        if False:
            file_path = os.path.join("/tmp/snippet", f"{guid}-masked.png")
            cv2.imwrite(file_path, image_masked)

        boxes_masked, results_masked = extract_icr(image_masked, boxp, icrp)

        x0 = 0
        y0 = 0

        for i, word in enumerate(results_masked["words"]):
            w_text = word["text"]
            x, y, w, h = word["box"]
            w_box = [x0 + x, y0 + y, x0 + x + w, y0 + y + h]
            adj_word = {"text": w_text, "box": w_box}
            item = {
                "id": index + i,
                "text": w_text,
                "box": w_box,
                "linking": [],
                "label": "other",
                "words": [adj_word],
            }
            data["form"].append(item)

        json_path = os.path.join(output_ann_dir, file)
        with open(json_path, "w") as json_file:
            json.dump(
                data,
                json_file,
                sort_keys=True,
                separators=(",", ": "),
                ensure_ascii=False,
                indent=2,
                cls=NumpyEncoder,
            )


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
    # return ImageFont.truetype("/home/gbugaj/dev/marie-ai/assets/fonts/FreeMono.ttf", font_size, layout_engine=ImageFont.Layout.BASIC)
    # return ImageFont.truetype("/home/gbugaj/dev/marie-ai/assets/fonts/FreeMonoBold.ttf", font_size)
    return ImageFont.truetype(font_path, font_size, layout_engine=ImageFont.Layout.BASIC)


def generate_text(label, width, height, fontPath):
    # if label != "member_name_answer":
    #     return "", 0

    avg_line_height = 45
    est_line_count = height // avg_line_height

    height = min(height, 60)
    # Generate text inside image
    font_size = int(height * 1)

    # print(f":: {font_size} : {fontPath}")
    # font = ImageFont.truetype(fontPath, font_size)
    font = get_cached_font(fontPath, font_size)
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    space_w, _ = draw.textsize(" ", font=font)

    dec = 2
    index = 0
    label_text = ""

    while True:
        if index > 5:
            font_size = font_size - dec
            # font = ImageFont.truetype(fontPath, font_size)
            font = get_cached_font(fontPath, font_size)
            index = 0
            space_w, _ = draw.textsize(" ", font=font)

        if label == "dos_answer":
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
            label_text = fake.date(pattern=random.choice(patterns))

        if label == "pan_answer":
            label_text = fake.member_id()
        if label == "member_number_answer":
            label_text = fake.member_id()
        if label == "member_name_answer" or label == "patient_name_answer":
            label_text = fake.name()
            if np.random.choice([0, 1], p=[0.5, 0.5]):
                label_text = label_text.upper()

        # partition data into boxes splitting on blank spaces
        text_chunks = label_text.split(" ")
        text_width, text_height = draw.textsize(label_text, font=font)

        start_x = 0
        segments = []
        padding_x = space_w // 2

        if len(text_chunks) == 1:
            box = [start_x, 0, width, text_height]
            segments.append({"text": label_text, "box": box})
        else:
            for i, chunk in enumerate(text_chunks):
                chunk_width, chunk_height = draw.textsize(chunk, font=font)
                # x0, y0, x1, y1
                end_x = min(start_x + chunk_width + padding_x, width)
                box = [start_x, 0, end_x, text_height]
                segments.append({"text": chunk, "box": box})
                start_x += chunk_width
                if i < len(text_chunks):
                    start_x += space_w

        if text_width < width:
            # print(
            #     f"GEN [{label}, {font_size}, {est_line_count} : {text_height} :  {round(rat, 2)}] : {width} , {height} :  [{text_width}, {text_height} ] >   {label_text}"
            # )
            break
        index = index + 1

    return font_size, label_text, segments


# @Timer(text="Aug in {:.4f} seconds")
def __augment_decorated_process(guid: int, count: int, file_path: str, src_dir: str, dest_dir: str):

    output_aug_images_dir = ensure_exists(os.path.join(dest_dir, "images"))
    output_aug_annotations_dir = ensure_exists(os.path.join(dest_dir, "annotations"))

    ann_dir = os.path.join(src_dir, "annotations")
    img_dir = os.path.join(src_dir, "images")

    # file_path = os.path.join(ann_dir, file)
    file = file_path.split("/")[-1]
    print(f"File: {file_path}")

    try:
        with open(file_path, "r", encoding="utf8") as f:
            data = json.load(f)
    except Exception as e:
        raise e

    image_path = os.path.join(img_dir, file)
    image_path = image_path.replace("json", "png")
    filename = image_path.split("/")[-1].split(".")[0]

    image_masked, size = load_image_pil(image_path)

    for k in range(0, count):
        print(f"Iter : {guid} , {k} of {count} ; {filename} ")
        font_face = np.random.choice(["FreeSansOblique.ttf", "FreeSansBold.ttf", "FreeSansBold.ttf", "FreeSans.ttf"])
        font_path = os.path.join("./assets/fonts", font_face)
        draw = ImageDraw.Draw(image_masked)
        data_copy = dict()  # copy.deepcopy(data["form"])
        data_copy["form"] = []

        for i, item in enumerate(data["form"]):
            label = item["label"]
            if label == "other" or not label.endswith("_answer"):
                data_copy["form"].append(item)
                continue

            # pan_answer  dos_answer member_number_answer
            # format : x0,y0,x1,y1
            box = np.array(item["box"]).astype(np.int32)
            x0, y0, x1, y1 = box
            w = x1 - x0
            h = y1 - y0
            xoffset = 5
            yoffset = 0

            # Generate text inside image
            font_size, label_text, segments = generate_text(label, w, h, font_path)
            # font = ImageFont.truetype(font_path, font_size)  # need to get this from cache
            font = get_cached_font(font_path, font_size)

            # x0, y0, x1, y1 = xy
            # Yellow with outline for debug
            # draw.rectangle(((x0, y0), (x1, y1)), fill="#FFFFCC", outline="#FF0000", width=1)
            # clear region
            draw.rectangle(((x0, y0), (x1, y1)), fill="#FFFFFF")

            dup_item = item  # copy.copy(item)
            dup_item["text"] = label_text
            dup_item["id"] = str(uuid.uuid4())  # random.randint(50000, 250000)
            dup_item["words"] = []
            dup_item["linking"] = []
            words = []

            for seg in segments:
                seg_text = seg["text"]
                sx0, sy0, sx1, sy1 = seg["box"]
                sw = sx1 - sx0
                adj_box = [x0 + sx0, y0, x0 + sx0 + sw, y1]
                word = {"text": seg_text, "box": adj_box}
                words.append(word)
                # draw.rectangle(((adj_box[0], adj_box[1]), (adj_box[2], adj_box[3])), outline="#00FF00", width=1)

            dup_item["words"] = words
            draw.text((x0 + xoffset, y0 + yoffset), text=label_text, fill="#000000", font=font, stroke_fill=1)
            # draw.text((x0 + xoffset, y0 + yoffset), text=label_text, fill="#FF0000", font=font, stroke_fill=1)
            index = i + 1
            # print("-" * 20)
            # print(item)
            # print(dup_item)
            data_copy["form"].append(dup_item)
            # data["form"].append(dup_item)

        # Save items
        out_name_prefix = f"{filename}_{guid}_{k}"

        json_path = os.path.join(output_aug_annotations_dir, f"{out_name_prefix}.json")
        dst_img_path = os.path.join(output_aug_images_dir, f"{out_name_prefix}.png")

        # print(f'Writing : {json_path}')
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

    ann_dir = os.path.join(src_dir, "annotations")
    # mp.cpu_count()

    if False:
        for guid, file in enumerate(sorted(os.listdir(ann_dir))):
            file_path = os.path.join(ann_dir, file)
            __augment_decorated_process(guid, count, file_path, src_dir, dest_dir)

    # slower comparing to  pool.starmap
    if False:
        futures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=int(mp.cpu_count() * 0.75)) as executor:
            for guid, file in enumerate(sorted(os.listdir(ann_dir))):
                file_path = os.path.join(ann_dir, file)
                feature = executor.submit(__augment_decorated_process, guid, count, file_path, src_dir, dest_dir)
                futures.append(feature)

        for future in concurrent.futures.as_completed(futures):
            try:
                print(future.result())
            except Exception as e:
                print(e)

            print("All tasks has been finished")

    if True:
        args = []
        for guid, file in enumerate(sorted(os.listdir(ann_dir))):
            file_path = os.path.join(ann_dir, file)
            if False and file != "152611418_2.json":
                continue

            print(file)
            __args = (guid, count, file_path, src_dir, dest_dir)
            args.append(__args)

        results = []
        start = time.time()
        print("\nPool Executor:")
        print("Time elapsed: %s" % (time.time() - start))

        pool = Pool(processes=int(mp.cpu_count() * 0.9))
        pool_results = pool.starmap(__augment_decorated_process, args)

        pool.close()
        pool.join()

        print("Time elapsed[submitted]: %s" % (time.time() - start))
        for r in pool_results:
            print("Time elapsed[result]: %s  , %s" % (time.time() - start, r))
            # results.append(result)
        print("Time elapsed[all]: %s" % (time.time() - start))


def load_image_pil(image_path):
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    return image, (w, h)


def visualize_funsd(src_dir: str):
    ann_dir = os.path.join(src_dir, "annotations")
    img_dir = os.path.join(src_dir, "images")

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
        label2color = {
            "pan": "blue",
            "pan_answer": "green",
            "dos": "orange",
            "dos_answer": "violet",
            "member": "blue",
            "member_answer": "green",
            "member_number": "blue",
            "member_number_answer": "green",
            "member_name": "blue",
            "member_name_answer": "green",
            "patient_name": "blue",
            "patient_name_answer": "green",
            "other": "red",
        }

        for i, item in enumerate(data["form"]):
            predicted_label = item["label"].lower()
            color = label2color[predicted_label]

            for word in item["words"]:
                box = word["box"]
                draw.rectangle(box, outline=color, width=1)

            box = item["box"]

            if predicted_label != "other":
                draw.rectangle(box, outline=color, width=1, fill=(0, 180, 0, 50))
            else:
                draw.rectangle(box, outline=color, width=1)

            draw.text((box[0] + 10, box[1] - 10), text=predicted_label, fill=color, font=font, stroke_width=0)

        image.save(f"/tmp/snippet/viz_{filename}.png")


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
    print(f"Recalling annotation : {src_json_path.split('/')[-1]}, {src_image_path.split('/')[-1]}")

    filename = src_image_path.split("/")[-1].split(".")[0]
    image, orig_size = load_image_pil(src_image_path)
    resized, target_size = __scale_height(image, 1000)
    # resized.save(f"/tmp/snippet/resized_{filename}.png")

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
                word["box"] = resize_align_bbox(bbox, orig_w, orig_h, target_w, target_h)
    except Exception as ex:
        print(src_json_path)
        print(ex)
        raise ex

    return data, resized


def __rescale_annotate_frames(ann_dir_dest, img_dir_dest, filename, json_path, image_path):
    if False and filename != "152618378_2":
        return

    # 152630220_3  152618378_2 152618400  152624795_3
    print(f"filename : {filename}")
    print(f"json_path : {json_path}")
    print(f"image_path : {image_path}")

    data, image = rescale_annotation_frame(json_path, image_path)

    # Figure out how to handle this
    # if the width > 1000 SKIP for now
    if max(image.size) > 1000:
        print(f"Skipping image due to size[{image.size}] : {filename}")
        return

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
            __rescale_annotate_frames(ann_dir_dest, img_dir_dest, filename, json_path, image_path)

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
        pool_results = pool.starmap(__rescale_annotate_frames, args)

        pool.close()
        pool.join()

        print("Time elapsed[submitted]: %s" % (time.time() - start))
        for r in pool_results:
            print("Time elapsed[result]: %s  , %s" % (time.time() - start, r))
        print("Time elapsed[all]: %s" % (time.time() - start))


if __name__ == "__main__":
    name = "test"

    # Home
    root_dir = "/home/greg/dataset/assets-private/corr-indexer"
    root_dir_converted = "/home/greg/dataset/assets-private/corr-indexer-converted"
    root_dir_aug = "/home/greg/dataset/assets-private/corr-indexer-augmented"

    # GPU-001
    root_dir = "/data/dataset/private/corr-indexer"
    root_dir_converted = "/data/dataset/private/corr-indexer-converted"
    root_dir_aug = "/data/dataset/private/corr-indexer-augmented"

    # LP-01
    root_dir = "/home/gbugaj/dataset/private/corr-indexer"
    root_dir_converted = "/home/gbugaj/dataset/private/corr-indexer-converted"
    root_dir_aug = "/home/gbugaj/dataset/private/corr-indexer-augmented"

    src_dir = os.path.join(root_dir, f"{name}deck-raw-01")
    dst_path = os.path.join(root_dir, "dataset", f"{name}ing_data")
    aligned_dst_path = os.path.join(root_dir_converted, "dataset", f"{name}ing_data")

    aug_dest_dir = os.path.join(root_dir, "dataset-aug", f"{name}ing_data")
    aug_aligned_dst_path = os.path.join(root_dir_aug, "dataset", f"{name}ing_data")

    # cat 152611418_2_2_8.json

    # TRAIN -> 1, 2, 3
    # TEST  -> 1, 2, 3

    # STEP 1 : Convert COCO to FUNSD like format
    # convert_coco_to_funsd(src_dir, dst_path)

    # STEP 2
    decorate_funsd(dst_path)

    # STEP 3
    # augment_decorated_annotation(count=1000, src_dir=dst_path, dest_dir=aug_dest_dir)

    # Step 4
    # rescale_annotate_frames(src_dir=aug_dest_dir, dest_dir=aug_aligned_dst_path)

    # Debug INFO
    # visualize_funsd("/home/gbugaj/dataset/private/corr-indexer/dataset/testing_data")
    # visualize_funsd(aug_dest_dir)

    # visualize_funsd(aug_aligned_dst_path)

    # /home/gbugaj/dataset/private/corr-indexer/dataset-aug/testing_data/images/152658536_0_2_9.png
    # # STEP 2 : No Augmentation
    # rescale_annotate_frames(src_dir=dst_path, dest_dir=aligned_dst_path)
    # visualize_funsd(aligned_dst_path)