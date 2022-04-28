import io
import json
import logging
import os
import shutil

import cv2
import numpy as np

from boxes.box_processor import PSMode
from boxes.craft_box_processor import BoxProcessorCraft
from document.trocr_icr_processor import TrOcrIcrProcessor
from numpyencoder import NumpyEncoder
from utils.utils import ensure_exists

# FUNSD format can be found here
# https://guillaumejaume.github.io/FUNSD/description/

logger = logging.getLogger(__name__)


def from_json_file(filename):
    with io.open(filename, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)
        return data


def convert_coco_to_funsd(src_dir: str, output_path: str = "/tmp/coco_funsd") -> None:
    """
    Convert CVAT annotated COCO dataset into FUNSD compatible format for finetuning models.
    """
    src_file = os.path.join(src_dir, "annotations/instances_default.json")
    print(f"src_dir : {src_dir}")
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

    ano_groups = {}
    # Group annotations by image_id as their key
    for ano in annotations:
        if ano["image_id"] not in ano_groups:
            ano_groups[ano["image_id"]] = []
        ano_groups[ano["image_id"]].append(ano)

    for group_id in ano_groups:
        grouping = ano_groups[group_id]
        # Validate that each annotation has associated question/answer pair
        found_cat_id = []
        for ano in grouping:
            found_cat_id.append(ano["category_id"])
        # if we have any missing mapping we will abort and fix the labeling data before continuing
        for question, answer in question_answer_map.items():
            qid = cat_name_id[question]
            aid = cat_name_id[answer]
            # we only have question but no answer
            if qid in found_cat_id and aid not in found_cat_id:
                raise Exception(
                    f"Pair not found for image_id[{group_id}] : {question} [{qid}] MISSING -> {answer} [{aid}]"
                )
            else:
                print(f"Pair found : {question} [{qid}] -> {answer} [{aid}]")

        # start conversion
        form_dict = {"form": []}

        for ano in grouping:
            category_id = ano["category_id"]
            bbox = [int(x) for x in ano["bbox"]]
            category_name = cat_id_name[category_id]

            item = {
                "id": id_map[category_name],
                "text": "POPULATE_VIA_ICR",
                "box": bbox,
                "linking": [link_map[category_name]],
                "label": category_name,
                "words": [
                    {"text": "POPULATE_VIA_ICR", "box": [0, 0, 0, 0]},
                ],
            }

            form_dict["form"].append(item)

        img_data = images_by_id[group_id]
        file_name = img_data["file_name"]
        filename = file_name.split("/")[-1].split(".")[0]

        src_img_path = os.path.join(src_dir, "images", file_name)
        os.makedirs(os.path.join(output_path, "annotations"), exist_ok=True)
        os.makedirs(os.path.join(output_path, "images"), exist_ok=True)

        json_path = os.path.join(output_path, "annotations", f"{filename}.json")
        dst_img_path = os.path.join(output_path, "images", f"{filename}.png")

        with open(json_path, "w") as json_file:
            json.dump(form_dict, json_file, indent=4)

        shutil.copyfile(src_img_path, dst_img_path)
        print(form_dict)


def load_image(image_path):
    image = cv2.imread(image_path)
    h, w = image.shape[0], image.shape[1]
    return image, (w, h)


def decorate_funsd(src_dir: str, output_ann_dir: str):
    work_dir_boxes = ensure_exists("/tmp/boxes")
    work_dir_icr = ensure_exists("/tmp/icr")
    output_ann_dir = ensure_exists(output_ann_dir)

    logger.info("⏳ Generating examples from = %s", src_dir)
    ann_dir = os.path.join(src_dir, "annotations")
    img_dir = os.path.join(src_dir, "images")

    boxp = BoxProcessorCraft(work_dir=work_dir_boxes, models_dir="./model_zoo/craft", cuda=False)
    icrp = TrOcrIcrProcessor(work_dir=work_dir_icr, cuda=False)

    for guid, file in enumerate(sorted(os.listdir(ann_dir))):
        file_path = os.path.join(ann_dir, file)
        with open(file_path, "r", encoding="utf8") as f:
            data = json.load(f)
        image_path = os.path.join(img_dir, file)
        image_path = image_path.replace("json", "png")
        image, size = load_image(image_path)

        for i, item in enumerate(data["form"]):
            # format : xywh
            box = np.array(item["box"]).astype(np.int32)
            print(box)
            x, y, w, h = box
            snippet = image[y : y + h, x : x + w :]
            # export cropped region
            file_path = os.path.join("/tmp/snippet", f"{guid}-snippet_{i}.png")
            cv2.imwrite(file_path, snippet)

            key = "coco"
            boxes, img_fragments, lines, _ = boxp.extract_bounding_boxes(key, "field", snippet, PSMode.SPARSE)
            result, overlay_image = icrp.recognize(key, "test", snippet, boxes, img_fragments, lines)

            print(boxes)
            print(result)

            file_path = os.path.join("/tmp/snippet", f"{guid}-snippet_{i}.png")
            cv2.imwrite(file_path, snippet)

            words = []
            text = " ".join([line["text"] for line in result["lines"]])
            print(text)

            for word in result["words"]:
                w_text = word["text"]
                wx, wy, ww, wh = word["box"]
                w_box = [wx + x, wy + y, ww, wh]
                adj_word = {"text": w_text, "box": w_box}
                words.append(adj_word)

            item["words"] = words
            item["text"] = text

        print(data)
        json_path = os.path.join(output_ann_dir, file)
        with open(json_path, "w") as json_file:
            json.dump(
                data,
                json_file,
                sort_keys=True,
                separators=(",", ": "),
                ensure_ascii=False,
                indent=4,
                cls=NumpyEncoder,
            )


if __name__ == "__main__":
    src_dir = "/home/gbugaj/data/private/corr-indexer/testdeck-raw-01"
    dst_step_1dir = "/tmp/coco_funsd/"
    dst_step_2dir = "/tmp/coco_funsd/adjusted_annotations"

    convert_coco_to_funsd(src_dir)
    decorate_funsd(dst_step_1dir, dst_step_2dir)
