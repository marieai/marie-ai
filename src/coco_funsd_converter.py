import io
import json
import logging
import os
import shutil

import cv2
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

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


# https://stackoverflow.com/questions/23853632/which-kind-of-interpolation-best-for-resizing-image
def __scale_height(img, target_size, crop_size, method=Image.LANCZOS):
    ow, oh = img.size
    scale = oh / target_size
    print(scale)
    w = ow / scale
    h = target_size  # int(max(oh / scale, crop_size))
    return img.resize((int(w), int(h)), method)


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
        for ano in grouping:
            found_cat_id.append(ano["category_id"])
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
            label = "QUESTION"
            if category_name.find("answer") > -1:
                label = "ANSWER"

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

        with open(json_path, "w") as json_file:
            json.dump(form_dict, json_file, indent=4)

        # copy and resize to 1000 H
        shutil.copyfile(src_img_path, dst_img_path)
        print(form_dict)
        # break


def normalize_bbox(bbox, size):
    return [
        int(1000 * bbox[0] / size[0]),
        int(1000 * bbox[1] / size[1]),
        int(1000 * bbox[2] / size[0]),
        int(1000 * bbox[3] / size[1]),
    ]


def decorate_funsd(src_dir: str):
    work_dir_boxes = ensure_exists("/tmp/boxes")
    work_dir_icr = ensure_exists("/tmp/icr")
    output_ann_dir = ensure_exists(os.path.join(src_dir, "annotations"))

    logger.info("⏳ Generating examples from = %s", src_dir)
    ann_dir = os.path.join(src_dir, "annotations_tmp")
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

        print(f"size : {size}")
        for i, item in enumerate(data["form"]):
            # format : x0,y0,x1,y1
            box = np.array(item["box"]).astype(np.int32)
            x0, y0, x1, y1 = box
            snippet = image[y0:y1, x0:x1, :]
            # export cropped region
            file_path = os.path.join("/tmp/snippet", f"{guid}-snippet_{i}.png")
            cv2.imwrite(file_path, snippet)

            key = "coco"
            boxes, img_fragments, lines, _ = boxp.extract_bounding_boxes(key, "field", snippet, PSMode.SPARSE)
            if boxes is None or len(boxes) == 0:
                print("Empty boxes")
                continue
            result, overlay_image = icrp.recognize(key, "test", snippet, boxes, img_fragments, lines)

            print(boxes)
            print(result)

            if result is None or len(result) == 0 or result["lines"] is None or len(result["lines"]) == 0:
                print(f"No results for : {guid}-{i}")
                continue

            file_path = os.path.join("/tmp/snippet", f"{guid}-snippet_{i}.png")
            cv2.imwrite(file_path, snippet)

            words = []
            text = ""
            try:
                text = " ".join([line["text"] for line in result["lines"]])
            except Exception as ex:
                # raise ex
                pass

            # boxes are in stored in x0,y0,x1,y1 where x0,y0 is upper left corner and x1,y1 if bottom/right
            # we need to account for offset from the snippet box
            # results["word"] are in a xywh format in local position and need to be converted to relative position
            print("-------------------------------")
            print(result["words"])
            for word in result["words"]:
                w_text = word["text"]
                x, y, w, h = word["box"]
                w_box = [x0 + x, y0 + y, x0 + x + w, y0 + y + h]
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


def load_image_pil(image_path):
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    return image, (w, h)


def visualize_funsd(src_dir: str):
    ann_dir = os.path.join(src_dir, "annotations")
    img_dir = os.path.join(src_dir, "images")

    for guid, file in enumerate(sorted(os.listdir(ann_dir))):
        file_path = os.path.join(ann_dir, file)
        with open(file_path, "r", encoding="utf8") as f:
            data = json.load(f)

        filename = file.split("/")[-1].split(".")[0]
        image_path = os.path.join(img_dir, file)
        image_path = image_path.replace("json", "png")
        print(data)
        print(f"image_path : {image_path}")
        print(f"filename : {filename}")
        image, size = load_image_pil(image_path)

        # draw predictions over the image
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        label2color = {"question": "blue", "answer": "green", "header": "orange", "other": "violet"}

        for i, item in enumerate(data["form"]):
            print(item)
            box = item["box"]
            predicted_label = item["label"].lower()
            draw.rectangle(box, outline=label2color[predicted_label], width=2)
            draw.text((box[0] + 10, box[1] - 10), text=predicted_label, fill=label2color[predicted_label], font=font)

            for word in item["words"]:
                box = word["box"]
                draw.rectangle(box, outline="red")

        image.save(f"/tmp/snippet/viz_{filename}.png")


if __name__ == "__main__":
    name = "train"
    root_dir = "/home/greg/dataset/assets-private/corr-indexer"
    root_dir = "/home/gbugaj/data/private/corr-indexer"
    src_dir = os.path.join(root_dir, f"{name}deck-raw-01")
    dst_path = os.path.join(root_dir, "dataset", f"{name}_dataset")

    convert_coco_to_funsd(src_dir, dst_path)
    decorate_funsd(dst_path)
    visualize_funsd(dst_path)
