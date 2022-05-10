import io
import json
import logging
import os
import shutil
from functools import lru_cache

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


def __scale_height(img, target_size, method=Image.LANCZOS):
    ow, oh = img.size
    scale = oh / target_size
    print(scale)
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

            label = category_name
            if False:
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


def extract_icr(snippet, boxp, icrp):
    key = "coco"
    boxes, img_fragments, lines, _ = boxp.extract_bounding_boxes(key, "field", snippet, PSMode.SPARSE)
    if boxes is None or len(boxes) == 0:
        print("Empty boxes")
        return [], []
    result, overlay_image = icrp.recognize(key, "test", snippet, boxes, img_fragments, lines)

    return boxes, result


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

        for i, item in enumerate(data["form"]):
            # format : x0,y0,x1,y1
            box = np.array(item["box"]).astype(np.int32)
            x0, y0, x1, y1 = box
            snippet = image[y0:y1, x0:x1, :]
            # export cropped region
            file_path = os.path.join("/tmp/snippet", f"{guid}-snippet_{i}.png")
            cv2.imwrite(file_path, snippet)

            boxes, results = extract_icr(snippet, boxp, icrp)

            print(boxes)
            print(results)

            if results is None or len(results) == 0 or results["lines"] is None or len(results["lines"]) == 0:
                print(f"No results for : {guid}-{i}")
                continue

            file_path = os.path.join("/tmp/snippet", f"{guid}-snippet_{i}.png")
            cv2.imwrite(file_path, snippet)

            words = []
            text = ""
            try:
                text = " ".join([line["text"] for line in results["lines"]])
            except Exception as ex:
                # raise ex
                pass

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

        file_path = os.path.join("/tmp/snippet", f"{guid}-masked.png")
        cv2.imwrite(file_path, image_masked)

        boxes_masked, results_masked = extract_icr(image_masked, boxp, icrp)

        print(">>>>>>>>>>>>>")
        print(results_masked)
        print(data)

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
                indent=4,
                cls=NumpyEncoder,
            )

        # break


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
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        label2color = {"question": "blue", "answer": "green", "header": "orange", "other": "violet"}
        label2colorXXX = {
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
            # print(item)
            box = item["box"]
            predicted_label = item["label"].lower()
            draw.rectangle(box, outline=label2color[predicted_label], width=1)
            draw.text((box[0] + 10, box[1] - 10), text=predicted_label, fill=label2color[predicted_label], font=font)

            for word in item["words"]:
                box = word["box"]
                draw.rectangle(box, outline="red")

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


def rescale_annotation_frame(src_json_path, src_image_path):
    print(f"Recalling annotation json: {src_json_path}")
    print(f"Recalling annotation frame: {src_image_path}")

    filename = src_image_path.split("/")[-1].split(".")[0]
    image, orig_size = load_image_pil(src_image_path)
    resized, target_size = __scale_height(image, 1000)
    # resized.save(f"/tmp/snippet/resized_{filename}.png")

    print(f"orig_size   = {orig_size}")
    print(f"target_size = {target_size}")
    orig_w, orig_h = orig_size
    target_w, target_h = target_size
    data = from_json_file(src_json_path)

    for i, item in enumerate(data["form"]):
        bbox = tuple(item["box"])  # np.array(item["box"]).astype(np.int32)
        item["box"] = resize_align_bbox(bbox, orig_w, orig_h, target_w, target_h)
        for word in item["words"]:
            bbox = tuple(word["box"])  # np.array(item["box"]).astype(np.int32)
            word["box"] = resize_align_bbox(bbox, orig_w, orig_h, target_w, target_h)

    return data, resized


def rescale_annotate_frames(src_dir: str, dest_dir: str):

    ann_dir = os.path.join(src_dir, "annotations")
    img_dir = os.path.join(src_dir, "images")

    ann_dir_dest = ensure_exists(os.path.join(dest_dir, "annotations"))
    img_dir_dest = ensure_exists(os.path.join(dest_dir, "images"))

    for guid, file in enumerate(sorted(os.listdir(ann_dir))):
        json_path = os.path.join(ann_dir, file)
        filename = file.split("/")[-1].split(".")[0]
        image_path = os.path.join(img_dir, file)
        image_path = image_path.replace("json", "png")

        if False and filename != "152618378_2":
            continue
        # 152630220_3  152618378_2 152618400  152624795_3
        print(f"filename : {filename}")
        print(f"json_path : {json_path}")
        print(f"image_path : {image_path}")

        data, image = rescale_annotation_frame(json_path, image_path)

        # Figure out how to handle this
        # if the width > 1000 SKIP for now
        if max(image.size) > 1000:
            print(f"Skipping image due to size[{image.size}] : {filename}")
            continue

        json_path_dest = os.path.join(ann_dir_dest, f"{filename}.json")
        image_path_dest = os.path.join(img_dir_dest,  f"{filename}.png")

        # save image and json data
        image.save(image_path_dest)

        with open(json_path_dest, "w") as json_file:
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
    name = "test"
    root_dir = "/home/greg/dataset/assets-private/corr-indexer"
    root_dir_converted = "/home/greg/dataset/assets-private/corr-indexer-converted"


    root_dir = "/home/gbugaj/dataset/private/corr-indexer"
    root_dir_converted = "/home/gbugaj/dataset/private/corr-indexer-converted"
    
    # root_dir = "/home/gbugaj/data/private/corr-indexer"

    src_dir = os.path.join(root_dir, f"{name}deck-raw-01")
    dst_path = os.path.join(root_dir, "dataset", f"{name}ing_dataset")
    aligned_dst_path = os.path.join(root_dir_converted, "dataset", f"{name}ing_dataset")

    # convert_coco_to_funsd(src_dir, dst_path)
    # decorate_funsd(dst_path)

    # visualize_funsd(dst_path)

    # rescale_annotate_frames(src_dir=dst_path, dest_dir=aligned_dst_path)
    # visualize_funsd(aligned_dst_path)
    # visualize_funsd("/home/greg/dataset/funsd/dataset/testing_data")
    visualize_funsd("/home/gbugaj/dataset/funsd/dataset/testing_data")
    