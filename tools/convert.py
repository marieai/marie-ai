import argparse
import json
import os
import glob
import distutils.util
import shutil

from tools import from_json_file, ensure_exists


def _validate_qa_balance(
    category_counts: dict, link_map: dict, id_to_name: dict
) -> dict:
    """Validate the question/answer balance given a set of categories with their total
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
        question_count, answer_count = (
            category_counts[question],
            category_counts[answer],
        )

        if question_count != answer_count:
            category_imbalance[(question, answer)] = (question_count, answer_count)

    return category_imbalance


def _convert_coco_to_funsd(
    images_path: str,
    output_path: str,
    annotations_filename: str,
    config: object,
    strip_file_path: bool,
) -> None:
    """
    Convert CVAT annotated COCO dataset into FUNSD compatible format for finetuning models.
    """
    print("******* Conversion info ***********")
    print(f"image_path     : {images_path}")
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
    unknown_categories = [
        category["name"]
        for category in coco["categories"]
        if not (category["name"] in config["link_map"])
    ]
    if len(unknown_categories) > 0:
        raise Exception(
            f"COCO file has categories not found in your config: {unknown_categories}"
        )

    # Initialize Image based contexts
    annotations_by_image = {}
    images_by_id = {}
    for image in coco["images"]:
        annotations_by_image[image["id"]] = []
        images_by_id[image["id"]] = (
            image["file_name"].split("/")[-1] if strip_file_path else image["file_name"]
        )

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
    category_imbalance = _validate_qa_balance(
        annotation_category_counts, link_map, config_id_to_name
    )

    if len(category_imbalance) > 0:
        print(
            f"WARNING: {len(category_imbalance)} Question/Answer pairs have an imbalanced number of annotations!"
        )
        print("Proceeding to Identify individual offenders image offenders.")
        # Identify question/answer imbalance by file
        images_with_cat_imbalance = {}
        for image_id, annotations in annotations_by_image.items():
            image_category_counts = category_counts.copy()
            for annotation in annotations:
                image_category_counts[cat_by_id[annotation["category_id"]]] += 1
            imbalance = _validate_qa_balance(
                image_category_counts, link_map, config_id_to_name
            )
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
    src_images = {
        dir.name.split(".")[0] for dir in os.scandir(images_path) if dir.is_file()
    }
    for image_id, image_annotations in annotations_by_image.items():
        filename = images_by_id[image_id].split("/")[-1].split(".")[0]
        if (
            filename not in src_images
        ):  # Check to see if this annotation is a part of this dataset
            continue
        form_dict = {"form": []}
        for i, annotation in enumerate(image_annotations):
            # Convert form XYWH -> xmin,ymin,xmax,ymax
            bbox = [int(x) for x in annotation["bbox"]]
            bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]

            category_name = cat_by_id[annotation["category_id"]]
            label = category_name

            form_dict["form"].append(
                {
                    "id": i,
                    "text": "POPULATE_VIA_ICR",
                    "box": bbox,
                    "linking": [
                        link_map[category_name]
                    ],  # TODO: Not in use. Will need refactor to use.
                    "label": label,
                    "words": [
                        {"text": "POPULATE_VIA_ICR_WORD", "box": [0, 0, 0, 0]},
                    ],
                }
            )

        src_img_path = os.path.join(images_path, f"{filename}.png")
        json_path = os.path.join(output_path, "annotations_tmp", f"{filename}.json")
        dst_img_path = os.path.join(output_path, "images", f"{filename}.png")
        # Save tmp state FUNSD JSON
        with open(json_path, "w") as json_file:
            json.dump(form_dict, json_file, indent=4)
        # Copy respective image with the above annotations
        shutil.copyfile(src_img_path, dst_img_path)


def convert_coco_to_funsd(
    src_dir: str,
    image_path: str,
    output_path: str,
    config: object,
    strip_file_name_path: bool,
) -> None:
    """
    Convert CVAT annotated COCO 1.0 dataset into FUNSD compatible format for finetuning models.
    source: "FUNSD: A Dataset for Form Understanding in Noisy Scanned Documents" https://arxiv.org/pdf/1905.13538.pdf
    """
    os.makedirs(output_path, exist_ok=True)
    src_dir = os.path.expanduser(src_dir)
    items = glob.glob(
        os.path.join(src_dir, "annotations/*.json")
    )  # instances_default.json
    if len(items) == 0:
        raise Exception(f"No annotations to process in : {src_dir}")

    for idx, annotations_filename in enumerate(items):
        try:
            print(f"Processing annotation : {annotations_filename}")
            _convert_coco_to_funsd(
                image_path,
                output_path,
                annotations_filename,
                config,
                strip_file_name_path,
            )
        except Exception as e:
            raise e


def default_convert(args: object):

    print("Default convert")
    print(args)
    print("*" * 180)
    mode = args.mode
    suffix = args.mode_suffix
    args.src_dir = os.path.expanduser(args.src_dir)
    strip_file_name_path = args.strip_file_name_path
    src_dir = os.path.abspath(args.src_dir)
    data_dir = os.path.join(args.src_dir, args.dataset_path)
    args.config = os.path.expanduser(args.config)

    dst_path = (
        args.dir_converted
        if args.dir_converted != "./converted"
        else os.path.join(args.src_dir, "output", f"{mode}")
    )

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"File not found : {args.config}")

    # load config file
    config = from_json_file(args.config)

    print(f"mode       = {mode}")
    print(f"suffix     = {suffix}")
    print(f"src_dir    = {src_dir}")
    print(f"dst_path   = {dst_path}")

    convert_coco_to_funsd(src_dir, data_dir, dst_path, config, strip_file_name_path)


def get_convert_parser(subparsers=None) -> argparse.ArgumentParser:
    """
    Argument parser

    PYTHONPATH="$PWD" python convert.py --mode test --step augment --strip_file_name_path true --dir ~/dataset/private/corr-indexer --config ~/dataset/private/corr-indexer/config.json

    :param subparsers: If left None, a new ArgumentParser will be made. Otherwise pass the object generated from
                       argparse.ArgumentParser(...).add_subparsers(...) to add this as a subparser.
    """
    if subparsers is not None:
        convert_parser = subparsers.add_parser(
            "convert",
            help="Convert documents from COCO to FUNSD-Like intermediate format",
        )
    else:
        convert_parser = argparse.ArgumentParser(
            prog="convert", description="COCO to FUNSD conversion"
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
        type=lambda x: bool(distutils.util.strtobool(x)),
        default=False,
        help="Should full image paths be striped from annotations file",
    )

    convert_parser.add_argument(
        "--src_dir",
        required=True,
        type=str,
        default="~/dataset/ds-001/indexer",
        help="Base data directory",
    )

    convert_parser.add_argument(
        "--dataset_path",
        required=True,
        type=str,
        default="images/my-project/my-data",
        help="A relative path from your src_dir: {source dir}/images/{Project name}/{Dataset name} ",
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
        default="./config.json",
        help="Configuration file used for conversion",
    )

    convert_parser.add_argument(
        "--debug",
        required=False,
        action="store_true",
        default=False,
        help="Turn on debug logging",
    )

    return convert_parser


if __name__ == "__main__":
    args = get_convert_parser().parse_args()
    print("-" * 120)
    print(args)
    print("-" * 120)
    args.func(args)
