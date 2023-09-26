import argparse
import json
import os
from copy import deepcopy

import numpy as np

from helpers import ensure_exists, copyFiles, from_json_file


# from tools import from_json_file, ensure_exists, copyFiles


def split_funsd_annotation_files(ann_dir: str, image_set):
    """ """
    ann_set = []
    for filename in image_set:
        ann_filename = os.path.split(filename)[1].replace(".png", ".json")
        ann_set.append(os.path.join(ann_dir, ann_filename))
    return ann_set


def split_coco_annotation_file(train: list, test: list, annotation_path: str):
    """ """
    coco_annotation_file = from_json_file(annotation_path)

    # Separate images and annotation from the original file

    # images = {"filename": image attributes, ...}
    images = {
        os.path.split(image["file_name"])[1]: image
        for image in coco_annotation_file.pop("images", [])
    }

    # annotations_by_image = {image_id: [{annotation attributes}, ...], ...}
    annotations = coco_annotation_file.pop("annotations", [])
    annotations_by_image = {annotation["image_id"]: [] for annotation in annotations}
    for annotation in annotations:
        annotations_by_image[annotation["image_id"]].append(annotation)

    def create_annotation_file(filenames):
        image_subset = [images[filename] for filename in filenames]
        annotation_subset = []
        for attributes in image_subset:
            annotation_subset += annotations_by_image[attributes["id"]]

        annotation_file = deepcopy(coco_annotation_file.copy())
        annotation_file["images"] = image_subset
        annotation_file["annotations"] = annotation_subset
        return annotation_file

    return create_annotation_file(train), create_annotation_file(test)


def split_dataset(
    src_dir,
    output_path,
    split_percentage: float = 0.8,
    format: str = "FUNSD",
    ann_filename: str = "instances_default.json",
):
    """
    Split dataset for training and test.

    ARGS:
        src_dir: input folder for COCO dataset, expected structure to have 'annotations' and 'image' folder.
        output_path: output folder where new train/test directory will be created.
        format: String indicating the format used for the annotation file. Currently supported formats:
                - "FUNSD"
                - "COCO"
        ann_filename: Name of annotation json file if only 1 is used for the format otherwise it is assumed all images
                      have a unique annotation file.
        split_percentage : split ration ex: .8  this will create 80/20 Train/Test split.
    """
    ann_dir = os.path.join(src_dir, "annotations")
    img_dir = os.path.join(src_dir, "images")

    assert os.path.exists(
        ann_dir
    ), "Source directory missing expected 'annotations' sub-directory"
    assert os.path.exists(
        img_dir
    ), "Source directory missing expected 'images' sub-directory"

    train_dir = os.path.join(output_path, "train")
    test_dir = os.path.join(output_path, "test")

    if os.path.exists(train_dir) or os.path.exists(test_dir):
        raise Exception(
            "Output directory not empty, manually remove test/train directories."
        )

    images = np.array([os.path.join(img_dir, image) for image in os.listdir(img_dir)])
    np.random.shuffle(images)
    total_images = len(images)

    assert total_images > 1, "Not enough images to split"
    training_size = int(total_images * split_percentage)

    train_set = images[:training_size]
    test_set = images[training_size:]

    ann_dir_out_train = ensure_exists(os.path.join(train_dir, "annotations"))
    img_dir_out_train = ensure_exists(os.path.join(train_dir, "images"))

    ann_dir_out_test = ensure_exists(os.path.join(test_dir, "annotations"))
    img_dir_out_test = ensure_exists(os.path.join(test_dir, "images"))

    if format == "COCO":
        json_path = os.path.join(ann_dir, ann_filename)
        assert os.path.exists(json_path), "Annotations file missing"
        train_ann, test_ann = split_coco_annotation_file(train_set, test_set, json_path)
        # Save annotation JSONs
        with open(os.path.join(ann_dir_out_train, ann_filename), "w") as json_file:
            json.dump(train_ann, json_file, indent=4)
        with open(os.path.join(ann_dir_out_test, ann_filename), "w") as json_file:
            json.dump(test_ann, json_file, indent=4)

    elif format == "FUNSD":
        train_ann_set = split_funsd_annotation_files(ann_dir, train_set)
        test_ann_set = split_funsd_annotation_files(ann_dir, test_set)

        copyFiles(train_ann_set, ann_dir_out_train)
        copyFiles(test_ann_set, ann_dir_out_test)

    copyFiles(train_set, img_dir_out_train)
    copyFiles(test_set, img_dir_out_test)


def default_split(args: object):
    print("Default split")
    print(args)
    print("*" * 180)

    print('args.dir_output', args.dir_output)
    print('args.dir', args.dir)
    print('args.dir_output', args.dir_output)

    dst_dir = (
        args.dir_output if args.dir_output != "./split" else os.path.abspath(args.dir)
    )

    print(f"src_dir = {args.dir}")
    print(f"dst_dir = {dst_dir}")
    print(f"ratio   = {args.ratio}")

    split_dataset(args.dir, dst_dir, args.ratio)


def get_split_parser(subparsers=None) -> argparse.ArgumentParser:
    """
    Argument parser

    PYTHONPATH="$PWD" python decorate.py --mode test --dir ~/datasets/CORR/output/dataset/

    :param subparsers: If left None, a new ArgumentParser will be made. Otherwise pass the object generated from
                       argparse.ArgumentParser(...).add_subparsers(...) to add this as a subparser.
    :return: an ArgumentParser either independent or attached as a subparser
    """
    if subparsers is not None:
        split_parser = subparsers.add_parser(
            "split", help="Split COCO dataset into train/test"
        )
    else:
        split_parser = argparse.ArgumentParser(
            prog="split_dataset", description="Split COCO dataset into train/test"
        )

    split_parser.set_defaults(func=default_split)

    split_parser.add_argument(
        "--dir",
        required=True,
        type=str,
        help="Source directory",
    )

    split_parser.add_argument(
        "--dir_output",
        default=f"./split",
        type=str,
        help="Destination directory",
    )

    split_parser.add_argument(
        "--ratio",
        default=0.8,
        type=float,
        help="Ratio of Test/Train split",
    )

    split_parser.add_argument(
        "--format",
        default=f"FUNSD",
        type=str,
        help="Annotation file format. *Currently supported: FUNSD, COCO",
    )

    return split_parser


if __name__ == "__main__":
    args = get_split_parser().parse_args()
    print("-" * 120)
    print(args)
    print("-" * 120)
    args.func(args)
