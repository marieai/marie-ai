import argparse
import distutils.util
import os
import shutil
import sys
import numpy as np

from tools import convert, decorate, augmenter, __tmp_path__

# FUNSD format can be found here
# https://guillaumejaume.github.io/FUNSD/description/

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

    if os.path.exists(os.path.join(output_path, "train")) or os.path.exists(os.path.join(output_path, "test")):
        raise Exception("Output directory not empty, manually remove test/train directories.")

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
    convert.default_convert(Namespace(**args_1))
    decorate.default_decorate(Namespace(**args_2))
    augmenter.default_augment(Namespace(**args_3))


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

    convert.get_convert_parser(subparsers)
    decorate.get_decorate_parser(subparsers)
    augmenter.get_augmenter_parser(subparsers)

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
        "--dir_output",
        default=f"{__tmp_path__}/split",
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

    convert_all_parser.add_argument(
        "--mask_config",
        required=True,
        type=str,
        default='./mask.json',
        help="Configuration file used for augmentation",
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
