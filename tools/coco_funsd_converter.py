import argparse
import distutils.util
import os
import sys

from tools import convert, decorate, augmenter, split_dataset


# FUNSD format can be found here
# https://guillaumejaume.github.io/FUNSD/description/


def default_all_steps(args: object):
    from argparse import Namespace

    print("Default all_steps")
    print(args)
    print("*" * 180)

    root_dir = args.src_dir
    aug_count = args.aug_count
    dataset_dir = os.path.join(root_dir, "output")

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

    # execute each step
    convert.default_convert(Namespace(**args_1))
    decorate.default_decorate(Namespace(**args_2))
    augmenter.default_augment(Namespace(**args_3))


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

    split_dataset.get_split_parser(subparsers)
    convert.get_convert_parser(subparsers)
    decorate.get_decorate_parser(subparsers)
    augmenter.get_augmenter_parser(subparsers)

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
        "--src_dir",
        required=True,
        type=str,
        default="~/dataset/ds-001/indexer",
        help="Base data directory",
    )

    convert_all_parser.add_argument(
        "--dataset_path",
        required=True,
        type=str,
        default="images/my-project/my-data",
        help="A relative path from your src_dir: {source dir}/images/{Project name}/{Dataset name} ",
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
