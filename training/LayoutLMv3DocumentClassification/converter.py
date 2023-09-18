import argparse
import json
import logging
import os
import re
import sys

import torch
from PIL import Image

from marie.boxes.box_processor import PSMode
from marie.ocr import CoordinateFormat, DefaultOcrEngine
from marie.utils.docs import frames_from_file
from marie.utils.json import store_json_object
from marie.utils.utils import ensure_exists

logger = logging.getLogger(__name__)


def load_image_pil(image_path):
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    return image, (w, h)


def __scale_height(img, target_size, method=Image.LANCZOS):
    ow, oh = img.size
    scale = oh / target_size
    w = ow / scale
    h = target_size  # int(max(oh / scale, crop_size))
    resized = img.resize((int(w), int(h)), method)
    return resized, resized.size


def default_decorate(args: object):
    print("Default decorate")
    print(args)
    print("*" * 180)

    # This should be our dataset folder
    src_dir = args.dir
    dst_dir = args.dir_output

    print(f"src_dir   = {src_dir}")
    print(f"dst_dir   = {dst_dir}")

    use_cuda = torch.cuda.is_available()
    ocr_engine = DefaultOcrEngine(cuda=use_cuda)

    for folder in os.listdir(src_dir):
        print(f"folder = {folder}")
        folder_path = os.path.join(src_dir, folder)
        if os.path.isdir(folder_path):
            print(f"folder_path = {folder_path}")

            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                if os.path.isfile(file_path):
                    try:
                        print(f"file_path = {file_path}")
                        image, orig_size = load_image_pil(file_path)
                        # save the image
                        dst_folder = os.path.join(dst_dir, folder)
                        # print(f"dst_folder = {dst_folder}")
                        ensure_exists(dst_folder)

                        # save the image in PNG format
                        filename, file_extension = os.path.splitext(file)
                        dst_file = os.path.join(dst_folder, f"{filename}.json")
                        print(f"dst_file = {dst_file}")
                        if os.path.isfile(dst_file):
                            print(f"File already exists: {dst_file}")
                            try:
                                with open(dst_file) as f:
                                    json.load(f)
                                print(f"File is valid json: {dst_file}")
                                continue
                            except Exception as e:
                                print(f"File is not valid json: {dst_file}")

                        frames = frames_from_file(file_path)
                        results = ocr_engine.extract(
                            frames, PSMode.SPARSE, CoordinateFormat.XYWH, regions=[]
                        )
                        store_json_object(results, dst_file)

                    except Exception as e:
                        print(f"Error: {e}")


def default_rescale(args: object):
    print("Default rescale")
    print(args)
    print("*" * 180)

    # This should be our dataset folder
    src_dir = args.dir
    dst_dir = args.dir_output
    normalize = args.normalize

    print(f"src_dir   = {src_dir}")
    print(f"dst_dir   = {dst_dir}")

    for folder in os.listdir(src_dir):
        print(f"folder = {folder}")
        folder_path = os.path.join(src_dir, folder)
        print(f"folder_path = {folder_path}")

        if os.path.isdir(folder_path):
            print(f"folder_path = {folder_path}")

            for file in os.listdir(folder_path):
                print(f"file = {file}")
                file_path = os.path.join(folder_path, file)
                print(f"file_path = {file_path}")
                if os.path.isfile(file_path):
                    print(f"file_path = {file_path}")

                    # check if the file is an image
                    filename, file_extension = os.path.splitext(file_path)
                    if file_extension.lower() not in [
                        ".jpg",
                        ".jpeg",
                        ".png",
                        ".tif",
                        ".tiff",
                    ]:
                        print(f"Skipping {file_path} : {file_extension} not supported")
                        continue
                    image, orig_size = load_image_pil(file_path)
                    # NO SCALE NEEDED
                    # resized, target_size = __scale_height(image, 1000)
                    resized = image

                    # save the image
                    dst_folder = os.path.join(dst_dir, folder)
                    ensure_exists(dst_folder)
                    filename, file_extension = os.path.splitext(file)

                    if normalize:
                        normalized = os.path.basename(file)
                        normalized = normalized.split(".")
                        normalized = normalized[0]
                        normalized = normalized.lower()
                        filename = re.sub(r'\W+', '_', normalized)

                    # save the image in PNG format
                    dst_file = os.path.join(dst_folder, f"{filename}.png")
                    print(f"dst_file = {dst_file}")
                    resized.save(dst_file)


def default_visualize(args: object):
    print("Default visualize")
    print(args)
    print("*" * 180)


def extract_args(args=None) -> object:
    """
    Argument parser

    PYTHONPATH="$PWD" python ./marie/coco_funsd_converter.py --mode test --step augment --strip_file_name_path true --dir ~/dataset/private/corr-indexer --config ~/dataset/private/corr-indexer/config.json --aug-count 2
    """
    parser = argparse.ArgumentParser(
        prog="converter", description="LayoutLmV3 conversion utility"
    )

    subparsers = parser.add_subparsers(
        dest='command', help='Commands to run', required=True
    )

    decorate_parser = subparsers.add_parser(
        "decorate", help="Decorate documents(Box detection, ICR)"
    )

    decorate_parser.set_defaults(func=default_decorate)

    decorate_parser.add_argument(
        "--dir",
        required=True,
        type=str,
        help="Base dataset directory where the document for decorating resize",
    )

    decorate_parser.add_argument(
        "--dir-output",
        default="./rescaled",
        type=str,
        help="Destination directory",
    )

    rescale_parser = subparsers.add_parser(
        "rescale", help="Rescale/Normalize documents to be used by UNILM"
    )
    rescale_parser.set_defaults(func=default_rescale)

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

    rescale_parser.add_argument(
        "--normalize",
        default=True,
        type=bool,
        help="Should we normalize the file names (remove spaces etc.)",
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
        default="/tmp/visualize",
        type=str,
        help="Destination directory",
    )

    visualize_parser.add_argument(
        "--config",
        type=str,
        default='./visualize-config.json',
        help="Configuration file used for conversion",
    )

    try:
        return parser.parse_args(args) if args else parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":

    #  Page classification dataset
    #  python converter.py rescale --dir ~/datasets/private/data-hipa/medical_page_classification/raw --dir-output ~/datasets/private/data-hipa/medical_page_classification/output/images
    #  python converter.py decorate --dir ~/datasets/private/data-hipa/medical_page_classification/output/images --dir-output ~/datasets/private/data-hipa/medical_page_classification/output/annotations

    # Payer dataset
    # python converter.py rescale --dir ~/datasets/private/data-hipa/payer/converted --dir-output ~/datasets/private/data-hipa/payer/output/images
    # python converter.py decorate --dir ~/datasets/private/data-hipa/payer/output/images --dir-output ~/datasets/private/data-hipa/payer/output/annotations

    # Corr dataset
    # python converter.py rescale --dir ~/datasets/corr-routing/raw --dir-output ~/datasets/corr-routing/converted
    # python converter.py rescale --dir ~/datasets/private/patpay-ner/303/images/patpay --dir-output ~/datasets/private/patpay-ner/303/images/patpay/converted

    # python converter.py rescale --dir ~/datasets/private/patpay-ner/complete/images/patpay --dir-output ~/datasets/private/patpay-ner/complete/images/patpay/converted

    args = extract_args()
    print("-" * 120)
    print(args)
    print("-" * 120)

    args.func(args)
