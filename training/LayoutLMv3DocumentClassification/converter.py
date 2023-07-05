import argparse
import concurrent.futures
import distutils.util
import glob
import hashlib
import io
import json
import logging
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
import rstr
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

from marie.ocr import CoordinateFormat, DefaultOcrEngine
from marie.utils.json import store_json_object, load_json_file

from marie.logging.profile import TimeContext
from marie.ocr.extract_pipeline import ExtractPipeline, split_filename, s3_asset_path
from marie.utils.docs import frames_from_file
import torch


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

                        print(f"image.size = {image.size}")

                        # save the image
                        dst_folder = os.path.join(dst_dir, folder)
                        print(f"dst_folder = {dst_folder}")
                        ensure_exists(dst_folder)

                        # save the image in PNG format
                        filename, file_extension = os.path.splitext(file)
                        print(f"file_extension = {file_extension}")

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

                    image, orig_size = load_image_pil(file_path)
                    resized, target_size = __scale_height(image, 1000)

                    print(f"image.size = {image.size}")

                    # save the image
                    dst_folder = os.path.join(dst_dir, folder)
                    print(f"dst_folder = {dst_folder}")
                    ensure_exists(dst_folder)

                    # save the image in PNG format
                    filename, file_extension = os.path.splitext(file)
                    print(f"file_extension = {file_extension}")

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

    #  python converter.py rescale --dir ~/datasets/private/medical_page_classification/raw --dir-output ~/datasets/private/medical_page_classification/output/images
    #  python converter.py decorate --dir ~/datasets/private/medical_page_classification/output/images --dir-output ~/datasets/private/medical_page_classification/output/annotations

    args = extract_args()
    print("-" * 120)
    print(args)
    print("-" * 120)

    args.func(args)
