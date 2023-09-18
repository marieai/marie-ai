import argparse
import json
import os
import random
import time
import numpy as np
import multiprocessing as mp

from tools import from_json_file, ensure_exists
from faker import Faker
from functools import lru_cache
from PIL import Image, ImageDraw, ImageFont
from marie.numpyencoder import NumpyEncoder
from marie.constants import __root_dir__

# setup data aug
fake = Faker("en_US")
fake_names_only = Faker(["it_IT", "en_US", "es_MX", "en_IN"])  # 'de_DE',


def load_image_pil(image_path):
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    return image, (w, h)


def generate_phone() -> str:
    """Generates a phone number"""
    return fake.phone_number()


def generate_date() -> str:
    """Generates a data from the patterns given:
    https://datatest.readthedocs.io/en/stable/how-to/date-time-str.html
    """
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
    return fake.date(pattern=random.choice(patterns))


def generate_money(
    max_value: float,
    neg_prob: float = 0.1,
    sign_prob: float = 0.2,
    dec_only_prob: float = 0.1,
) -> str:
    """Generate a random string money value within set conditions."""
    while True:
        label_text = fake.pricetag()
        generated_value = float(label_text.replace("$", "").replace(",", ""))
        if max_value >= generated_value:
            break
    # TODO: make the below conditionals a passable list of augmentations
    # Only decimal probability
    if np.random.choice([0, 1], p=[1 - dec_only_prob, dec_only_prob]):
        label_text = f".{label_text.split('.')[-1]}"
    # Negative value probability
    if np.random.choice([0, 1], p=[1 - neg_prob, neg_prob]):
        neg_type = np.random.choice([0, 1], p=[0.5, 0.5])
        label_text = (
            f"-{label_text}" if neg_type else f"({label_text.replace('$', '')})"
        )
    # contains $ probability
    if np.random.choice([0, 1], p=[1 - sign_prob, sign_prob]):
        label_text = label_text.replace("$", "")
    return label_text


def generate_name(length: int) -> str:
    """Generates a name given the number of words in the original string"""
    name = [fake_names_only.first_name()]
    if length > 1:
        name += [fake_names_only.last_name() for _ in range(1, length)]
    return " ".join(name)


def generate_address() -> str:
    """Generates a 2 line address"""
    return fake.address()


def generate_alpha_numeric(
    length: int, alpha: bool = True, numeric: bool = True
) -> str:
    """Generates a random alpha-numeric value based on the length and contents of the input

    :param length: Length of generated string
    :param alpha: generate string with alpha characters
    :param numeric: generate string with numeric characters
    """
    assert (
        alpha or numeric
    ), "Cannot generate string: param alpha or numeric must be True"
    return fake.password(
        length=length,
        digits=numeric,
        upper_case=alpha,
        special_chars=False,
        lower_case=False,
    )


def generate_text(original: str, mask_type: str) -> str:
    """Generate text for specific type of label"""
    if mask_type == "money":
        money = original.replace("$", "").replace(",", "").split(".")[0]
        # Ensure that the amount of money exceeds the number of decimal places of the original
        max_value = int("".join(["9"] * len(money))) if len(money) > 0 else 9
        return generate_money(max_value)
    elif mask_type == "date":
        return generate_date()
    elif mask_type == "name":
        return generate_name(len(original.split(" ")))
    elif mask_type == "address":
        return generate_address()
    elif mask_type == "phone":
        return generate_phone()
    elif mask_type == "alpha-numeric":  # Alpha-Numeric
        alpha = original.isalpha()
        numeric = original.isnumeric()
        if not alpha and not numeric:
            alpha = numeric = True
            if len(original) < 2:
                alpha = np.random.choice([True, False])
                numeric = not alpha
        return generate_alpha_numeric(len(original), alpha, numeric)
    else:
        raise Exception(f"Unknown mask type: {mask_type}")


@lru_cache(maxsize=20)
def get_cached_font(font_path, font_size):
    # return ImageFont.truetype(font_path, font_size, layout_engine=ImageFont.Layout.BASIC)
    return ImageFont.truetype(font_path, font_size)


def create_annotation_data(
    text: str, width: int, height: int, font_path: str, dec: int = 2
):
    """
    Create FUNSD annotations from a given text, using a given font, in a given area.

    :param text: any string with standard '\n' newline boundaries (if applicable)
    :param width: max pixel width of area.
    :param height: max pixel height of area.
    :param font_path: Path to font used for size reference.
    :param dec: font point decrement rate

    :return: 'words' field annotation, line coordinates relative, and font size(pt)
    """

    # Reference image area to calculate font size values
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    lines = text.splitlines()
    line_count = len(lines)
    font_size = int(
        (height / line_count) * 0.75
    )  # 72pt/96px = 0.75 point to pixel ratio
    font = get_cached_font(font_path, font_size)

    space_w, line_height = draw.textsize(" ", font=font)
    line_spacing = max(4, (height - line_height * line_count) // line_count - 1)

    # Calculate FUNSD format annotations
    index = 0
    text_height = 0
    word_annotations = []

    while index < line_count:
        line_text = lines[index]
        # text_width, text_height = draw.textsize(line_text, font=font)
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((0, 0), line_text, font=font)
        else:
            w, h = draw.textsize(line_text, font=font)
            bbox = (0, 0, w, h)
        text_width, text_height = bbox[2], bbox[3]

        # Can the text be contained?
        if (
            text_width > width
            or (text_height + line_spacing) * line_count - line_spacing > height
        ):
            # If not reduce the size of the font and start over
            font_size -= dec
            # assert font_size <= 0, f"Text cannot fit in the given area.\nText: {text}\nDimensions: {width}x{height}"
            font = get_cached_font(font_path, font_size)
            space_w, line_height = draw.textsize(" ", font=font)
            assert (
                line_height != 0
            ), f"Text cannot fit in the given area.\nText: {text}\nDimensions: {width}x{height}"
            line_spacing = max(4, (height - line_height * line_count) // line_count - 1)
            word_annotations = []
            index = 0
            continue

        start_x = 0
        start_y = index * (text_height + line_spacing)
        padding = space_w // 2
        words = lines[index].split(" ")
        for word in words:
            word_width, _ = draw.textsize(word, font=font)
            end_x = min(start_x + word_width + padding, width)
            box = [start_x, start_y, end_x, start_y + text_height]  # x0, y0, x1, y1
            word_annotations.append({"text": word, "box": box})
            start_x += word_width + space_w
        index += 1

    line_coordinates = [
        np.array([0, (text_height + line_spacing) * i]) for i in range(line_count)
    ]

    return word_annotations, line_coordinates, font_size


def _augment_decorated_process(
    count: int,
    file_path: str,
    image_path: str,
    font_dir: str,
    dest_annotation_dir: str,
    dest_image_dir: str,
    mask_config: dict,
) -> None:
    """Generate a number of new FUNSD annotation files and images for a given FUNSD annotation file and
    corresponding image that augment and mask the fields given in the mask_config.

    :param count: number of augmentations
    :param file_path: path to source FUNSD annotation JSON file
    :param image_path: path to source image PNG file
    :param dest_annotation_dir: path to output location for augmented annotations files
    :param dest_image_dir: path to output location for augmented image files
    :param mask_config: A dictionary for how and what fields to augment
        {
            'prefixes':         ['prefix to labels', ... ],
            'fonts':            ['font file name', ... ],
            'masks_by_type':    {'type': ['annotation label(no prefix)', ... ], ... }
        }
    """

    # Faker.seed(0)

    filename = file_path.split("/")[-1].split(".")[0]
    prefixes = mask_config["prefixes"] if "prefixes" in mask_config else None
    fonts = mask_config["fonts"] if "fonts" in mask_config else None
    print(f"File: {file_path}")
    with open(file_path, "r", encoding="utf8") as f:
        data = json.load(f)

    # Subset of annotations we don't intend to mask
    data_constant = {"form": []}
    i = 0
    while i < len(data["form"]):
        item = data["form"][i]
        label = item["label"][2:] if prefixes is not None else item["label"]
        # Add mask type to annotation item
        for mask_type in mask_config["masks_by_type"]:
            if label in mask_config["masks_by_type"][mask_type]:
                # if it is a string, skip it
                # this could be a  "_comment": "check_amt_text_answer is an alpha value tied to a money value."
                if isinstance(mask_config["masks_by_type"][mask_type], str):
                    continue
                data["form"][i]["mask_type"] = mask_type
                break
        # Remove annotations we don't intend to mask from 'data'
        if "mask_type" not in data["form"][i]:
            data_constant["form"].append(data["form"].pop(i))
        else:
            i += 1

    if len(data["form"]) == 0:
        print(
            f"SKIPPING File: {filename}. It has no fields that need masking from the config."
        )
        return None

    for k in range(count):
        print(f"Augmentation : {k+1} of {count} ; {filename} ")
        font_face = np.random.choice(fonts)
        font_path = os.path.join(font_dir, font_face)
        image_masked, size = load_image_pil(image_path)
        draw = ImageDraw.Draw(image_masked)

        data_aug = {"form": []}
        for item in data["form"]:
            # box format : x0,y0,x1,y1
            x0, y0, x1, y1 = np.array(item["box"]).astype(np.int32)
            w = x1 - x0
            h = y1 - y0

            aug_text = generate_text(item["text"], item["mask_type"])
            words_annotations, line_coordinates, font_size = create_annotation_data(
                aug_text, w, h, font_path
            )

            font = get_cached_font(font_path, font_size)

            # clear region
            draw.rectangle(((x0, y0), (x1, y1)), fill="#FFFFFF")

            # Yellow boxes with outline for debug
            if False:
                draw.rectangle(
                    ((x0, y0), (x1, y1)), fill="#FFFFCC", outline="#FF0000", width=1
                )

            x0_y0 = np.array([x0, y0])
            for i, text_line in enumerate(aug_text.splitlines()):
                draw.text(
                    x0_y0 + line_coordinates[i],
                    text=text_line,
                    fill="#000000",
                    font=font,
                    stroke_fill=1,
                )
            data_aug["form"].append(
                {
                    "id": item["id"],
                    "text": aug_text,
                    "label": item["label"],
                    "words": words_annotations,
                    "line_number": item["line_number"],
                    "word_index": item["word_index"],
                    "linking": [],
                }
            )

        data_aug["form"] += data_constant["form"]
        # Save items
        out_name_prefix = f"{filename}_{k}"

        json_path = os.path.join(dest_annotation_dir, f"{out_name_prefix}.json")
        dst_img_path = os.path.join(dest_image_dir, f"{out_name_prefix}.png")

        print(f"Writing : {json_path}")
        with open(json_path, "w") as json_file:
            json.dump(
                data_aug,
                json_file,
                # sort_keys=True,
                separators=(",", ": "),
                ensure_ascii=False,
                indent=2,
                cls=NumpyEncoder,
            )

        image_masked.save(dst_img_path, compress_level=2)

        del draw


def augment_decorated_annotation(
    count: int, src_dir: str, dest_dir: str, mask_config_path: str
):
    """Generate a number of new FUNSD annotation files and images for a given dataset that
    augment and mask the fields given in the mask_config.

    :param count: number of augmentations
    :param src_dir: path to a dataset with 'annotations' and 'images' directories
    :param dest_dir: path to output location for augmented files
    :param mask_config_path: A json in the format
        {
            "prefixes":         ["prefix to labels", ... ],
            "fonts":            ["font file name", ... ],
            "font_dir":         "DEFAULT",
            "masks_by_type":    {"type": ["annotation label(no prefix)", ... ], ... }
        }
    """
    mask_config = from_json_file(mask_config_path)
    font_dir = os.path.join(__root_dir__, "../assets/fonts")
    if mask_config["font_dir"] != "DEFAULT":
        font_dir = mask_config["font_dir"]

    font_dir = ensure_exists(font_dir)
    ann_dir = ensure_exists(os.path.join(src_dir, "annotations"))
    img_dir = ensure_exists(os.path.join(src_dir, "images"))
    dest_aug_annotations_dir = ensure_exists(os.path.join(dest_dir, "annotations"))
    dest_aug_images_dir = ensure_exists(os.path.join(dest_dir, "images"))

    aug_args = []
    for guid, file in enumerate(os.listdir(ann_dir)):
        file_path = os.path.join(ann_dir, file)
        img_path = os.path.join(img_dir, file.replace("json", "png"))
        fileid = file.split("_")[0]
        # if fileid != "179431630":
        #     continue

        print("Processing file: ", file_path)

        _augment_decorated_process(
            count,
            file_path,
            img_path,
            font_dir,
            dest_aug_annotations_dir,
            dest_aug_images_dir,
            mask_config,
        )

        _args = (
            count,
            file_path,
            img_path,
            font_dir,
            dest_aug_annotations_dir,
            dest_aug_images_dir,
            mask_config,
        )
        aug_args.append(_args)

    return
    start = time.time()
    print("\nPool Executor:")
    pool = mp.Pool(processes=int(mp.cpu_count() * 0.95))
    pool_results = pool.starmap(_augment_decorated_process, aug_args)
    print("Time elapsed: %s" % (time.time() - start))
    pool.close()
    pool.join()
    print("Time elapsed[submitted]: %s" % (time.time() - start))
    for r in pool_results:
        print("Time elapsed[result]: %s  , %s" % (time.time() - start, r))
    print("Time elapsed[all]: %s" % (time.time() - start))


def default_augment(args: object):
    print("Default augmenter")
    print(args)
    print("*" * 180)

    # This should be our dataset folder
    args.dir = os.path.expanduser(args.dir)
    args.dir_output = os.path.expanduser(args.dir_output)

    mode = args.mode
    aug_count = args.aug_count
    root_dir = args.dir
    src_dir = os.path.join(args.dir, f"{mode}")
    dst_dir = (
        args.dir_output
        if args.dir_output != "./augmented"
        else os.path.abspath(os.path.join(root_dir, f"{mode}-augmented"))
    )

    print(f"mode      = {mode}")
    print(f"aug_count = {aug_count}")
    print(f"src_dir   = {src_dir}")
    print(f"dst_dir   = {dst_dir}")
    augment_decorated_annotation(
        aug_count, src_dir, dst_dir, os.path.expanduser(args.mask_config)
    )


def get_augmenter_parser(subparsers=None) -> argparse.ArgumentParser:
    """
    Argument parser

    PYTHONPATH="$PWD" python augmentor.py --mode test --dir ~/datasets/CORR/output/dataset/

    :param subparsers: If left None, a new ArgumentParser will be made. Otherwise pass the object generated from
                       argparse.ArgumentParser(...).add_subparsers(...) to add this as a subparser.
    :return: an ArgumentParser either independent or attached as a subparser
    """
    if subparsers is not None:
        augmenter_parser = subparsers.add_parser(
            "augment",
            help="Creates a number of augmented images and annotation from a source dataset.",
        )
    else:
        augmenter_parser = argparse.ArgumentParser(
            prog="augmenter",
            description="Creates a number of augmented images and annotation from a source dataset.",
        )

    augmenter_parser.set_defaults(func=default_augment)

    augmenter_parser.add_argument(
        "--mode",
        required=True,
        type=str,
        default="train",
        help="Conversion mode : train/test/validate/etc",
    )

    augmenter_parser.add_argument(
        "--aug-count",
        required=True,
        type=int,
        help="Number of augmentations per annotation",
    )

    augmenter_parser.add_argument(
        "--dir",
        required=True,
        type=str,
        default="~/dataset/ds-001/indexer",
        help="Base data directory",
    )

    augmenter_parser.add_argument(
        "--dir_output",
        required=False,
        type=str,
        default="./augmented",
        help="Augmented data directory",
    )

    augmenter_parser.add_argument(
        "--mask_config",
        required=True,
        type=str,
        default="./mask.json",
        help="Configuration file used for augmentation",
    )

    return augmenter_parser


if __name__ == "__main__":
    args = get_augmenter_parser().parse_args()
    # logger = logging.getLogger(__name__)
    print("-" * 120)
    print(args)
    print("-" * 120)
    args.func(args)
