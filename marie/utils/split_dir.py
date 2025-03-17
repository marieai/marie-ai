import argparse
import math
import os
import random
import shutil
from typing import Optional


def ensure_exists(
    dir_to_validate, validate_dir_is_empty: Optional[bool] = False
) -> str:
    """Ensure directory exists and is empty if required.
    :param dir_to_validate: Directory to validate.
    :param validate_dir_is_empty: If True, the directory must be empty.
    :return: Directory to validate.
    """
    if not os.path.exists(dir_to_validate):
        os.makedirs(dir_to_validate, exist_ok=True)

    if validate_dir_is_empty and os.path.exists(dir_to_validate):
        if len(os.listdir(dir_to_validate)) > 0:
            raise ValueError(f"Directory {dir_to_validate} is not empty.")

    return dir_to_validate


def split_dir(dir_src, dir_dest, data_dir_name="image", mask_dir_name="mask"):
    dir_src = os.path.expanduser(dir_src)
    dir_dest = os.path.expanduser(dir_dest)

    print(f"dir_src : {dir_src}")
    print(f"dir_dest : {dir_dest}")

    print(f"data_dir_name : {data_dir_name}")
    print(f"mask_dir_name : {mask_dir_name}")

    # expecting two directories [image, masked]
    data_dir_src = os.path.join(dir_src, data_dir_name)
    mask_dir_src = os.path.join(dir_src, mask_dir_name)

    mask_filenames = os.listdir(mask_dir_src)
    mask_filenames = random.sample(mask_filenames, len(mask_filenames))

    size = len(mask_filenames)

    validation_size = math.ceil(size * 0.10)  # percent validation size
    test_size = math.ceil(size * 0.20)  # percent testing size
    training_size = size - validation_size - test_size  # percent training

    print(
        "Class >>  size = {} training = {} validation = {} test = {} ".format(
            size, training_size, validation_size, test_size
        )
    )

    validation_files = mask_filenames[:validation_size]
    testing_files = mask_filenames[validation_size : validation_size + test_size]
    training_files = mask_filenames[validation_size + test_size :]

    print("Number of training images   : {}".format(len(training_files)))
    print("Number of validation images : {}".format(len(validation_files)))
    print("Number of testing images    : {}".format(len(testing_files)))

    # prepare output directories
    test_image_dir_out = os.path.join(dir_dest, "test", "image")
    test_mask_dir_out = os.path.join(dir_dest, "test", "mask")

    train_image_dir_out = os.path.join(dir_dest, "train", "image")
    train_mask_dir_out = os.path.join(dir_dest, "train", "mask")

    validation_image_dir_out = os.path.join(dir_dest, "validation", "image")
    validation_mask_dir_out = os.path.join(dir_dest, "validation", "mask")

    ensure_exists(test_image_dir_out)
    ensure_exists(test_mask_dir_out)

    ensure_exists(train_image_dir_out)
    ensure_exists(train_mask_dir_out)

    ensure_exists(validation_image_dir_out)
    ensure_exists(validation_mask_dir_out)

    def copyfiles(files, srcDir, destDir):
        if not os.path.exists(destDir):
            os.makedirs(destDir)

        for filename in files:
            src = os.path.join(srcDir, filename)
            dest = os.path.join(destDir, filename)
            shutil.copy(src, dest)

    copyfiles(training_files, data_dir_src, train_image_dir_out)
    copyfiles(training_files, mask_dir_src, train_mask_dir_out)

    copyfiles(testing_files, data_dir_src, test_image_dir_out)
    copyfiles(testing_files, mask_dir_src, test_mask_dir_out)

    copyfiles(validation_files, data_dir_src, validation_image_dir_out)
    copyfiles(validation_files, mask_dir_src, validation_mask_dir_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split a directory into training, validation, and testing sets."
    )
    parser.add_argument("dir_src", type=str, help="Source directory")
    parser.add_argument("dir_dest", type=str, help="Destination directory")
    parser.add_argument(
        "--data_dir_name",
        type=str,
        default="image",
        help="Name of the data directory (default: 'image')",
    )
    parser.add_argument(
        "--mask_dir_name",
        type=str,
        default="mask",
        help="Name of the mask directory (default: 'mask')",
    )

    args = parser.parse_args()

    split_dir(args.dir_src, args.dir_dest, args.data_dir_name, args.mask_dir_name)
