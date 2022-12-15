import os

from utils.utils import ensure_exists


def split_dir(dir_src, dir_dest):
    import math
    import random
    import shutil

    print("dir_src : %s" % (dir_src))
    print("dir_dest : %s" % (dir_dest))

    # expecting two directories [image, masked]
    image_dir_src = os.path.join(dir_src, "image")
    mask_dir_src = os.path.join(dir_src, "mask")

    mask_filenames = os.listdir(mask_dir_src)
    mask_filenames = random.sample(mask_filenames, len(mask_filenames))

    size = len(mask_filenames)

    validation_size = math.ceil(size * 0.0)  # 5 percent validation size
    test_size = math.ceil(size * 0.20)  # 25 percent testing size
    training_size = size - validation_size - test_size  # 70 percent training
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

    copyfiles(training_files, image_dir_src, train_image_dir_out)
    copyfiles(training_files, mask_dir_src, train_mask_dir_out)

    copyfiles(testing_files, image_dir_src, test_image_dir_out)
    copyfiles(testing_files, mask_dir_src, test_mask_dir_out)

    copyfiles(validation_files, image_dir_src, validation_image_dir_out)
    copyfiles(validation_files, mask_dir_src, validation_mask_dir_out)
