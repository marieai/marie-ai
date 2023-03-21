import glob
import multiprocessing as mp
import os
import shutil
import tempfile
import uuid
from concurrent.futures.thread import ThreadPoolExecutor
from datetime import datetime
from typing import Callable, Optional, List

import cv2
import numpy as np

# http://148.216.108.129/vython38/lib/python3.8/site-packages/willow/plugins/wand.py
from tifffile import TiffWriter

from marie.utils.docs import frames_from_file, get_document_type


# https://github.com/joeatwork/python-lzw
# exiftool PID_576_7188_0_150300431.tif
# sudo apt-get install libtiff5-dev
# https://stackoverflow.com/questions/64403525/failed-to-find-tiff-header-file-tifflib-error-on-python-ubuntu-20-04
# https://www.tecmint.com/install-imagemagick-on-debian-ubuntu/


def convert_group4(src_path, dst_path):
    """
    Error:
    cache resources exhausted Imagemagick
    https://stackoverflow.com/questions/31407010/cache-resources-exhausted-imagemagick
    https://github.com/ImageMagick/ImageMagick/issues/396

    identify -verbose /path/to/img.tiff | grep photometric
    """
    from ctypes import c_char_p, c_void_p

    from wand.api import library
    from wand.image import Image

    # Tell python about the MagickSetOption method
    library.MagickSetOption.argtypes = [
        c_void_p,  # MagickWand * wand
        c_char_p,  # const char * option
        c_char_p,
    ]  # const char * value

    with Image(filename=src_path) as image:
        # -define quantum:polarity=min-is-white
        library.MagickSetOption(
            image.wand,
            "quantum:polarity".encode("utf-8"),
            "min-is-white".encode("utf-8"),  # MagickWand  # option
        )  # value

        library.MagickSetOption(
            image.wand,
            "tiff:rows-per-strip".encode("utf-8"),
            "1".encode("utf-8"),  # MagickWand  # option
        )  # value

        library.MagickSetImageCompression(image.wand, 8)
        library.MagickSetImageDepth(image.wand, 1)

        # "-compress".encode('utf-8'),  # option
        # "Group4".encode('utf-8'))  # value
        # Write min-is-white image

        image.compression = "group4"
        image.resolution = (300, 300)
        image.save(filename=dst_path)


def __process_burst(frame, bitonal, generated_name, dest_dir, index, tmp_dir):
    try:
        output_path_tmp = os.path.join(tmp_dir, generated_name)
        output_path = os.path.join(dest_dir, generated_name)
        print(f"Bursting page# {index} : {generated_name} > {output_path}")
        # check if root directory exists
        photometric = "rbg"
        if bitonal:
            photometric = "minisblack"
            # at this time we expect TIFF frame to be already bitonal
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.threshold(
                    gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
                )[1]

        # TODO : Replace this with image magic methods so we can do this  in one step
        with TiffWriter(output_path_tmp) as tif_writer:
            tif_writer.write(
                frame,
                photometric=photometric,
                description=generated_name,
                metadata=None,
            )
        convert_group4(output_path_tmp, output_path)
    except Exception as ident:
        raise ident


def burst_tiff_frames(
    frames: List[np.ndarray],
    dest_dir,
    bitonal=True,
    sequential=True,
    filename_generator: Optional[Callable] = None,
) -> None:
    """
    Burst multipage tiff into individual frames and save them to output directory
    Ref : https://stackoverflow.com/questions/9627652/split-multi-page-tiff-with-python

    :param frames: Source image
    :param dest_dir: Destination directory
    :param bitonal: Should image be converted to bitonal image
    :param sequential: Should the document be process sequentially or in multithreaded fashion
    :param filename_generator: Function that generates filename for each frame
    """

    filename_generator = filename_generator or (
        lambda pagenumber: f"{pagenumber:05}.tif"
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        if sequential:
            print("created temporary directory", tmp_dir)
            for i, frame in enumerate(frames):
                index = i + 1
                generated_name = filename_generator(pagenumber=index)
                __process_burst(
                    frame,
                    bitonal,
                    generated_name,
                    dest_dir,
                    index,
                    tmp_dir,
                )
        else:
            with ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
                print("created temporary directory", tmp_dir)
                for i, frame in enumerate(frames):
                    index = i + 1
                    generated_name = filename_generator(pagenumber=index)
                    executor.submit(
                        __process_burst,
                        frame,
                        bitonal,
                        generated_name,
                        dest_dir,
                        index,
                        tmp_dir,
                    )


def burst_tiff(
    src_img_path,
    dest_dir,
    bitonal=True,
    sequential=True,
    filename_generator: Optional[Callable] = None,
    silence_errors=False,
):
    """
    Burst multipage tiff into individual frames and save them to output directory

    :param src_img_path: Source image
    :param dest_dir: Destination directory
    :param bitonal: Should image be converted to bitonal image
    :param sequential: Should the document be process sequentially or in multithreaded fashion
    :param filename_generator: Function that generates filename for each frame
    :param silence_errors: If True, errors will be silenced and file copy will be performed.
    """

    image_type = get_document_type(src_img_path)
    if image_type != "tiff":
        if silence_errors:
            print(f"Expected tiff file, got {image_type}, skipping...")
            # copy file to destination directory
            shutil.copy(src_img_path, dest_dir)
            return
        else:
            raise ValueError(f"Expected tiff file, got {image_type}")

    frames = frames_from_file(src_img_path)
    burst_tiff_frames(frames, dest_dir, bitonal, sequential, filename_generator)


def merge_tiff(src_dir, dst_img_path, sort_key):
    """Merge individual tiff frames into a multipage tiff"""
    from wand.image import Image

    with Image() as composite:
        for _path in sorted(glob.glob(os.path.join(src_dir, "*.*")), key=sort_key):
            try:
                if False:
                    curren_time = datetime.now().strftime("%H:%M:%S.%f")
                    print(f"Merging document :{curren_time} {_path}")
                with Image(filename=_path) as src_img:
                    frame = src_img.image_get()
                    composite.image_add(frame)
            except Exception as ident:
                raise ident

        composite.compression = "group4"
        composite.resolution = (300, 300)
        composite.save(filename=dst_img_path)


def merge_tiff_frames(
    frames,
    dst_img_path,
):
    """Merge individual tiff frames into a multipage tiff"""
    from wand.image import Image

    print(f"Creating multipage tiff : {dst_img_path}")
    with Image() as composite:
        try:
            for src_img in frames:
                src_img = Image.from_array(src_img)
                frame = src_img.image_get()
                composite.image_add(frame)
        except Exception as ident:
            raise ident

        composite.compression = "group4"
        composite.resolution = (300, 300)
        composite.save(filename=dst_img_path)


def save_frame_as_tiff_g4(frame, output_filename):
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            generated_name = f"{uuid.uuid4()}.tif"
            output_path_tmp = os.path.join(tmp_dir, generated_name)

            print(
                f"save_frame_as_tiff_g4  page# > {output_filename} > : {output_path_tmp}"
            )
            # check if root directory exists
            bitonal = True

            photometric = "rbg"
            if bitonal:
                photometric = "minisblack"
                # at this time we expect TIFF frame to be already bitonal
                if len(frame.shape) == 3:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frame = cv2.threshold(
                        gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
                    )[1]

            # TODO : Replace this with image magic methods so we can do this  in one step
            with TiffWriter(output_path_tmp) as tif_writer:
                tif_writer.write(
                    frame,
                    photometric=photometric,
                    description=generated_name,
                    metadata=None,
                )

            convert_group4(output_path_tmp, output_filename)
    except Exception as ident:
        raise ident
