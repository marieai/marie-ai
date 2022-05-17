import glob
import os
import tempfile
import multiprocessing as mp
from concurrent.futures.thread import ThreadPoolExecutor

import cv2

# http://148.216.108.129/vython38/lib/python3.8/site-packages/willow/plugins/wand.py
from tifffile import TiffWriter


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
    from ctypes import c_void_p, c_char_p
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
            image.wand, "quantum:polarity".encode("utf-8"), "min-is-white".encode("utf-8")  # MagickWand  # option
        )  # value

        library.MagickSetOption(
            image.wand, "tiff:rows-per-strip".encode("utf-8"), "1".encode("utf-8")  # MagickWand  # option
        )  # value

        library.MagickSetImageCompression(image.wand, 8)
        library.MagickSetImageDepth(image.wand, 1)

        # "-compress".encode('utf-8'),  # option
        # "Group4".encode('utf-8'))  # value
        # Write min-is-white image

        image.compression = "group4"
        image.resolution = (300, 300)
        image.save(filename=dst_path)


def __process_burst(frame, name, generated_name, dest_dir, index, tmpdirname):
    try:
        output_path_tmp = os.path.join(tmpdirname, generated_name)
        output_path = os.path.join(dest_dir, generated_name)
        print(f"Bursting page# {index} : {name} > {generated_name} > {output_path}")
        # TODO : Replace this with image magic methods so we can do this  in one step
        with TiffWriter(output_path_tmp) as tif_writer:
            tif_writer.write(frame, photometric="minisblack", description=generated_name, metadata=None)
        convert_group4(output_path_tmp, output_path)
    except Exception as ident:
        raise ident


def burst_tiff(src_img_path, dest_dir):
    """Burst multipage tiff into individual frames and save them to output directory"""
    ret, frames = cv2.imreadmulti(src_img_path, [], cv2.IMREAD_ANYCOLOR)
    name = src_img_path.split("/")[-1].split(".")[0]

    # fireup new threads for processing
    with tempfile.TemporaryDirectory() as tmp_dir:
        with ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
            print("created temporary directory", tmp_dir)
            for i, frame in enumerate(frames):
                index = i + 1
                generated_name = f"{name}_page_{index:04}.tif"
                executor.submit(__process_burst, frame, name, generated_name, dest_dir, index, tmp_dir)


def merge_tiff(src_dir, dst_img_path, sort_key):
    """Merge individual tiff frames into a multipage tiff"""
    from wand.image import Image

    print(f"Creating multipage tiff : {dst_img_path}")
    with Image() as composite:
        for _path in sorted(glob.glob(os.path.join(src_dir, "*.tif*")), key=sort_key):
            try:
                print(f"Merging document : {_path}")
                filename = _path.split("/")[-1].split(".")[0]
                with Image(filename=_path) as src_img:
                    frame = src_img.image_get()
                    composite.image_add(frame)
            except Exception as ident:
                raise ident

        composite.compression = "group4"
        composite.resolution = (300, 300)
        composite.save(filename=dst_img_path)
