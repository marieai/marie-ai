import glob
import os

import cv2

# from marie.executor.ner import NerExtractionExecutor
from marie.utils.docs import docs_from_file, frames_from_docs, frames_from_file
from marie.utils.image_utils import hash_frames_fast
from marie.utils.tiff_ops import merge_tiff_frames, save_frame_as_tiff_g4
from marie.utils.utils import ensure_exists

executor = None


def process_image(img_path):
    # get name from filenama
    name = os.path.basename(img_path)
    name = os.path.splitext(name)[0]

    docs = docs_from_file(img_path)
    arr = frames_from_docs(docs)
    checksum = hash_frames_fast(arr)
    kwa = {}
    results = executor.extract(docs, **kwa)
    print(results)
    # store_json_object(results, f"/tmp/pdf_2_tif/json/{name}.json")
    return results


def process_dir_ner(image_dir: str):
    for idx, img_path in enumerate(glob.glob(os.path.join(image_dir, "*.tif"))):
        try:
            print(img_path)
            process_image(img_path)
        except Exception as e:
            print(e)
            # raise e


def process_dir_pdf(image_dir: str):
    from marie.utils.resize_image import resize_image

    for idx, img_path in enumerate(glob.glob(os.path.join(image_dir, "*.pdf"))):
        try:
            print(img_path)
            # get name from filenama
            name = os.path.basename(img_path)
            name = os.path.splitext(name)[0]
            print(name)
            # read fraes from pdf
            frames = frames_from_file(img_path)
            print(len(frames))
            merge_tiff_frames(
                frames, "~/tmp/analysis/resized-pdf/gen/{}.tif".format(name)
            )
        except Exception as e:
            print(e)
            # raise e


def process_dir_tiff(image_dir: str):
    # Sort .tif files numerically based on filename
    def numeric_key(file_path):
        base = os.path.basename(file_path)
        name = os.path.splitext(base)[0]
        # Extract trailing digits like 001, 002, etc.
        digits = ''.join(filter(str.isdigit, name))
        return int(digits) if digits.isdigit() else 0

    tiff_files = sorted(glob.glob(os.path.join(image_dir, "*.tif")), key=numeric_key)
    tiff_files = sorted(
        glob.glob(os.path.join("~/tmp/analysis/resized-pdf/g4", "*.tif")),
        key=numeric_key,
    )
    frames = []

    output_dir = os.path.join("~/tmp/analysis/resized-pdf/g4")
    ensure_exists(output_dir)
    for idx, img_path in enumerate(tiff_files):
        try:
            print(f"Reading TIFF: {img_path}")
            name = os.path.basename(img_path)
            name = os.path.splitext(name)[0]
            print(f"Base name: {name}")

            frame = frames_from_file(img_path)[0]  # Read the first frame

            # save real cleaned image
            # save_path = os.path.join(output_dir, f"{idx}.tif")
            # resized = resize_with_aspect_ratio(frame, width=2550, height=None)[1]
            # save_frame_as_tiff_g4(resized, save_path)

            print(f"{len(frames)} frames read from {img_path}")
            frames.append(frame)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            raise e

        print(len(frames))
        merge_tiff_frames(frames, "~/tmp/analysis/resized-pdf/converted-g4.tif")


def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    r = 1
    if width is None and height is None:
        return r, image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return r, cv2.resize(image, dim, interpolation=inter)


if __name__ == "__main__":
    # ensure_exists("/tmp/pdf_2_tif")
    # can also use pdftoppm to convert pdf to tiff as the PDFU can fail on some PDFs
    # https://stackoverflow.com/questions/75500/best-way-to-convert-pdf-files-to-tiff-files
    # pdftoppm bw.tiff.pdf pages -tiff

    # convert input.tif -units PixelsPerInch -density 300 output.tif
    # process_dir_pdf("~/tmp/analysis/resized-pdf")
    process_dir_tiff("~/tmp/analysis/resized-pdf/gen")
