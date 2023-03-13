import glob
import os
from functools import partial
import argparse
import numpy as np
import torch
from PIL import Image

from marie.overlay.overlay import OverlayProcessor
from marie.utils.docs import load_image
from marie.utils.image_utils import imwrite
from marie.utils.tiff_ops import burst_tiff
from marie.utils.utils import ensure_exists


def split_filename(img_path: str) -> (str, str, str):
    filename = img_path.split("/")[-1]
    prefix = filename.split(".")[0]
    suffix = filename.split(".")[-1]

    return filename, prefix, suffix


def filename_supplier_page(
    filename: str, prefix: str, suffix: str, pagenumber: int
) -> str:
    return f"{prefix}_{pagenumber:05}.{suffix}"


#  Frechet Inception Distance (FID) for Evaluating GANs
#
def clean(src_dir, dst_dir):
    work_dir = ensure_exists("/tmp/form-segmentation")
    overlay_processor = OverlayProcessor(work_dir=work_dir, cuda=True)
    stack_dir = ensure_exists(os.path.join(dst_dir, "stack"))

    ensure_exists(stack_dir)
    src_dir = os.path.expanduser(src_dir)
    dst_dir = os.path.expanduser(dst_dir)

    print(f"Processing : {src_dir}")
    framed = False
    # process each image from the bursts directory
    for _path in sorted(glob.glob(os.path.join(src_dir, "*.tif"))):
        try:
            print(f"Processing : {_path}")
            filename = _path.split("/")[-1]
            docId = filename.split("/")[-1].split(".")[0]
            print(f"DocumentId : {docId}")
            if os.path.exists(os.path.join(dst_dir, filename)):
                print(f"Image exists : {docId}")

            src_img_path = os.path.join(src_dir, filename)

            if framed:
                loaded, frames = load_image(src_img_path)
                real, fake, blended = overlay_processor.segment_frame(docId, frames[0])
            else:
                real, fake, blended = overlay_processor.segment(docId, src_img_path)

            # debug image
            if True:
                stacked = np.hstack((real, fake, blended))
                save_path = os.path.join(stack_dir, f"{docId}.png")
                imwrite(save_path, stacked)

            save_path = os.path.join(
                dst_dir, f"{docId}.tif"
            )  # This will have the .tif extension
            imwrite(save_path, blended)
            print(f"Saving  document : {save_path}")

        except Exception as ident:
            # raise ident
            print(ident)


def burst(src_dir, dst_dir):
    src_dir = os.path.expanduser(src_dir)
    dst_dir = os.path.expanduser(dst_dir)

    burst_dir = ensure_exists(os.path.join(dst_dir))
    # process each image from the bursts directory
    for _path in sorted(glob.glob(os.path.join(src_dir, "*.tif"))):
        try:
            ref_id = _path.split("/")[-1]
            filename, prefix, suffix = split_filename(ref_id)
            filename_generator = partial(
                filename_supplier_page, filename, prefix, suffix
            )
            burst_tiff(_path, burst_dir, filename_generator=filename_generator)
        except Exception as ident:
            raise ident


def ssim_score(real_img, gen_img):
    from skimage.metrics import structural_similarity as ssim

    img = np.array(real_img)
    gen = np.array(gen_img)
    ssim_const = ssim(img, gen, multichannel=True, data_range=gen.max() - gen.min())
    return ssim_const


def score(src_dir, dst_dir):
    src_dir = os.path.expanduser(src_dir)
    dst_dir = os.path.expanduser(dst_dir)

    for _path in sorted(glob.glob(os.path.join(src_dir, "*.tif"))):
        try:
            filename, prefix, suffix = split_filename(_path)
            src_img_path = os.path.join(src_dir, filename)
            dst_img_path = os.path.join(dst_dir, filename)
            src_img = Image.open(src_img_path).convert('RGB')
            dst_img = Image.open(dst_img_path).convert('RGB')

            ssim = ssim_score(src_img, dst_img)
            print(f"SSIM : {ssim}  > {filename}")
        except Exception as ident:
            raise ident


def main(args: argparse.Namespace):
    if args.action == "clean":
        clean(args.src_dir, args.dst_dir)
    elif args.action == "score":
        score(args.src_dir, args.dst_dir)
    elif args.action == "burst":
        burst(args.src_dir, args.dst_dir)
    else:
        raise ValueError(f"Unknown action {args.action}")


if __name__ == "__main__":

    # os.environ["OMP_NUM_THREADS"] = str(multiprocessing.cpu_count())
    os.environ["OMP_NUM_THREADS"] = str(1)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--action", type=str, default="burst", help="burst, clean, score"
    )
    parser.add_argument("--src_dir", type=str, default="~/tmp/marie-cleaner/")
    parser.add_argument("--output_dir", type=str, default="~/tmp/marie-cleaner/burst")

    args = parser.parse_args()

    # args.action = "burst"
    # args.src_dir = "~/tmp/marie-cleaner/to-clean-001"
    # args.dst_dir = "~/tmp/marie-cleaner/to-clean-001/burst"

    args.action = "clean"
    args.src_dir = "~/tmp/marie-cleaner/to-clean-001/burst"
    args.dst_dir = "~/tmp/marie-cleaner/to-clean-001/clean"

    args.action = "score"
    args.src_dir = "~/tmp/marie-cleaner/to-clean-001/real"
    args.dst_dir = "~/tmp/marie-cleaner/to-clean-001/clean"

    main(args)
