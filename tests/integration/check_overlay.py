import glob
import os
import numpy as np
import torch

from marie.overlay.overlay import OverlayProcessor
from marie.utils.docs import load_image
from marie.utils.image_utils import imwrite
from marie.utils.tiff_ops import burst_tiff
from marie.utils.utils import ensure_exists

# Example script that shows how to clean document
if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    work_dir = ensure_exists("/tmp/form-segmentation")
    root_dir = "~/tmp/marie-cleaner/to-clean-001"

    root_dir = os.path.expanduser(root_dir)
    burst_dir = ensure_exists(os.path.join(root_dir, "burst"))
    stack_dir = ensure_exists(os.path.join(root_dir, "stack"))
    clean_dir = ensure_exists(os.path.join(root_dir, "clean"))

    img_path = os.path.join(root_dir, "159015281_2.png")
    burst_tiff(img_path, burst_dir, silence_errors=True)

    # os.environ["OMP_NUM_THREADS"] = str(multiprocessing.cpu_count())
    os.environ["OMP_NUM_THREADS"] = str(1)

    overlay_processor = OverlayProcessor(work_dir=work_dir, cuda=True)

    framed = False
    NITER = 1

    # process each image from the bursts directory
    for _path in sorted(glob.glob(os.path.join(burst_dir, "*.*"))):
        try:
            for i in range(0, NITER):
                print(f"Processing : {_path}")
                filename = _path.split("/")[-1]
                docId = filename.split("/")[-1].split(".")[0]
                print(f"DocumentId : {docId}")

                if os.path.exists(os.path.join(clean_dir, filename)):
                    print(f"Image exists : {docId}")
                    # continue

                src_img_path = os.path.join(burst_dir, filename)

                if framed:
                    loaded, frames = load_image(src_img_path)
                    real, fake, blended = overlay_processor.segment_frame(
                        docId, frames[0]
                    )
                else:
                    real, fake, blended = overlay_processor.segment(docId, src_img_path)

                # debug image
                if True:
                    stacked = np.hstack((real, fake, blended))
                    save_path = os.path.join(stack_dir, f"{i}_{docId}.png")
                    imwrite(save_path, stacked)

                save_path = os.path.join(
                    clean_dir, f"{docId}.tif"
                )  # This will have the .tif extension
                imwrite(save_path, blended)
                print(f"Saving  document : {save_path}")

        except Exception as ident:
            # raise ident
            print(ident)
