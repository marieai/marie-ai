import glob
import os

import numpy as np

from marie.overlay.overlay import OverlayProcessor
from marie.utils.image_utils import imwrite
from marie.utils.tiff_ops import burst_tiff
from marie.utils.utils import ensure_exists

# Example script that shows how to clean document
if __name__ == "__main__":

    img_path = "/home/gbugaj/tmp/marie-cleaner/161970410/PID_1956_9362_0_161970410.tif"
    burst_dir = "/home/gbugaj/tmp/marie-cleaner/161970410/burst"
    # burst_tiff(img_path, burst_dir)

    # os.exit()
    work_dir = ensure_exists("/tmp/form-segmentation")
    # this is the image working directory
    root_dir = "/home/greg/dataset/medprov/PID/150300431/"
    root_dir = "/home/gbugaj/tmp/marie-cleaner/to-clean-001"
    root_dir = "/home/gbugaj/tmp/marie-cleaner/161970410"

    burst_dir = ensure_exists(os.path.join(root_dir, "burst"))
    stack_dir = ensure_exists(os.path.join(root_dir, "stack"))
    clean_dir = ensure_exists(os.path.join(root_dir, "clean"))

    overlay_processor = OverlayProcessor(work_dir=work_dir)

    # process each image from the bursts directory
    for _path in sorted(glob.glob(os.path.join(burst_dir, "*.tif"))):
        try:
            filename = _path.split("/")[-1]
            docId = filename.split("/")[-1].split(".")[0]
            # docId = _path.split("/")[-1]
            print(f"DocumentId : {docId}")

            if os.path.exists(os.path.join(clean_dir, filename)):
                print(f"Image exists : {docId}")
                continue

            src_img_path = os.path.join(burst_dir, filename)
            real, fake, blended = overlay_processor.segment(docId, src_img_path)

            # debug image
            if True:
                stacked = np.hstack((real, fake, blended))
                save_path = os.path.join(stack_dir, f"{docId}.png")
                imwrite(save_path, stacked)

            save_path = os.path.join(
                clean_dir, f"{docId}.tif"
            )  # This will have the .tif extension
            imwrite(save_path, blended)
            print(f"Saving  document : {save_path}")

        except Exception as ident:
            # raise ident
            print(ident)
