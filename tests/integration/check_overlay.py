import glob
import logging
import multiprocessing
import os
import platform

import numpy as np
import psutil
import torch

from marie.logging_core.profile import TimeContext
from marie.models.utils import enable_tf32, openmp_setup
from marie.overlay.overlay import OverlayProcessor
from marie.utils.docs import load_image
from marie.utils.image_utils import imwrite
from marie.utils.process import load_omp_library
from marie.utils.tiff_ops import burst_tiff
from marie.utils.utils import ensure_exists

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

    img_path = os.path.join(root_dir, "PID_576_7188_0_150300411_4.tif")
    burst_tiff(img_path, burst_dir, silence_errors=True)

    os.environ["OMP_NUM_THREADS"] = str(multiprocessing.cpu_count())
    # os.environ["OMP_NUM_THREADS"] = str(1)

    overlay_processor = OverlayProcessor(work_dir=work_dir, cuda=True)

    framed = True
    NITER = 10

    k = load_omp_library()

    jemallocpath = "/usr/lib/%s-linux-gnu/libjemalloc.so.2" % (platform.machine(),)

    if os.path.isfile(jemallocpath):
        os.environ["LD_PRELOAD"] = jemallocpath
    else:
        logger.info("Could not find %s, will not use" % (jemallocpath,))

    # Optimizations for PyTorch
    core_count = psutil.cpu_count(logical=False)

    logger.info(f"Setting up TF32")
    enable_tf32()

    logger.info(f"Setting up OpenMP with {core_count} threads")
    openmp_setup(core_count)
    torch.set_num_threads(core_count)

    # Enable oneDNN Graph
    torch.jit.enable_onednn_fusion(True)

    # process each image from the bursts directory
    for _path in sorted(glob.glob(os.path.join(burst_dir, "*.*"))):
        try:
            for i in range(0, NITER):
                with TimeContext(f"Eval # {i} "):
                    print(f"\nProcessing : {_path}")
                    filename = _path.split("/")[-1]
                    docId = filename.split("/")[-1].split(".")[0]
                    print(f"DocumentId : {docId}")

                    if os.path.exists(os.path.join(clean_dir, filename)):
                        print(f"Image exists : {docId}")
                        # continue

                    src_img_path = os.path.join(burst_dir, filename)
                    loaded, frames = load_image(src_img_path)
                    frame = frames[0]

                    if framed:
                        real, fake, blended = overlay_processor.segment_frame(
                            docId, frame
                        )
                    else:
                        real, fake, blended = overlay_processor.segment(
                            docId, src_img_path
                        )

                    assert real is not None
                    assert fake is not None
                    assert blended is not None

                    assert frame.shape == fake.shape
                    assert frame.shape == blended.shape

                    assert real.shape == fake.shape
                    assert real.shape == blended.shape

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
            raise ident
