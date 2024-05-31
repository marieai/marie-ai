import os.path
import random

import numpy as np
import torch

from marie.components.document_registration.unilm_dit import (
    DocumentBoundaryPrediction,
    UnilmDocumentBoundaryRegistration,
)
from marie.utils.docs import docs_from_file, frames_from_docs
from marie.utils.tiff_ops import merge_tiff, save_frame_as_tiff_g4
from marie.utils.utils import ensure_exists


def setup_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def check_boundry_registration():
    setup_seed(42)
    processor = UnilmDocumentBoundaryRegistration(
        model_name_or_path="../../model_zoo/unilm/dit/object_detection/document_boundary",
        use_gpu=True,
    )
    filepath = "~/PID_3736_11058_0_200942298.tif"
    filepath = "~/tmp/analysis/document-boundary/samples/204169581/PID_3585_10907_0_204169581.tif"
    filepath = "~/tmp/analysis/document-boundary/samples/203822066/PID_3585_10907_0_203822066.tif"
    filepath = "~/tmp/analysis/document-boundary/samples/204446542/PID_3585_10907_0_204446542.tif"

    basename = filepath.split("/")[-1].split(".")[0]
    documents = docs_from_file(filepath)

    print("Documents: ", len(documents))
    registration_method = "fit_to_page"  # fit_to_page, absolute
    results = processor.run(documents, registration_method)

    frames = frames_from_docs(documents)
    converted_frames = []

    output_dir = os.path.expanduser(f"~/tmp/grapnel/aligned/workdir/{basename}")
    ensure_exists(output_dir)

    for i, (frame, result) in enumerate(zip(frames, results)):
        boundary: DocumentBoundaryPrediction = result.tags["document_boundary"]
        if boundary.detected:
            frame = boundary.aligned_image
        converted_frames.append(frame)
        save_path = os.path.join(output_dir, f"{i}.tif")
        save_frame_as_tiff_g4(frame, save_path)

    print("Converted frames: ", len(converted_frames))
    basedir = os.path.expanduser("~/tmp/grapnel/aligned")
    # merge_tiff_frames(converted_frames, os.path.join(basedir, f"{basename}_{registration_method}.tif"))
    clean_filename = os.path.join(basedir, f"{basename}_{registration_method}.tif")
    merge_tiff(
        output_dir,
        clean_filename,
        sort_key=lambda name: int(name.split("/")[-1].split(".")[0]),
    )


if __name__ == "__main__":
    check_boundry_registration()
