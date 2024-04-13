import os.path
import random

import numpy as np
import torch

from marie.components.document_registration.unilm_dit import (
    DocumentBoundaryPrediction,
    UnilmDocumentBoundaryRegistration,
)
from marie.utils.docs import docs_from_file, frames_from_docs, frames_from_file
from marie.utils.tiff_ops import (
    convert_group4,
    merge_tiff,
    merge_tiff_frames,
    save_frame_as_tiff_g4,
)
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
    # "/home/greg/tmp/demo/PID_114_6529_0_177272981_page_0002.png"
    # "/home/greg/tmp/demo/PID_114_6529_0_177104327_page_0004.png"
    # "/home/greg/tmp/demo/PID_114_6416_0_177360024_page_0002.png"
    # "/home/greg/tmp/demo/PID_114_6529_0_177104346_page_0003.png"
    # "/home/greg/tmp/demo/PID_114_6416_0_177272667_page_0002.png"
    # "/home/greg/tmp/demo/PID_114_6416_0_177360024_page_0001.png"
    # "/home/gbugaj/tmp/demo/158851107_3.png"
    # "/home/gbugaj/dev/workflow/mbx-grapnel/mbx-grapnel-engine/src/test/resources/test-deck/Integration_201295814/PID_1707_8728_0_201218706.tif"

    filepath = "/home/gbugaj/datasets/private/scan-of-scan/04-08-2024/PID_402_8220_0_200802683.tif"
    filepath = "/home/gbugaj/analysis/grapnel/document-boundary/200942298_1712966490091/PID_3736_11058_0_200942298.tif"  # => 201458362
    # filepath = "/home/gbugaj/analysis/grapnel/document-boundary/201324503_1712956863847/PID_5872_13170_0_201324503.tif" # => 201493262

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
            aligned = boundary.aligned_image
            frame = aligned
            converted_frames.append(aligned)
        else:
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

    # tiff_frames = frames_from_file(clean_filename)
    # merge_tiff_frames(tiff_frames, os.path.join(basedir, f"{basename}_resave_{registration_method}.tif"))


if __name__ == "__main__":
    check_boundry_registration()
