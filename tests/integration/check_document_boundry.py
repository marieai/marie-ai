import multiprocessing
import os
import random

import cv2
import numpy as np
import psutil
import torch
from PIL import Image

from marie.components.document_registration.unilm_dit import (
    UnilmDocumentBoundaryRegistration,
)
from marie.components.template_matching import (
    BaseTemplateMatcher,
    CompositeTemplateMatcher,
    MetaTemplateMatcher,
    VQNNFTemplateMatcher,
)
from marie.logging.profile import TimeContext
from marie.utils.docs import docs_from_file, docs_from_image, frames_from_docs
from marie.utils.json import load_json_file
from marie.utils.resize_image import resize_image, resize_image_progressive


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
    documents = docs_from_file(
        # "/home/greg/tmp/demo/PID_114_6529_0_177272981_page_0002.png"
        # "/home/greg/tmp/demo/PID_114_6529_0_177104327_page_0004.png"
        # "/home/greg/tmp/demo/PID_114_6416_0_177360024_page_0002.png"
        # "/home/greg/tmp/demo/PID_114_6529_0_177104346_page_0003.png"
        # "/home/greg/tmp/demo/PID_114_6416_0_177272667_page_0002.png"
        "/home/greg/tmp/demo/PID_114_6416_0_177360024_page_0001.png"
    )
    print("Documents: ", len(documents))
    results = processor.run(documents)


if __name__ == "__main__":
    check_boundry_registration()
