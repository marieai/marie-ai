import glob
import os
import time

import cv2.cv2
import numpy as np
import transformers
from PIL import Image, ImageDraw, ImageFont

from marie.executor import NerExtractionExecutor
from marie.executor.storage.PostgreSQLStorage import PostgreSQLStorage
from marie.logging.profile import TimeContext
from marie.utils.docs import load_image, docs_from_file, array_from_docs
from marie.utils.image_utils import hash_file, hash_bytes
from marie.utils.json import store_json_object
from marie.utils.utils import ensure_exists
from marie import Document, DocumentArray, Executor, Flow, requests, __model_path__

from transformers import (
    LayoutLMv3Processor,
    LayoutLMv3FeatureExtractor,
    LayoutLMv3ForTokenClassification,
    LayoutLMv3TokenizerFast,
)


def create_processor():
    """prepare for the model"""
    # Method:2 Create Layout processor with custom future extractor
    # Max model size is 512, so we will need to handle any documents larger than that
    feature_extractor = LayoutLMv3FeatureExtractor(apply_ocr=False)
    tokenizer = LayoutLMv3TokenizerFast.from_pretrained(
        "microsoft/layoutlmv3-large", only_label_first_subword=False
    )
    processor = LayoutLMv3Processor(
        feature_extractor=feature_extractor, tokenizer=tokenizer
    )

    return processor


def process_file(executor: NerExtractionExecutor, img_path: str):

    with TimeContext(f'### extraction info'):
        filename = img_path.split("/")[-1].replace(".png", "")
        checksum = hash_file(img_path)
        docs = None
        kwa = {"checksum": checksum, "img_path": img_path}
        payload = executor.extract(docs, **kwa)
        print(payload)
        store_json_object(payload, f"/tmp/tensors/json/{filename}.json")

        if False:
            storage = PostgreSQLStorage()
            handler = storage.handler
            print(storage)
            print(handler)
            #
            dd = DocumentArray([Document(id=str(f"lbxid:{filename}"), content=payload)])
            handler.add(dd)

        return payload


def process_dir(executor: NerExtractionExecutor, image_dir: str):
    for idx, img_path in enumerate(glob.glob(os.path.join(image_dir, "*.*"))):
        try:
            process_file(executor, img_path)
        except Exception as e:
            print(e)
            # raise e

def encoding_test():
    processor = create_processor()
    image = Image.open(img_path).convert("RGB")
    words = ["hello", "world"]
    boxes = [[1, 2, 3, 4], [5, 6, 7, 8]]  # make sure to normalize your bounding boxes
    word_labels = [1, 2]
    encoding = processor(
        image, words, boxes=boxes, word_labels=word_labels, return_tensors="pt"
    )
    print(encoding.keys())

    print(encoding["input_ids"])


if __name__ == "__main__":

    # pip install git+https://github.com/huggingface/transformers
    # 4.18.0  -> 4.21.0.dev0 : We should pin it to this version
    print(transformers.__version__)

    img_path = f"/home/greg/dataset/assets-private/corr-indexer/validation/PID_631_7267_0_156693952.png"

    # models_dir = os.path.join(__model_path__, "ner-rms-corr", "checkpoint-best")

    models_dir = (
        "/mnt/data/models/layoutlmv3-large-fullyannotated-dropout/checkpoint-22000"
        "/mnt/data/models/layoutlmv3-large-fullyannotated/checkpoint-best"
    )

    executor = NerExtractionExecutor(models_dir)
    process_dir(executor, "/home/greg/dataset/assets-private/corr-indexer/validation/")
    # process_dir(executor, "/home/gbugaj/tmp/medrx-missing-corr/")

    if True:
        img_path = f"/home/greg/dataset/assets-private/corr-indexer/validation/PID_162_6505_0_156695212.png"
        # img_path = f"/home/greg/dataset/assets-private/corr-indexer/validation/PID_1898_9200_0_156692336.png"

        # img_path = f"/home/greg/dataset/assets-private/corr-indexer/validation_multipage/PID_631_7267_0_156693862.tif"
        # img_path = f"/home/gbugaj/tmp/medrx/PID_1864_9100_0_157637194.tif"
        # img_path = f"/home/greg/dataset/assets-private/corr-indexer/validation_multipage/PID_631_7267_0_156693862.tif"
        # img_path = f"/home/gbugaj/tmp/medrx-missing-corr/PID_1055_7854_0_158147069.tif"

        process_file(executor, img_path)

