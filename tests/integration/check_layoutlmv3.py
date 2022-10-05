import glob
import os
import time

import numpy as np
import transformers
from PIL import Image, ImageDraw, ImageFont


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


def encoding_test(img_path):
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
