import logging
import os

import torch
from PIL import Image

from torch.utils.data import Dataset
import io
import json
from typing import Any, List

from pandas import DataFrame
from transformers import LayoutLMv3Processor


def scale_bounding_box(
    box: List[int], width_scale: float = 1.0, height_scale: float = 1.0
) -> List[int]:
    return [
        int(box[0] * width_scale),
        int(box[1] * height_scale),
        int(box[2] * width_scale),
        int(box[3] * height_scale),
    ]


def load_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)

    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    return image, (w, h)


def normalize_bbox(bbox, size):
    return [
        int(1000 * bbox[0] / size[0]),
        int(1000 * bbox[1] / size[1]),
        int(1000 * bbox[2] / size[0]),
        int(1000 * bbox[3] / size[1]),
    ]


logger = logging.getLogger(__name__)


class DocumentClassificationDataset(Dataset):
    def __init__(
        self,
        scale_bounding_boxes: bool,
        pd_data: DataFrame,
        classes: List[str],
        processor: LayoutLMv3Processor,
    ):
        super().__init__()
        self.image_paths = pd_data["image_path"]
        self.classes = classes  # sorted(pd_data['label'].unique())
        self.pd_data = pd_data
        self.processor = processor
        self.scale_bounding_boxes = scale_bounding_boxes

        self.idx2label = {idx: label for idx, label in enumerate(self.classes)}
        self.label2idx = {label: idx for idx, label in enumerate(self.classes)}

        # print(f"Classes: {self.classes}")
        # print(f"idx2label: {self.idx2label}")
        # print(f"label2idx: {self.label2idx}")
        # print("-------------------")

        # validate that all images have an annotation file
        image_paths_valid = []
        for image_path in self.image_paths:
            annotation_path = image_path.replace("images", "annotations")
            last = annotation_path.rfind(".")
            annotation_path = annotation_path[:last] + ".json"

            if not os.path.exists(annotation_path):
                print(
                    f"Missing annotation file for {annotation_path} for image {image_path}"
                )

            else:
                image_paths_valid.append(image_path)

        self.image_paths = image_paths_valid

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        # print(f"Loading item {item} of {len(self.image_paths)}")
        image_path = self.image_paths[item]
        annotation_path = image_path.replace("images", "annotations")
        last = annotation_path.rfind(".")
        annotation_path = annotation_path[:last] + ".json"

        if not os.path.exists(annotation_path):
            raise ValueError(
                f"Missing annotation file for {annotation_path} for image {image_path}"
            )

        boxes = []
        words = []

        with io.open(annotation_path, "r", encoding="utf-8") as json_file:
            try:
                ocr_results = json.load(json_file)
            except Exception as e:
                print(f"Error loading annotation file {annotation_path}: {e}")
                raise e

        self.scale_bounding_box = True

        with Image.open(image_path).convert("RGB") as image:
            size = image.size
            width, height = image.size
            width_scale = 1000 / width
            height_scale = 1000 / height

            if len(ocr_results) != 1:
                raise ValueError(
                    f"Expected 1 page in annotation file {annotation_path} for image {image_path}, "
                )
            for w in ocr_results[0]["words"]:
                if self.scale_bounding_box:
                    bbox = normalize_bbox(w["box"], size)
                    bbox = scale_bounding_box(w["box"], width_scale, height_scale)
                else:
                    bbox = w["box"]

                # The `bbox` coordinate values should be within 0-1000 range.
                assert all(
                    0 <= v <= 1000 for v in bbox
                ), f"Invalid bbox coordinates {bbox} for image {image_path}"

                boxes.append(bbox)
                words.append(w["text"])

            assert len(boxes) == len(words)
            encoding = self.processor(
                image,
                words,
                boxes=boxes,
                max_length=512,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            # print(
            #     f"""
            # input_ids:  {list(encoding["input_ids"].squeeze().shape)}
            # word boxes: {list(encoding["bbox"].squeeze().shape)}
            # image data: {list(encoding["pixel_values"].squeeze().shape)}
            # image size: {image.size}
            # """
            # )

            # image_data = encoding["pixel_values"][0]
            # transform = T.ToPILImage()
            # transform(image_data)
            #
            label = self.label2idx[
                self.pd_data.loc[self.pd_data["image_path"] == image_path][
                    "label"
                ].values[0]
            ]

            return dict(
                input_ids=encoding["input_ids"].flatten(),
                attention_mask=encoding["attention_mask"].flatten(),
                bbox=encoding["bbox"].flatten(end_dim=1),
                pixel_values=encoding["pixel_values"].flatten(end_dim=1),
                labels=torch.tensor(label, dtype=torch.long),
            )
