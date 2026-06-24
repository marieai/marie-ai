from typing import Any

import torch
from docarray import DocList
from torch.utils.data import Dataset

from marie.api.docs import BatchableMarieDoc
from marie.components.util import scale_bounding_box


class DocumentClassificationInferenceDataset(Dataset):

    def __init__(self, documents: DocList[BatchableMarieDoc], processor: Any) -> None:
        """
        Data Loader for Document Classification
        Args:
            documents: documents database processed via dataset_preparation.py
            processor: LayoutLMv3Processor (tokenizer + feature extractor)
        """
        self.documents = documents
        # self.keys = list(self.documents.keys())  # doc ids list
        self.processor = processor

    def __len__(self) -> int:
        # return len(self.documents)
        return 1  # one sample = one multi-page document #todo

    def __getitem__(self, idx: int) -> dict:
        # doc_id = self.keys[idx]
        # document_idx = self.documents[doc_id]
        # document_idx = self.documents[idx]

        # pages_labels = document_idx["page_enc_labels"]

        pages = []
        for page in self.documents:  # image_path = (base_dir / image_path).resolve()
            # with Image.open(image_path) as image:
            #     image = image.convert("RGB")
            # for doc in self.documents.values():
            image = page.tensor
            words = page.words
            bboxes = page.boxes

            width_scale, height_scale = 1000 / image.shape[1], 1000 / image.shape[0]
            boxes_normalized = [
                scale_bounding_box(box, width_scale, height_scale) for box in bboxes
            ]

            encoding = self.processor(
                image,
                words,
                boxes=boxes_normalized,
                max_length=512,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            pages.append(
                {
                    "input_ids": encoding["input_ids"].squeeze(0),
                    "attention_mask": encoding["attention_mask"].squeeze(0),
                    "bbox": encoding["bbox"].squeeze(0),
                    "pixel_values": encoding["pixel_values"].squeeze(0),
                    "page_label": torch.tensor(
                        0
                    ),  # dummy label for inference # MPC classifications should go here
                }
            )

        return {
            "pages": pages,
        }
