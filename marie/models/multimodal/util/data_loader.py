"""
Module to create custom Data Loaders:
    - DocumentClassificationDataset - Loader for documents classification
    - BoundaryDetectionDataset - Loader for documents splitting
"""

import os
from pathlib import Path
from typing import Any, Optional

import torch
from PIL import Image
from docarray import DocList
from torch.utils.data import Dataset
from tqdm import tqdm

from marie.api.docs import BatchableMarieDoc
from marie.components.util import scale_bounding_box

base_dir = Path(__file__).resolve().parent


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


class BoundaryDetectionInferenceDataset(Dataset):

    def __init__(
        self,
        pages_sequences: dict,
        processor: Any,
        cache_dir: Optional[str | Path] = None,
        context_size: int = 2,
    ) -> None:
        """
        Data Loader for Document Splitter Inference
        Args:
            pages_sequences: Sequence of pages for single LbxID. Single or multi docs. At least 2 pages sequence.
            processor: LayoutLMv3Processor (tokenizer + feature extractor)
            cache_dir: Director with encoded pages
            context_size: +- Number of pages around (current_page, next_page) taken into account
        """
        self.processor = processor
        self.context_size = context_size
        self.cache_dir = cache_dir
        if self.cache_dir:
            self._cache_pages_to_disk(pages_sequences)
        self.samples = self._build_samples_include_context(pages_sequences)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample: dict = self.samples[idx]
        images = sample["images"]
        words = sample["words"]
        boxes = sample["boxes"]
        pages: list = []

        if self.cache_dir:
            for image in images:
                image = (base_dir / image).resolve()
                page_cache_path = self._get_cache_path(image)
                page = torch.load(page_cache_path, weights_only=False)
                pages.append(page)
        else:
            for i, image in enumerate(images):
                bboxes = boxes[i]

                width_scale, height_scale = 1000 / image.shape[1], 1000 / image.shape[0]
                boxes_normalized = [
                    scale_bounding_box(box, width_scale, height_scale) for box in bboxes
                ]

                encoding = self.processor(
                    image,
                    words[i],
                    boxes=boxes_normalized,
                    max_length=512,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                page = {
                    "input_ids": encoding["input_ids"].squeeze(0),
                    "attention_mask": encoding["attention_mask"].squeeze(0),
                    "bbox": encoding["bbox"].squeeze(0),
                    "pixel_values": encoding["pixel_values"].squeeze(0),
                }
                pages.append(page)

        return {
            "pages": pages,
            "center_page_idx": sample["center_page_idx"],
            "general_page_idx": sample["general_page_idx"],
        }

    def _build_samples_include_context(self, pages_sequences: dict) -> list:
        """Samples creation: [context_left, current_page, context_right]"""
        samples = []
        for seq_id, seq_data in pages_sequences.items():
            image_paths = seq_data["images"]
            words = seq_data["words"]
            boxes = seq_data["boxes"]

            # context applied
            for i in range(len(image_paths) - 1):  # i: current page
                start = max(0, i - self.context_size)
                end = min(len(image_paths), i + self.context_size + 1)
                sample = {
                    "images": image_paths[start:end],
                    "words": words[start:end],
                    "boxes": boxes[start:end],
                    "center_page_idx": i - start,
                    "general_page_idx": i,
                    "seq_id": seq_id,
                }
                samples.append(sample)
        return samples

    def _cache_pages_to_disk(self, pages_sequences: dict) -> None:
        """Encode and cache each page once, saved as .pt on disk."""
        for seq_id, seq_data in tqdm(
            pages_sequences.items(), desc="Caching pages to disk"
        ):
            image_paths = seq_data["images"]
            words = seq_data["words"]
            boxes = seq_data["boxes"]

            for idx, img_path in enumerate(image_paths):
                img_path = (base_dir / img_path).resolve()
                page_cache_path = self._get_cache_path(img_path)

                if Path(page_cache_path).exists():
                    continue  # skip if already cached

                with Image.open(img_path) as image:
                    image = image.convert("RGB")
                    encoding = self.processor(
                        image,
                        words[idx],
                        boxes=boxes[idx],
                        max_length=512,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                    )

                    encoded_page = {
                        "input_ids": encoding["input_ids"].squeeze(0),
                        "attention_mask": encoding["attention_mask"].squeeze(0),
                        "bbox": encoding["bbox"].squeeze(0),
                        "pixel_values": encoding["pixel_values"].squeeze(0),
                    }
                    torch.save(encoded_page, page_cache_path)

    def _get_cache_path(self, img_path: str) -> str:
        assert isinstance(self.cache_dir, str), "cache_dir should be string"
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        page_cache_path = os.path.join(self.cache_dir, f"{img_name}.pt")
        return page_cache_path
