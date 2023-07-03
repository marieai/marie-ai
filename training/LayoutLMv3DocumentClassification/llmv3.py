import os

import numpy as np
import pandas as pd

from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split

from PIL import Image, ImageDraw, ImageFont

import torch

import torch
from datasets import Dataset, Features, Sequence, ClassLabel, Value, Array2D, Array3D
from transformers import (
    LayoutLMv3FeatureExtractor,
    LayoutLMv3Tokenizer,
    LayoutLMv3Processor,
    LayoutLMv3ForSequenceClassification,
    AdamW,
)

dataset_path = (
    "/home/greg/datasets/private/data-hipa/medical_page_classification/output/images"
)

labels = [
    label
    for label in os.listdir(dataset_path)
    if os.path.isdir(os.path.join(dataset_path, label))
]
idx2label = {idx: label for idx, label in enumerate(labels)}
label2idx = {label: idx for idx, label in enumerate(labels)}

print(labels)
print(idx2label)
print(label2idx)

images = []
labels = []

for label in os.listdir(dataset_path):
    for image in os.listdir(os.path.join(dataset_path, label)):
        images.append(os.path.join(dataset_path, label, image))
        # labels.append(label2idx[label])
        labels.append(label)

print(len(images))
print(len(labels))

training_features = Features(
    {
        'pixel_values': Array3D(dtype="float32", shape=(3, 224, 224)),
        'input_ids': Sequence(feature=Value(dtype='int64')),
        'attention_mask': Sequence(Value(dtype='int64')),
        'token_type_ids': Sequence(Value(dtype='int64')),
        'bbox': Array2D(dtype="int64", shape=(512, 4)),
        'labels': ClassLabel(num_classes=len(label2idx), names=list(label2idx.keys())),
    }
)

feature_extractor = LayoutLMv3FeatureExtractor()
tokenizer = LayoutLMv3Tokenizer.from_pretrained("microsoft/layoutlmv3-base")
processor = LayoutLMv3Processor(feature_extractor, tokenizer)


data = pd.DataFrame({"image_path": images, "label": labels})


train_data, valid_data = train_test_split(
    data, test_size=0.2, random_state=42, stratify=data["label"]
)
train_data = train_data.reset_index(drop=True)
valid_data = valid_data.reset_index(drop=True)
print(f"{len(train_data)} training examples, {len(valid_data)} validation examples")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def encode_training_example(examples):
    images = [Image.open(path).convert("RGB") for path in examples['image_path']]

    encoded_inputs = processor(
        images,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_token_type_ids=True,
    )
    encoded_inputs["labels"] = [label2idx[label] for label in examples["label"]]

    return encoded_inputs


def training_dataloader_from_df(data):
    dataset = Dataset.from_pandas(data)

    encoded_dataset = dataset.map(
        encode_training_example,
        remove_columns=dataset.column_names,
        features=training_features,
        batched=True,
        batch_size=2,
    )

    encoded_dataset.set_format(type='torch', device=device)
    dataloader = torch.utils.data.DataLoader(
        encoded_dataset, batch_size=4, shuffle=True
    )
    batch = next(iter(dataloader))
    return dataloader


train_dataloader = training_dataloader_from_df(train_data)
valid_dataloader = training_dataloader_from_df(valid_data)
