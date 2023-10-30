import os

import pandas as pd
import torch
from PIL import Image
from datasets import Dataset, Features, Sequence, ClassLabel, Value, Array2D, Array3D
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import trange
from tqdm.auto import tqdm
from transformers import (
    LayoutLMv3Tokenizer,
    LayoutLMv3Processor,
    LayoutLMv3ForSequenceClassification,
    AdamW,
    LayoutLMv3ImageProcessor,
)

from training.LayoutLMv3DocumentClassification.llmv3_dataset import (
    DocumentClassificationDataset,
)

dataset_path = os.path.expanduser(
    "~/datasets/private/data-hipa/medical_page_classification/output/images"
)

dataset_path = os.path.expanduser(
    "~/datasets/private/medical_page_classification/output/images"
)

dataset_path = os.path.expanduser(
    "~/datasets/private/payer-determination/output/images"
)

labels = [label for label in os.listdir(dataset_path)]

idx2label = {idx: label for idx, label in enumerate(labels)}
label2idx = {label: idx for idx, label in enumerate(labels)}

print(labels)
print(idx2label)
print(label2idx)

images = []
labels = []

for label in sorted(os.listdir(dataset_path)):
    for image in os.listdir(os.path.join(dataset_path, label)):
        images.append(os.path.join(dataset_path, label, image))
        # labels.append(label2idx[label])
        labels.append(label)

# for label in os.listdir(dataset_path):
#     images.extend([
#         f"{dataset_path}/{label}/{img_name}" for img_name in os.listdir(f"{dataset_path}/{label}")
#     ])
#     labels.extend([
#         label for _ in range(len(os.listdir(f"{dataset_path}/{label}")))
#     ])
data = pd.DataFrame({'image_path': images, 'label': labels})

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

feature_extractor = LayoutLMv3ImageProcessor(apply_ocr=False)
tokenizer = LayoutLMv3Tokenizer.from_pretrained("microsoft/layoutlmv3-base")
processor = LayoutLMv3Processor(feature_extractor, tokenizer)

train_data, valid_data = train_test_split(
    data, test_size=0.2, random_state=42, stratify=data["label"]
)

train_data = train_data.reset_index(drop=True)
valid_data = valid_data.reset_index(drop=True)
print(f"{len(train_data)} training examples, {len(valid_data)} validation examples")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def encode_training_exampleXX(examples):
    # Load the images
    images = [Image.open(path).convert("RGB") for path in examples['image_path']]

    # Encode the images
    words = ["hello", "world"]
    boxes = [[0, 0, 10, 10], [10, 10, 20, 20]]

    encoded_inputs = processor(
        # fmt: off
        images,
        words,
        boxes=boxes,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_token_type_ids=True
        # fmt: on
    )
    encoded_inputs["labels"] = [label2idx[label] for label in examples["label"]]

    return encoded_inputs


def encode_training_example(examples):
    images = [Image.open(path).convert("RGB") for path in examples['image_path']]

    # Encode the images
    words = []
    boxes = []

    encoded_inputs = processor(
        images,
        words,
        boxes=boxes,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_token_type_ids=True,
    )
    encoded_inputs["labels"] = [label2idx[label] for label in examples["label"]]

    return encoded_inputs


def training_dataloader_from_df(data):
    dataset = Dataset.from_pandas(data)

    print(dataset)

    encoded_dataset = dataset.map(
        encode_training_example,
        remove_columns=dataset.column_names,
        features=training_features,
        batched=True,
        batch_size=2,
    )

    encoded_dataset.set_format(type='torch', device=device)
    dataloader = DataLoader(encoded_dataset, batch_size=4, shuffle=True)

    batch = next(iter(dataloader))
    return dataloader


# train_dataloader = training_dataloader_from_df(train_data)
# valid_dataloader = training_dataloader_from_df(valid_data)


train_dataset = DocumentClassificationDataset(False, train_data, processor)
test_dataset = DocumentClassificationDataset(False, valid_data, processor)

train_data_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)

test_data_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)

model = LayoutLMv3ForSequenceClassification.from_pretrained(
    "microsoft/layoutlmv3-base", num_labels=len(label2idx)
)
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)
num_epochs = 5

for epoch in trange(num_epochs, desc="Training"):
    print("Epoch:", epoch)
    training_loss = 0.0
    training_correct = 0
    # put the model in training mode
    model.train()
    # for batch in tqdm(train_data_loader):
    for batch in tqdm(
        train_data_loader, desc=f"Epoch {epoch + 1} in training", leave=False
    ):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss

        training_loss += loss.item()
        predictions = outputs.logits.argmax(-1)
        training_correct += (predictions == batch['labels']).float().sum()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print("Training Loss:", training_loss / batch["input_ids"].shape[0])
    training_accuracy = 100 * training_correct / len(train_data)
    print("Training accuracy:", training_accuracy.item())

    validation_loss = 0.0
    validation_correct = 0
    for batch in tqdm(test_data_loader):
        outputs = model(**batch)
        loss = outputs.loss

        validation_loss += loss.item()
        predictions = outputs.logits.argmax(-1)
        validation_correct += (predictions == batch['labels']).float().sum()

    print("Validation Loss:", validation_loss / batch["input_ids"].shape[0])
    validation_accuracy = 100 * validation_correct / len(valid_data)
    print("Validation accuracy:", validation_accuracy.item())

model.save_pretrained('saved_model/')
