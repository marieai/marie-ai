import os

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
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

pl.seed_everything(42)

dataset_path = os.path.expanduser(
    "~/datasets/private/data-hipa/medical_page_classification/output/images"
)

dataset_path = os.path.expanduser(
    "~/datasets/private/medical_page_classification/output/images"
)

images = []
labels_unique = []
labels = []

for label in sorted(os.listdir(dataset_path)):
    labels_unique.append(label)
    for image in os.listdir(os.path.join(dataset_path, label)):
        images.append(os.path.join(dataset_path, label, image))
        labels.append(label)

idx2label = {idx: label for idx, label in enumerate(labels_unique)}
label2idx = {label: idx for idx, label in enumerate(labels_unique)}

print(labels)
print(idx2label)
print(label2idx)

data = pd.DataFrame({'image_path': images, 'label': labels})

print(len(images))
print(len(labels))

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

train_dataset = DocumentClassificationDataset(
    False, train_data, labels_unique, processor
)
test_dataset = DocumentClassificationDataset(
    False, valid_data, labels_unique, processor
)

train_data_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)

test_data_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)


class ModelModule(pl.LightningModule):
    def __init__(self, classes: list):
        super().__init__()
        n_classes = len(classes)
        self.model = LayoutLMv3ForSequenceClassification.from_pretrained(
            "microsoft/layoutlmv3-base", num_labels=n_classes
        )
        self.model.config.id2label = {k: v for k, v in enumerate(classes)}
        self.model.config.label2id = {v: k for k, v in enumerate(classes)}
        self.train_accuracy = Accuracy(task="multiclass", num_classes=n_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=n_classes)

    def forward(self, input_ids, attention_mask, bbox, pixel_values, labels=None):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            bbox=bbox,
            pixel_values=pixel_values,
            labels=labels,
        )

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        bbox = batch["bbox"]
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]
        output = self(input_ids, attention_mask, bbox, pixel_values, labels)
        self.log("train_loss", output.loss)
        self.log(
            "train_acc",
            self.train_accuracy(output.logits, labels),
            on_step=True,
            on_epoch=True,
        )
        return output.loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        bbox = batch["bbox"]
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]
        output = self(input_ids, attention_mask, bbox, pixel_values, labels)
        self.log("val_loss", output.loss)
        self.log(
            "val_acc",
            self.val_accuracy(output.logits, labels),
            on_step=False,
            on_epoch=True,
        )
        return output.loss

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=0.00001)  # 1e-5
        return optimizer


def train():
    model_module = ModelModule(classes=labels)
    model_checkpoint = ModelCheckpoint(
        filename="{epoch}-{step}-{val_loss:.4f}",
        save_last=True,
        save_top_k=3,
        monitor="val_loss",
        mode="min",
    )

    trainer = pl.Trainer(
        accelerator="cpu",
        precision=16,
        devices=1,
        max_epochs=5,
        callbacks=[model_checkpoint],
    )

    trainer.fit(model_module, train_data_loader, test_data_loader)


if __name__ == "__main__":
    train()
