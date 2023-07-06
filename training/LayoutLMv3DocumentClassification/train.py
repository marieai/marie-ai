import io
import json
import multiprocessing
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from tqdm import tqdm
from transformers import (
    LayoutLMv3Tokenizer,
    LayoutLMv3Processor,
    LayoutLMv3ForSequenceClassification,
    AdamW,
    LayoutLMv3ImageProcessor,
    AutoConfig,
    AutoModel,
)

from training.LayoutLMv3DocumentClassification.llmv3_dataset import (
    DocumentClassificationDataset,
    scale_bounding_box,
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

model_name_or_path = "microsoft/layoutlmv3-base"
model_name_or_path = "microsoft/layoutlmv3-large"

feature_extractor = LayoutLMv3ImageProcessor(
    apply_ocr=False, do_resize=True, resample=Image.LANCZOS
)
tokenizer = LayoutLMv3Tokenizer.from_pretrained(model_name_or_path)
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
# multiprocessing.cpu_count() //
train_data_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)


class ModelModule(pl.LightningModule):
    def __init__(self, classes: list):
        super().__init__()
        n_classes = len(classes)

        config = AutoConfig.from_pretrained(
            model_name_or_path,
            num_labels=len(labels),
            finetuning_task="document-classification",
            cache_dir="/mnt/data/cache",
            input_size=224,
            hidden_dropout_prob=0.2,
            attention_probs_dropout_prob=0.2,
            has_relative_attention_bias=False,
        )

        self.model = LayoutLMv3ForSequenceClassification.from_pretrained(
            model_name_or_path,
            # num_labels=n_classes,
            config=config,
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

    early_stop_callback = EarlyStopping(
        monitor='val_loss', patience=1, strict=False, verbose=False, mode='min'
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        precision=16,
        devices=1,
        max_epochs=25,
        callbacks=[model_checkpoint],
    )

    trainer.fit(model_module, train_data_loader, test_data_loader)


def predict_document_image(
    image_path: Path,
    model: LayoutLMv3ForSequenceClassification,
    processor: LayoutLMv3Processor,
    device: str = "cpu",
):

    annotation_path = (
        str(image_path).replace('images', 'annotations').replace('.png', '.json')
    )
    if not os.path.exists(annotation_path):
        print(f"Missing annotation file for {annotation_path} for image {image_path}")

    with io.open(annotation_path, "r", encoding="utf-8") as json_file:
        ocr_results = json.load(json_file)

        with Image.open(image_path).convert("RGB") as image:
            width, height = image.size
            width_scale = 1000 / width
            height_scale = 1000 / height

            words = []
            boxes = []
            for w in ocr_results[0]["words"]:
                boxes.append(scale_bounding_box(w["box"], width_scale, height_scale))
                words.append(w["text"])

            encoding = processor(
                image,
                words,
                boxes=boxes,
                max_length=512,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

    with torch.inference_mode():
        output = model(
            input_ids=encoding["input_ids"].to(device),
            attention_mask=encoding["attention_mask"].to(device),
            bbox=encoding["bbox"].to(device),
            pixel_values=encoding["pixel_values"].to(device),
        )

    predicted_class = output.logits.argmax()
    return model.config.id2label[predicted_class.item()]


def inference():
    # load ckpt for inference
    model_name_or_path = "~/dev/marieai/marie-ai/training/LayoutLMv3DocumentClassification/lightning_logs/version_4/checkpoints/epoch=21-step=9702-val_loss=0.1855.ckpt"
    model = ModelModule.load_from_checkpoint(model_name_or_path)

    return

    # load model
    device = "cpu"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_checkpoint_path = "~/dev/marieai/marie-ai/training/LayoutLMv3DocumentClassification/lightning_logs/version_4/checkpoints/epoch=21-step=9702-val_loss=0.1855.ckpt"
    model = LayoutLMv3ForSequenceClassification.from_pretrained(
        os.path.expanduser(model_checkpoint_path)
    )
    model = model.eval().to(device)

    labels = []
    predictions = []
    test_images = []

    for image_path in tqdm(test_images):
        labels.append(image_path.parent.name)
        predictions.append(predict_document_image(image_path, model, processor))


if __name__ == "__main__":
    # train()
    inference()
