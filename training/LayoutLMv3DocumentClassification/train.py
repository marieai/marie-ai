import io
import json
import multiprocessing
import os
from pathlib import Path
import torch.nn.functional as F

from PIL import Image, ImageDraw, ImageFont
from fsspec.core import url_to_fs
import torch.nn.functional as F
import pandas as pd
import pytorch_lightning as pl
import torch
from matplotlib import pyplot as plt
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
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

dataset_pathXX = os.path.expanduser(
    "~/datasets/private/medical_page_classification/output/images"
)


def load_data():
    labels = []
    df_labels = []
    df_images = []

    for label in sorted(os.listdir(dataset_path)):
        labels.append(label)
        for image in os.listdir(os.path.join(dataset_path, label)):
            df_images.append(os.path.join(dataset_path, label, image))
            df_labels.append(label)

    idx2label = {idx: label for idx, label in enumerate(labels)}
    label2idx = {label: idx for idx, label in enumerate(labels)}

    print(labels)
    print(idx2label)
    print(label2idx)

    data = pd.DataFrame({'image_path': df_images, 'label': df_labels})
    return data, labels, idx2label, label2idx


model_name_or_path = "microsoft/layoutlmv3-base"
# model_name_or_path = "microsoft/layoutlmv3-large"


def create_processor():
    feature_extractor = LayoutLMv3ImageProcessor(
        apply_ocr=False, do_resize=True, resample=Image.LANCZOS
    )
    tokenizer = LayoutLMv3Tokenizer.from_pretrained(model_name_or_path)
    processor = LayoutLMv3Processor(feature_extractor, tokenizer)
    return processor


def create_split_data(data):
    train_data, test_data = train_test_split(
        data, test_size=0.2, random_state=42, stratify=data["label"]
    )

    train_data = train_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)
    valid_data = test_data  # TODO : need to have proper validation dataset

    print(f"{len(train_data)} training examples, {len(test_data)} test examples")
    return train_data, test_data, valid_data


def create_dataset(data, labels, processor):
    train_data, test_data, valid_data = create_split_data(data)

    train_dataset = DocumentClassificationDataset(False, train_data, labels, processor)
    test_dataset = DocumentClassificationDataset(False, valid_data, labels, processor)
    validation_dataset = test_dataset  # TODO : need to have proper validation dataset

    return train_dataset, test_dataset, validation_dataset


class HfModelCheckpoint(ModelCheckpoint):
    def _save_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
        super()._save_checkpoint(trainer, filepath)
        print("Saving model checkpoint to Huggingface model...")
        hf_save_dir = filepath + ".dir"
        if trainer.is_global_zero:
            trainer.lightning_module.model.save_pretrained(hf_save_dir)
            trainer.lightning_module.tokenizer.save_pretrained(hf_save_dir)

    # https://github.com/Lightning-AI/lightning/pull/16067
    def _remove_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
        super()._remove_checkpoint(trainer, filepath)
        print("Removing model checkpoint from Huggingface model...")
        hf_save_dir = filepath + ".dir"
        if trainer.is_global_zero:
            fs, _ = url_to_fs(hf_save_dir)
            if fs.exists(hf_save_dir):
                fs.rm(hf_save_dir, recursive=True)


class ModelModule(pl.LightningModule):
    def __init__(self, classes: list):
        super().__init__()
        n_classes = len(classes)

        config = AutoConfig.from_pretrained(
            model_name_or_path,
            num_labels=n_classes,
            finetuning_task="document-classification",
            cache_dir="/mnt/data/cache",
            input_size=224,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            has_relative_attention_bias=True,
        )

        self.model = LayoutLMv3ForSequenceClassification.from_pretrained(
            model_name_or_path,
            # num_labels=n_classes,
            config=config,
        )
        self.tokenizer = LayoutLMv3Tokenizer.from_pretrained(model_name_or_path)

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
    # prepare data
    data, labels, idx2label, label2idx = load_data()
    processor = create_processor()

    train_dataset, test_dataset, valid_dataset = create_dataset(
        data, labels, processor=processor
    )
    train_data_loader = DataLoader(
        train_dataset, batch_size=12, shuffle=True, num_workers=12
    )
    test_data_loader = DataLoader(
        test_dataset, batch_size=4, shuffle=False, num_workers=4
    )

    # train
    model_module = ModelModule(classes=labels)
    model_checkpoint = HfModelCheckpoint(
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
        max_epochs=50,
        callbacks=[model_checkpoint],
    )

    trainer.fit(
        model_module,
        train_data_loader,
        test_data_loader,
        ckpt_path="/home/greg/dev/marieai/marie-ai/training/LayoutLMv3DocumentClassification/lightning_logs/version_2/checkpoints/epoch=4-step=10020-val_loss=0.1739.ckpt",
    )


def predict_document_image(
    image_path: Path,
    model: LayoutLMv3ForSequenceClassification,
    processor: LayoutLMv3Processor,
    device: str = "cuda",
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
    #
    # _predictions = outputs.logits.argmax(-1).squeeze().tolist()
    # _token_boxes = encoding.bbox.squeeze().tolist()
    # normalized_logits = outputs.logits.softmax(dim=-1).squeeze().tolist()
    #
    #
    logits = output.logits
    predicted_class = logits.argmax(-1)
    probabilities = F.softmax(logits, dim=-1).squeeze().tolist()

    return (
        model.config.id2label[predicted_class.item()],
        probabilities[predicted_class.item()],
    )


def inference():
    # load ckpt for inference
    model_checkpoint_path = "/home/greg/dev/marieai/marie-ai/training/LayoutLMv3DocumentClassification/lightning_logs/version_0/checkpoints/epoch=4-step=10020-val_loss=0.1633.ckpt.dir"
    model_checkpoint_path = "/home/greg/dev/marieai/marie-ai/training/LayoutLMv3DocumentClassification/lightning_logs/version_4/checkpoints/epoch=7-step=13026-val_loss=0.1810.ckpt.dir"
    model_checkpoint_path = "/home/greg/dev/marieai/marie-ai/training/LayoutLMv3DocumentClassification/lightning_logs/version_2/checkpoints/epoch=4-step=10020-val_loss=0.1739.ckpt.dir"
    model_checkpoint_path = "/home/greg/dev/marieai/marie-ai/training/LayoutLMv3DocumentClassification/lightning_logs/version_8/checkpoints/epoch=11-step=23523-val_loss=0.1027.ckpt.dir"
    data, labels, idx2label, label2idx = load_data()
    processor = create_processor()
    train_data, test_data, valid_data = create_split_data(data)

    # load model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    model = LayoutLMv3ForSequenceClassification.from_pretrained(
        os.path.expanduser(model_checkpoint_path)
    )
    model = model.eval().to(device)

    print(model.config.id2label)
    true_labels = []
    pred_labels = []

    for df in tqdm(zip(valid_data['label'], valid_data['image_path'])):
        label, image_path = df
        annotation_path = image_path.replace('images', 'annotations').replace(
            '.png', '.json'
        )
        if not os.path.exists(annotation_path):
            print(
                f"Missing annotation file for {annotation_path} for image {image_path}"
            )
            continue

        predicted_label, probabilities = predict_document_image(
            image_path, model, processor, device
        )

        print(
            f"Expected / predicted label: {label} , {predicted_label} with probabilities: {probabilities}"
        )

        true_labels.append(label)
        pred_labels.append(predicted_label)

        # if len(true_labels) > 5000:
        #     break

    print("Classification report")
    print(labels)
    print(true_labels)
    print(pred_labels)

    print(classification_report(true_labels, pred_labels, labels=labels))

    cm = confusion_matrix(true_labels, pred_labels, labels=labels)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    cm_display.plot()
    cm_display.ax_.set_xticklabels(labels, rotation=45)
    cm_display.figure_.set_size_inches(16, 8)

    plt.show()


if __name__ == "__main__":
    # train()
    inference()
