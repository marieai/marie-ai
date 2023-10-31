import io
import json
import os
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from PIL import Image
from fsspec.core import url_to_fs
from imblearn.over_sampling import RandomOverSampler
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
)
from pytorch_lightning.loggers import WandbLogger

from marie.logging.profile import TimeContext
from training.LayoutLMv3DocumentClassification.llmv3_dataset import (
    DocumentClassificationDataset,
    scale_bounding_box,
)

wandb_logger = WandbLogger(log_model="all")
pl.seed_everything(42)

dataset_pathXX = os.path.expanduser(
    "~/datasets/private/data-hipa/medical_page_classification/output/images"
)

dataset_path = os.path.expanduser(
    "~/datasets/private/payer-determination/output/images"
)
dataset_path = os.path.expanduser("~/datasets/private/corr-routing/ready/images")

dataset_path = os.path.expanduser("~/datasets/private/corr-routing/ready/images")


def load_data():
    labels = []
    df_labels = []
    df_images = []

    for label in sorted(os.listdir(dataset_path)):
        items = os.listdir(os.path.join(dataset_path, label))
        labels.append(label)
        print(f"label: {label} >> {len(items)}")

        for image in items:
            # check if file is an image
            extension = os.path.splitext(image)[1]
            if extension not in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
                continue
            df_images.append(os.path.join(dataset_path, label, image))
            df_labels.append(label)

    idx2label = {idx: label for idx, label in enumerate(labels)}
    label2idx = {label: idx for idx, label in enumerate(labels)}

    print(labels)
    print(idx2label)
    print(label2idx)

    data = pd.DataFrame({"image_path": df_images, "label": df_labels})
    return data, labels, idx2label, label2idx


# model_name_or_path = "microsoft/layoutlmv3-base"
model_name_or_path = "microsoft/layoutlmv3-large"


def create_processor():
    feature_extractor = LayoutLMv3ImageProcessor(
        apply_ocr=False, do_resize=True, resample=Image.BICUBIC
    )
    tokenizer = LayoutLMv3Tokenizer.from_pretrained(model_name_or_path)
    processor = LayoutLMv3Processor(feature_extractor, tokenizer)
    return processor


def create_split_data(data):
    # limit data for testing / debugging
    # data = data.sample(frac=0.1, random_state=42)

    train_data, test_data = train_test_split(
        data, test_size=0.20, random_state=42, stratify=data["label"]
    )

    train_data = train_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)
    valid_data = test_data  # TODO : need to have proper validation dataset

    print(f"{len(train_data)} training examples, {len(test_data)} test examples")
    return train_data, test_data, valid_data


from imblearn.under_sampling import RandomUnderSampler


def create_split_data_strat(data):
    print(data)
    X = data  # .drop(columns=["label"])
    y = data["label"]

    # perform under-sampling.
    # rus = RandomUnderSampler(random_state=42)
    rus = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = rus.fit_resample(X, y)

    # split data into train, test, and validation sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
    )

    train_data = X_train.reset_index(drop=True)
    test_data = X_test.reset_index(drop=True)

    return train_data, test_data


def create_datasetSTRAT(data, labels, processor):
    X_train, X_test = create_split_data_strat(data)

    print(
        f"STRAT DS: {len(X_train)} training examples, {len(X_test)} test examples, {len(X_test)}"
    )

    train_dataset = DocumentClassificationDataset(False, X_train, labels, processor)
    test_dataset = DocumentClassificationDataset(False, X_test, labels, processor)
    validation_dataset = test_dataset  # TODO : need to have proper validation dataset

    print(
        f"{len(train_dataset)} training examples, {len(test_dataset)} test examples, {len(validation_dataset)} validation examples"
    )
    return train_dataset, test_dataset, validation_dataset


def create_dataset(data, labels, processor):
    train_data, test_data, valid_data = create_split_data(data)

    print(
        f"{len(train_data)} training examples, {len(test_data)} test examples, {len(valid_data)} validation examples"
    )

    train_dataset = DocumentClassificationDataset(False, train_data, labels, processor)
    test_dataset = DocumentClassificationDataset(False, valid_data, labels, processor)
    validation_dataset = test_dataset  # TODO : need to have proper validation dataset

    print(
        f"{len(train_dataset)} training examples, {len(test_dataset)} test examples, {len(validation_dataset)} validation examples"
    )
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
        self.save_hyperparameters()
        n_classes = len(classes)

        config = AutoConfig.from_pretrained(
            model_name_or_path,
            num_labels=n_classes,
            finetuning_task="corr-classification",
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
        train_dataset, batch_size=4, shuffle=True, num_workers=0
    )
    test_data_loader = DataLoader(
        test_dataset, batch_size=2, shuffle=False, num_workers=0
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
        monitor="val_loss", patience=1, strict=False, verbose=False, mode="min"
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        precision="32-true",
        devices=1,
        max_epochs=50,
        callbacks=[model_checkpoint],
        logger=wandb_logger,
    )

    trainer.fit(
        model_module,
        train_data_loader,
        test_data_loader,
        # ckpt_path="/home/greg/dev/marieai/marie-ai/training/LayoutLMv3DocumentClassification/lightning_logs/version_2/checkpoints/epoch=4-step=10020-val_loss=0.1739.ckpt",
        # ckpt_path="/home/gbugaj/dev/marieai/marie-ai/training/LayoutLMv3DocumentClassification/lightning_logs/9ymwbfy4/checkpoints/epoch=3-step=11380-val_loss=0.3742.ckpt",
    )


def predict_document_image(
    image_path: Path,
    model: LayoutLMv3ForSequenceClassification,
    processor: LayoutLMv3Processor,
    device: str = "cuda",
):
    annotation_path = image_path.replace("images", "annotations")
    last = annotation_path.rfind(".")
    annotation_path = annotation_path[:last] + ".json"

    if not os.path.exists(annotation_path):
        print(f"Missing annotation file for {annotation_path} for image {image_path}")
        return -1, -1

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


def infer_single_image(label, image_path, model, processor, device):
    annotation_path = image_path.replace("images", "annotations")
    last = annotation_path.rfind(".")
    annotation_path = annotation_path[:last] + ".json"
    if not os.path.exists(annotation_path):
        print(f"Missing annotation file for {annotation_path} for image {image_path}")
        return -1, -1

    return predict_document_image(image_path, model, processor, device)


def inference(model_checkpoint_path: str):
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

    # With compile :
    #  Inference time modes :
    #  reduce-overhead : Inference time  takes 54 seconds (54.157s)
    #  default         : Inference time takes 51 seconds (51.553s)
    #  max-autotune    : Inference time takes 56 seconds (56.915s)
    # mode :onnxrt Inference time takes 49 seconds (49.465s)

    # Without compile :
    #   Inference time  takes 1 minute and 1 second (61.28s)

    print(model.config.id2label)
    print(torch._dynamo.list_backends())

    if True:
        try:
            with TimeContext("Compile model"):
                import torchvision.models as models
                import torch._dynamo as dynamo

                # model = torch.compile(model, backend="inductor", mode="max-autotune")
                model = torch.compile(model)
                # model = torch.compile(model, backend="onnxrt", fullgraph=False)
                # model = torch.compile(model)
                print("Model compiled set")
        except Exception as err:
            print(f"Model compile not supported: {err}")

    true_labels = []
    pred_labels = []

    # forcing compilation of model
    with TimeContext("Inference model compile"):
        for df in tqdm(zip(valid_data["label"], valid_data["image_path"])):
            label, image_path = df
            infer_single_image(label, image_path, model, processor, device)
            break

    with TimeContext("Inference time"):
        for df in tqdm(zip(valid_data["label"], valid_data["image_path"])):
            label, image_path = df
            predicted_label, probabilities = infer_single_image(
                label, image_path, model, processor, device
            )

            true_labels.append(label)
            pred_labels.append(predicted_label)

            print(
                f"Expected / predicted label: {label} , {predicted_label} with probabilities: {probabilities}"
            )
            # get the last two directories from the path
            image_path = "/".join(image_path.split("/")[-2:])
            # write detailed results to a csv file
            matched = label == predicted_label
            with open("results.csv", "a") as f:
                f.write(
                    f"{matched},{label},{predicted_label},{probabilities},{image_path}\n"
                )

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

    # save the confusion matrix
    plt.savefig("confusion_matrix.png")
    plt.show()


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    torch.set_grad_enabled(True)
    # train()

    # load ckpt for inference
    model_checkpoint_path = (
        "/home/greg/dev/marieai/marie-ai/model_zoo/rms/corr-layoutlmv3"
    )
    inference(model_checkpoint_path)

# set this to avoid error
# export QT_QPA_PLATFORM=offscreen
# QObject::moveToThread: Current thread (0xa476410) is not the object's thread (0xc2504e0).
# ref : https://github.com/NVlabs/instant-ngp/discussions/300
