import json
import os
import pickle as pkl
import re
from PIL import Image
import pytesseract
from tqdm import tqdm
from typing import Optional, Any, Tuple
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    recall_score,
    precision_score,
)
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from multiprocessing import Pool
import torch
from torch import Tensor
import numpy as np
from typing import List, Dict


def convert_classes_to_list(classes: Dict[str, int]) -> List[str]:
    """get classes from config file and returns list of classes in right order"""
    classes_rev = {v: k for k, v in classes.items()}
    classes_names = [classes_rev[i] for i in range(len(classes_rev))]
    return classes_names


def check_doc_orientation(file_path: str, resize: bool = False) -> Tuple[str, int]:
    """detect orientation for single document-image and return rotation"""
    file_name = os.path.basename(file_path)
    img = Image.open(file_path).convert("RGB")
    if resize:
        img.thumbnail((1000, 1000), Image.Resampling.LANCZOS)
    try:
        osd_output = pytesseract.image_to_osd(img)
        if (match := re.search(r"Rotate: (\d+)", osd_output)) is not None:
            rotation = int(match.group(1))
        else:
            rotation = 0
    except pytesseract.TesseractError:
        rotation = 0  # probably blank page
    return file_name, rotation


def check_orientation_prc_in_database(dir_path: str, chunk: int = 20) -> None:
    """calculate percentage of pages with non-zero rotation for input database"""
    db_files = [os.path.join(dir_path, f) for f in os.listdir(dir_path)]
    db_rotations = []
    with Pool() as pool:
        for result in tqdm(
            pool.imap(check_doc_orientation, db_files, chunksize=chunk),
            total=len(db_files),
        ):
            db_rotations.append(result)
    df = pd.DataFrame(db_rotations, columns=["filename", "rotation"])
    df.to_csv("ImagesRotationProcessed.csv", index=False)

    rotation_prc = (df["rotation"] != 0).mean() * 100
    print(f"{rotation_prc:.2f}% of rotated pages. Total no. pages {len(db_files)}")


def detect_page_rotation(page_path: str, compress_size: bool = False) -> int:
    """detect single page (single image) orientation/rotation"""
    with Image.open(page_path) as img:
        img = img.convert("RGB")
        if compress_size:
            img.thumbnail((1000, 1000), Image.Resampling.LANCZOS)
        try:
            osd = pytesseract.image_to_osd(img)
            if (match := re.search(r"Rotate: (\d+)", osd)) is not None:
                rotation = int(match.group(1))
            else:
                rotation = 0
        except pytesseract.TesseractError:
            rotation = 0  # probably blank page
    return rotation


def save_data_to_pkl(data: Any, out_path: str, filename: Optional[str] = None) -> None:
    if not filename:
        filename = "input_db.pkl"
    out_path = os.path.join(out_path, filename)
    with open(out_path, "wb") as f:
        pkl.dump(data, f)


def load_data_from_pkl(data_path: str) -> Any:
    with open(data_path, "rb") as f:
        data = pkl.load(f)
    return data


def load_json(file_path: str) -> Any:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def apply_classification_report(
    y_true: list,
    y_predict: list,
    num_classes: int,
    epoch: int,
    writer: Any,
    classes_names: list[Any] | None = None,
) -> None:
    """Performance evaluation - main evaluation metrics"""
    # if classes_names is None:
    #     classes_names = class_names

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_predict, labels=list(range(num_classes)), zero_division=0
    )
    header = "| Class | Precision | Recall | F1 | Support |\n|-------|-----------|--------|----|---------|\n"
    rows = ""
    for i in range(num_classes):
        rows += f"| {classes_names[i]} | {precision[i]:.3f} | {recall[i]:.3f} | {f1[i]:.3f} | {support[i]} |\n"
    macro_f1 = f1.mean()
    macro_recall = recall_score(y_true, y_predict, average="macro")
    macro_precision = precision_score(y_true, y_predict, average="macro")
    rows += f"\n**Macro F1**: {macro_f1:.3f}"
    rows += f"\n**Macro Recall**: {macro_recall:.3f}"
    rows += f"\n**Macro Precision**: {macro_precision:.3f}"
    writer.add_text("Val/Classification_Report", header + rows, epoch)


def compute_document_level_metrics(
    predicted_docs: set[tuple[int, ...]],
    true_docs: set[tuple[int, ...]],
    epoch: int,
    writer: Any,
) -> None:
    """Performance evaluation - metrics on documents level
    Args:
        predicted_docs (set of tuples): np. {(0,0,0), (1,1)}
        true_docs (set of tuples): np. {(0,0,0), (1,1,1)}
        epoch: current epoch number
        writer: tensorboard writer
    """
    tp = len(predicted_docs & true_docs)
    fp = len(predicted_docs - true_docs)
    fn = len(true_docs - predicted_docs)
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    )
    rows = f"\n**Document Level F1**: {f1:.3f}"
    rows += f"\n**Document Level Recall**: {recall:.3f}"
    rows += f"\n**Document Level Precision**: {precision:.3f}"
    writer.add_text("Val/Classification_Report_Document_Level", rows, epoch)


def apply_confusion_matrix(
    y_true: list,
    y_predict: list,
    num_classes: int,
    epoch: int,
    writer: Any,
    classes_names: list[Any] | None = None,
) -> None:
    """Performance evaluation - confusion matrix"""
    # if classes_names is None:
    #     classes_names = class_names

    cm = confusion_matrix(y_true, y_predict, labels=list(range(num_classes)))

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=classes_names,
        yticklabels=classes_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix – Epoch {epoch + 1}")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    writer.add_figure("Val/Confusion_Matrix", fig, epoch)
    plt.close(fig)


def apply_proba_confidence_heatmap(
    all_labels: List[Tensor],
    all_logits: List[Tensor],
    num_classes: int,
    epoch: int,
    writer: Any,
    classes_names: Optional[list] = None,
) -> None:
    """Performance evaluation - confidence heatmap"""
    # if classes_names is None:
    #     classes_names = class_names

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    probs = torch.softmax(logits, dim=1)  # [n_steps, num_classes]
    predictions = probs.argmax(dim=1)
    confidences = probs.max(dim=1).values
    correct = predictions == labels
    wrong = ~correct

    heatmap_data = np.zeros((2, num_classes))  # 0: correct, 1: wrong
    for cls in range(num_classes):
        cls_mask = labels == cls
        correct_conf = confidences[correct & cls_mask]
        wrong_conf = confidences[wrong & cls_mask]
        heatmap_data[0, cls] = (
            correct_conf.mean().item() if len(correct_conf) > 0 else np.nan
        )
        heatmap_data[1, cls] = (
            wrong_conf.mean().item() if len(wrong_conf) > 0 else np.nan
        )

    fig, ax = plt.subplots(figsize=(12, 4))
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        cbar=False,
        xticklabels=classes_names,
        yticklabels=["correct", "wrong"],
        ax=ax,
    )
    ax.set_title(f"Model Confidence Heatmap – Epoch {epoch + 1}")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    writer.add_figure("Val/Confidence_Heatmap", fig, epoch)
    plt.close(fig)


def count_label_stats(sequences: dict) -> Tuple[int, int, int]:
    """Calculate number of classes instances for binary classification of split/no-split documents"""
    total_labels = 0
    count_0 = 0
    count_1 = 0
    for seq in sequences.values():
        total_labels += len(seq["labels"])
        count_1 += sum(seq["labels"])
        count_0 += len(seq["labels"]) - sum(seq["labels"])
    return total_labels, count_0, count_1


def reconstruct_lbx_documents(
    documents_sequences: dict[int, list[tuple[int, int]]],
) -> set[tuple[int, ...]]:
    """
    Args:
        documents_sequences: {lbx_id1: [(page_idx, label), (page_idx, label), (page_idx, label)], lbx_id2: [(.., ..)],}
    Returns: set{(p1,p2,p3), (p4,p5), (p6), .....}
    """
    docs = set()
    for seq_id, pages in documents_sequences.items():
        sorted_pages = sorted(pages, key=lambda x: x[0])
        current_doc = [sorted_pages[0][0]]
        for idx, label in sorted_pages[1:]:
            current_doc.append(idx)
            if label == 1:
                docs.add(tuple(current_doc))
                current_doc = []
        if current_doc:
            docs.add(tuple(current_doc))
    return docs
