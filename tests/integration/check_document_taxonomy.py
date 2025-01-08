import os.path
import random
from typing import List

import numpy as np
import torch

from marie.boxes.dit.ulim_dit_box_processor import visualize_bboxes
from marie.components.document_taxonomy import (
    DocumentTaxonomyClassification,
    DocumentTaxonomySeq2SeqLM,
    QaVitDocumentTaxonomy,
    TaxonomyPrediction,
)
from marie.components.document_taxonomy.util import (
    group_taxonomies_by_label,
    merge_annotations,
)
from marie.constants import __model_path__
from marie.utils.docs import docs_from_file, frames_from_docs
from marie.utils.json import load_json_file


def setup_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def load_docs_metadata(file_path: str) -> tuple:
    base_dir = os.path.dirname(file_path)
    filename = os.path.basename(file_path)

    filepath = os.path.expanduser(os.path.join(base_dir, filename))
    documents = docs_from_file(filepath)
    metadata = load_json_file(os.path.expanduser(os.path.join(base_dir, "result.json")))
    return documents, metadata


def check_document_taxonomy():
    setup_seed(42)
    model_name_or_path = os.path.join(__model_path__, "taxonomy/document")

    model_type = 'qavit'
    model_type = 'flan-t5-seq2seq'

    if model_type == 'flan-t5':
        model_name_or_path = 'marie/flan-t5-taxonomy-document'
        processor = DocumentTaxonomyClassification(
            model_name_or_path=model_name_or_path,
            use_gpu=True,
        )
    elif model_type == 'flan-t5-seq2seq':
        model_name_or_path = 'marie/flan-t5-taxonomy-document-seq2seq'
        label2id = {"TABLE": 0, "SECTION": 1, "CODES": 2, "OTHER": 3}
        id2label = {id: label for label, id in label2id.items()}

        processor = DocumentTaxonomySeq2SeqLM(
            model_name_or_path=model_name_or_path,
            use_gpu=True,
            k_completions=5,
            id2label=id2label,
        )
    elif model_type == 'qavit':
        model_name_or_path = 'marie/visual-t5-taxonomy-document'
        processor = QaVitDocumentTaxonomy(
            model_name_or_path=model_name_or_path,
            use_gpu=True
        )

    filepath = "~/tmp/test-deck/0_1735852778890/158955602_1.png"
    documents, metadata = load_docs_metadata(filepath)
    frames = frames_from_docs(documents)
    print("Documents: ", len(documents))
    taxonomy_key = f"taxonomy"
    return_docs = processor.run(documents, [metadata], taxonomy_key=taxonomy_key)
    annotations: List[TaxonomyPrediction] = return_docs[0].tags[taxonomy_key]
    merged = merge_annotations(annotations, metadata, taxonomy_key, default_label="OTHER")
    taxonomy_groups = group_taxonomies_by_label(merged["lines"], taxonomy_key)

    print("Annotations: ", annotations)
    print("taxonomy_groups: ", taxonomy_groups)

    image = frames[0]
    image_width = image.shape[1]
    taxonomy_line_bboxes = [group["bbox"] for group in taxonomy_groups]
    taxonomy_labels = [group["label"] for group in taxonomy_groups]
    taxonomy_line_bboxes = [
        [0, bbox[1], image_width, bbox[3]] for bbox in taxonomy_line_bboxes
    ]

    taxonomy_overlay_image = visualize_bboxes(
        image, np.array(taxonomy_line_bboxes), format="xywh", labels=taxonomy_labels
    )
    overlay_filename = os.path.expanduser(os.path.join(os.path.dirname(filepath), "taxonomy_overlay.png"))
    taxonomy_overlay_image.save(overlay_filename)
    print(f"Saved overlay image to {overlay_filename}")


if __name__ == "__main__":
    check_document_taxonomy()
