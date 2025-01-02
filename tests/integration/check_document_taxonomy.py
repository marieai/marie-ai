import copy
import os.path
import random
from typing import Dict, List

import numpy as np
import torch

from marie.components.document_taxonomy.datamodel import TaxonomyPrediction
from marie.components.document_taxonomy.transformers import TransformersDocumentTaxonomy
from marie.components.document_taxonomy.util import (
    group_taxonomies_by_label,
    merge_annotations,
)
from marie.constants import __model_path__
from marie.utils.docs import docs_from_file
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
    model_name_or_path = 'marie/taxonomy-document'

    processor = TransformersDocumentTaxonomy(
        model_name_or_path=model_name_or_path,
        use_gpu=True,
    )

    filepath = "~/tmp/test-deck/0_1735852778890/158955602_1.png"
    documents, metadata = load_docs_metadata(filepath)

    print("Documents: ", len(documents))
    taxonomy_key = f"taxonomy"
    return_docs = processor.run(documents, [metadata], taxonomy_key=taxonomy_key)
    annotations: List[TaxonomyPrediction] = return_docs[0].tags[taxonomy_key]
    merged = merge_annotations(annotations, metadata, taxonomy_key)
    grouped = group_taxonomies_by_label(merged["lines"], taxonomy_key)

    print("Annotations: ", annotations)
    print("grouped: ", grouped)


if __name__ == "__main__":
    check_document_taxonomy()
