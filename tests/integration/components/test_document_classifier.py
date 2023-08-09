import os
from typing import Dict, Any
from urllib.parse import urlparse

from docarray import DocumentArray

from marie.components import TransformersDocumentClassifier
from marie.logging.mdc import MDC
from marie.registry.model_registry import ModelRegistry
from marie.utils.docs import docs_from_file
from marie.utils.json import load_json_file
from marie.constants import __model_path__

MDC.put("request_id", "")


def get_words_and_boxes(ocr_results) -> tuple:
    words = []
    boxes = []
    for w in ocr_results[0]["words"]:
        boxes.append(w["box"])
        words.append(w["text"])
    return words, boxes


def test_sequence_classifier():
    # kwargs = {"__model_path__": __model_path__}
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    model_name_or_path = "zoo://marie/layoutlmv3-document-classification"  # Test model based on LayoutLMv3
    # model_name_or_path = "hf://microsoft/layoutlmv3-base"

    if True:
        kwargs = {"__model_path__": os.path.expanduser("~/tmp/models")}  # custom model path
        kwargs = {"use_auth_token": False}

        resolved_model_name_or_path = ModelRegistry.get(model_name_or_path, version=None,
                                                        raise_exceptions_for_missing_entries=True,
                                                        **kwargs)
        print("resolved_model_name_or_path", resolved_model_name_or_path)

    return
    documents = docs_from_file("~/tmp/models/mpc/158955602_1.png")
    ocr_results = load_json_file("~/tmp/models/mpc/158955602_1.json")
    words, boxes = get_words_and_boxes(ocr_results)

    classifier = TransformersDocumentClassifier(model_name_or_path=model_name_or_path)
    # classifier.run(documents=DocumentArray(documents), words=[words], boxes=[boxes])
    print("classifier", classifier)
