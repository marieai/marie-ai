import os

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
    kwargs = {"__model_path__": os.path.expanduser("~/tmp/models")}
    model_name_or_path = (
        "marie/layoutlmv3-document-classification"  # Test model based on LayoutLMv3
    )
    model_name_or_path = ModelRegistry.get_local_path(model_name_or_path, **kwargs)
    assert os.path.exists(model_name_or_path)

    documents = docs_from_file("~/tmp/models/mpc/158955602_1.png")
    ocr_results = load_json_file("~/tmp/models/mpc/158955602_1.json")
    words, boxes = get_words_and_boxes(ocr_results)

    classifier = TransformersDocumentClassifier(model_name_or_path=model_name_or_path)
    classifier.run(documents=DocumentArray(documents), words=[words], boxes=[boxes])

    print("classifier", classifier)
