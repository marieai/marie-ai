import os

from docarray import DocumentArray

from marie.components import TransformersDocumentClassifier
from marie.logging.mdc import MDC
from marie.ocr.util import get_words_and_boxes
from marie.registry.model_registry import ModelRegistry
from marie.utils.docs import docs_from_file
from marie.utils.json import load_json_file


def test_sequence_classifier():
    # kwargs = {"__model_path__": __model_path__}
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    model_name_or_path = "marie/layoutlmv3-document-classification"
    # model_name_or_path = "hf://microsoft/layoutlmv3-base"

    if False:
        kwargs = {
            # "__model_path__": os.path.expanduser("~/tmp/models"),
            "use_auth_token": False,
        }  # custom model path

        resolved_model_name_or_path = ModelRegistry.get(model_name_or_path, version=None,
                                                        raise_exceptions_for_missing_entries=True,
                                                        **kwargs)
        print("resolved_model_name_or_path", resolved_model_name_or_path)
        return

    documents = docs_from_file("~/tmp/models/mpc/158955602_1.png")
    ocr_results = load_json_file("~/tmp/models/mpc/158955602_1.json")
    words, boxes = get_words_and_boxes(ocr_results, 0)

    classifier = TransformersDocumentClassifier(model_name_or_path=model_name_or_path)
    results = classifier.run(documents=DocumentArray(documents), words=[words], boxes=[boxes])

    for document in results:
        assert 'classification' in document.tags
        classification = document.tags['classification']
        assert 'label' in classification
        assert 'score' in classification
        print("classification", classification)
