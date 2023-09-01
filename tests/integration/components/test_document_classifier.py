import os

from docarray import DocumentArray

from marie.components import TransformersDocumentClassifier
from marie.logging.mdc import MDC
from marie.logging.profile import TimeContext
from marie.ocr.util import get_words_and_boxes
from marie.registry.model_registry import ModelRegistry
from marie.utils.docs import docs_from_file
from marie.utils.json import load_json_file


def test_sequence_classifier():
    # kwargs = {"__model_path__": __model_path__}
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    import torch
    # import intel_extension_for_pytorch as ipex
    print(torch.__version__)
    # print(ipex.__version__)

    # os.environ["OMP_NUM_THREADS"] = str(multiprocessing.cpu_count())
    os.environ["OMP_NUM_THREADS"] = str(16)
    os.environ["OMP_SCHEDULE"] = "static"
    os.environ["OMP_PROC_BIND"] = "true"
    os.environ["OMP_PLACES"] = "cores"

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Only this extra line of code is required to use oneDNN Graph
    torch.jit.enable_onednn_fusion(True)

    # return
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

    classifier = TransformersDocumentClassifier(model_name_or_path=model_name_or_path)
    for i in range(10):
        documents = docs_from_file("~/tmp/models/mpc/158955602_1.png")
        ocr_results = load_json_file("~/tmp/models/mpc/158955602_1.json")
        words, boxes = get_words_and_boxes(ocr_results, 0)

        with TimeContext(f"Eval # {i}"):
            results = classifier.run(documents=DocumentArray(documents), words=[words], boxes=[boxes])

            for document in results:
                assert 'classification' in document.tags
                classification = document.tags['classification']
                assert 'label' in classification
                assert 'score' in classification
                print("classification", classification)
