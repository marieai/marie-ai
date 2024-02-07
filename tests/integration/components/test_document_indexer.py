import multiprocessing
import os

import psutil

from marie.components import TransformersDocumentIndexer
from marie.logging.profile import TimeContext
from marie.ocr.util import get_words_and_boxes
from marie.registry.model_registry import ModelRegistry
from marie.utils.docs import docs_from_file
from marie.utils.json import load_json_file


def test_named_entity_component():
    # kwargs = {"__model_path__": __model_path__}
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    import torch

    # import intel_extension_for_pytorch as ipex
    print(torch.__version__)
    # print(ipex.__version__)

    os.environ["OMP_NUM_THREADS"] = str(multiprocessing.cpu_count())
    os.environ["OMP_SCHEDULE"] = "STATIC"
    os.environ["OMP_PROC_BIND"] = "CLOSE"
    os.environ["OMP_PLACES"] = "CORES"

    # set to core-count of your CPU
    torch.set_num_threads(psutil.cpu_count(logical=False))

    # return
    model_name_or_path = "rms/layoutlmv3-large-corr-ner"
    ensure_model(model_name_or_path)

    indexer = TransformersDocumentIndexer(model_name_or_path=model_name_or_path)
    NITER = 1

    for i in range(NITER):
        documents = docs_from_file("~/tmp/models/mpc/158955602_1.png")
        ocr_results = load_json_file("~/tmp/models/mpc/158955602_1.json")
        words, boxes = get_words_and_boxes(ocr_results, 0)

        with TimeContext(f"Eval # {i}"):
            results = indexer.run(documents=documents, words=[words], boxes=[boxes])

            for document in results:
                print("results", document)


def ensure_model(model_name_or_path):
    kwargs = {
        # "__model_path__": os.path.expanduser("~/tmp/models"),
        "use_auth_token": False,
    }  # custom model path
    resolved_model_name_or_path = ModelRegistry.get(
        model_name_or_path,
        version=None,
        raise_exceptions_for_missing_entries=True,
        **kwargs,
    )
    print("resolved_model_name_or_path", resolved_model_name_or_path)
    return
