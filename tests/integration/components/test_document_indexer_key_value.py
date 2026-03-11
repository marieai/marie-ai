import multiprocessing
import os

import psutil
import torch
from jinja2 import Environment, FileSystemLoader

from marie.components.document_indexer.transformers_seq2seq import (
    TransformersSeq2SeqDocumentIndexer,
)
from marie.constants import __config_dir__
from marie.logging_core.profile import TimeContext
from marie.ocr.util import get_words_and_boxes, meta_to_text
from marie.registry.model_registry import ModelRegistry
from marie.utils.docs import docs_from_file
from marie.utils.json import load_json_file


def ensure_model(model_name_or_path: str) -> str:
    """Ensure model is available locally"""
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
    return resolved_model_name_or_path


def test_transformer_document_indexer():
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    os.environ["OMP_NUM_THREADS"] = str(multiprocessing.cpu_count())
    os.environ["OMP_SCHEDULE"] = "STATIC"
    os.environ["OMP_PROC_BIND"] = "CLOSE"
    os.environ["OMP_PLACES"] = "CORES"

    # set to core-count of your CPU
    torch.set_num_threads(psutil.cpu_count(logical=False))

    model_name_or_path = "marie/key-value-relation"
    resolved_model_path = ensure_model(model_name_or_path)
    print("resolved_model_path", resolved_model_path)

    indexer = TransformersSeq2SeqDocumentIndexer(
        model_name_or_path=resolved_model_path, ocr_engine=None
    )

    NITER = 4
    print(__config_dir__)
    env = Environment(loader=FileSystemLoader(os.path.join(__config_dir__, "zoo/prompt-templates/relation-extraction")))
    template = env.get_template("inference_prompt.txt.j2")

    for i in range(NITER):
        documents = docs_from_file("~/assets/section-frag/001/001.png")
        ocr_results = load_json_file("~/assets/section-frag/001/result.json")
        words, boxes = get_words_and_boxes(ocr_results, 0)
        text = meta_to_text(ocr_results)
        print('-' * 50)
        default_prompt = template.render(text=text)
        print(default_prompt)

        with TimeContext(f"Eval # {i}"):
            results = indexer.run(documents=documents, words=[words], boxes=[boxes], prompts=[default_prompt])

            for document in results:
                print("results", document.tags)
                indexer_result = document.tags['indexer']
                kv = indexer_result['kv']
                print("indexer", kv)
