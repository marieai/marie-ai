from marie.api.docs import DOC_KEY_INDEXER
from marie.components.document_indexer.llm import MMLLMDocumentIndexer
from marie.logging_core.profile import TimeContext
from marie.ocr.util import get_words_and_boxes
from marie.pipe import PipelineContext
from marie.pipe.llm_indexer import LLMIndexerPipelineComponent
from marie.utils.docs import docs_from_file
from marie.utils.json import load_json_file, to_json


def test_llm_pipeline_component():
    indexer = MMLLMDocumentIndexer(
        model_path="CPREFIX/corr-indexing-llm/qwen3",
        devices=["cuda"],
        ocr_engine=None,
    )

    pipe_component = LLMIndexerPipelineComponent(
        "test_llm_pipeline_component",
        {
            "corr_page_indexer_qwen": {  # Config["page_indexer"]["name" | "model_name_or_path"]
                "indexer": indexer,
                "group": "corr_page_indexing_llm",
            },

        }
    )

    documents = docs_from_file("~/data/tmp/0000000/0000000.tif")
    ocr_results = load_json_file("~/data/tmp/0000000/0000000.meta.json")["ocr"]

    words = []
    boxes = []
    lines = []
    for page_idx, doc in enumerate(documents):
        page_words, page_boxes, page_lines = get_words_and_boxes(ocr_results, page_idx, include_lines=True)
        words.append(page_words)
        boxes.append(page_boxes)
        lines.append(page_lines)
        # TODO: add page level classifications to doc

    context = PipelineContext(pipeline_id="test_llm_pipeline")
    context["metadata"] = {}

    with TimeContext(f"MMLLM Indexing tasks"):
        document_meta = pipe_component.predict(documents, context, words=words, boxes=boxes, lines=lines)

    for i, document in enumerate(documents):
        assert DOC_KEY_INDEXER in document.tags
        assert all(task_name in document.tags[DOC_KEY_INDEXER]
                   for task_name, task in indexer.task_map.items()
                   if task.store_results)
        print(f"############ Results Page {i}:\n", to_json(document.tags))

    print(document_meta)
