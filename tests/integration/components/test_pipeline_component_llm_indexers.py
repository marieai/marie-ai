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
        model_path="rms/corr-indexing-llm/qwen",
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
            # Test for load with multiple LLM indexers
            # "corr_page_indexer_qwen_1": {
            #     "indexer": indexer,
            #     "group": "corr_page_indexing_llm",
            # },
            # "corr_page_indexer_qwen_2": {
            #     "indexer": indexer,
            #     "group": "corr_page_indexing_llm",
            # },
            # "corr_page_indexer_qwen_3": {
            #     "indexer": indexer,
            #     "group": "corr_page_indexing_llm",
            # },
            # "corr_page_indexer_qwen_4": {
            #     "indexer": indexer,
            #     "group": "corr_page_indexing_llm",
            # },
        },
        llm_tasks=["corr_patients"]
    )

    documents = docs_from_file("~/data/tmp/227803751/227803751.tif")
    ocr_results = load_json_file("~/data/tmp/227803751/227803751.meta.json")["ocr"]

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

    document_meta = pipe_component.run(documents, context, words=words, boxes=boxes, lines=lines)

    for i, document in enumerate(documents):
        assert DOC_KEY_INDEXER in document.tags
        print(f"############ Results Page {i}:\n", to_json(document.tags))

    print(document_meta)
