from marie.api.docs import DOC_KEY_INDEXER
from marie.components.document_indexer.llm import MMLLMDocumentIndexer
from marie.logging_core.profile import TimeContext
from marie.ocr.util import get_words_and_boxes
from marie.utils.docs import docs_from_file
from marie.utils.json import load_json_file, to_json


def test_llm_document_indexer():

    model_name_or_path = "temp/indexing-llm"
    indexer = MMLLMDocumentIndexer(model_path=model_name_or_path,)

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

    with TimeContext(f"MMLLM Indexing tasks"):
        indexer.run(documents=documents, words=words, boxes=boxes, lines=lines)

    for i, document in enumerate(documents):
        assert DOC_KEY_INDEXER in document.tags
        print(f"############ Results Page {i}:\n", to_json(document.tags))


