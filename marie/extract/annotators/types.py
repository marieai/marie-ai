from typing import Type, Union

from marie.extract.annotators.faiss_hybrid_annotator import FaissHybridAnnotator
from marie.extract.annotators.llm_annotator import LLMAnnotator
from marie.extract.annotators.llm_table_annotator import LLMTableAnnotator
from marie.extract.annotators.regex_annotator import RegexAnnotator

AnnotatorClassType = Type[
    Union[LLMAnnotator, LLMTableAnnotator, FaissHybridAnnotator, RegexAnnotator]
]

ANNOTATOR_NAME_TO_ANNOTATOR_TYPE: dict[str, AnnotatorClassType] = {
    "llm": LLMAnnotator,
    "llm_table": LLMTableAnnotator,
    "embedding": FaissHybridAnnotator,
    "regex": RegexAnnotator,
    # IMAGE annotators should be added here
    # "image": ImageAnnotator,
}
