import os
from typing import List, Union

import pytest
from PIL import Image

from marie.engine import EngineLM, get_engine
from marie.engine.engine_utils import as_bytes
from marie.engine.multimodal_ops import MultimodalLLMCall

DEFAULT_PROMPT = f"""
### Task: Extract Key-Value Pairs

Extract structured key-value pairs from the given text while maintaining accuracy and formatting.

### **Rules:**
1 **Extract only key-value pairs** — Do not include explanations, summaries, or extra text.  
2 **Preserve key names exactly as they appear** — No modifications, abbreviations, or rewording.  
3 **Ensure values are extracted accurately** — No truncation or paraphrasing.  
4 **If a key has no value, return:** `KeyName: [MISSING]`  
5 **If no key-value pairs are found, return exactly:** `"No key-value pairs found."`  

### **Strict Output Format:**
Key1: Value1;
Key2: Value2;
Key3: Value3;
...

Your response **must contain only** the extracted key-value pairs in the format above. No additional text.
"""


@pytest.fixture(scope="module", params=["qwen2_5_vl_7b_awq"])
def engine_setup(request):
    """
    Creates a single engine instance for each model in params.
    Yields:
        A dict containing the engine instance and the model name.
    """
    model_name = request.param
    engine_instance = get_engine(model_name)
    yield {
        "engine": engine_instance,
        "model_name": model_name
    }


@pytest.fixture
def test_image() -> Image.Image:
    image_path = os.path.expanduser("~/tmp/demo/159861652_2.png")
    return Image.open(image_path).convert("RGB")


def test_single_multimodal_call_with_no_guidance(engine_setup, test_image):
    """
    Test a single multimodal call with no guidance for each model.
    """
    engine_instance: EngineLM = engine_setup["engine"]
    llm_call = MultimodalLLMCall(engine_instance)
    response = llm_call([test_image, DEFAULT_PROMPT])
    print(response)
    assert isinstance(response, str), "Expected a string response."


def test_single_multimodal_call_with_grammar(engine_setup, test_image):
    """
    Test a single multimodal call for each model using various grammar settings.
    """
    engine_instance: EngineLM = engine_setup["engine"]
    llm_call = MultimodalLLMCall(engine_instance)

    structured_kv_grammar = r"""
    root    ::= pairs | no_pairs
    pairs   ::= (pair)+
    pair    ::= key ": " value ";"
    key     ::= [^:;]+
    value   ::= [^;]+ | "[MISSING]"
    no_pairs::= "No key-value pairs found."
    """

    response = llm_call([test_image, DEFAULT_PROMPT], guided_grammar=structured_kv_grammar)
    print(response)

    assert isinstance(response, str), "Expected a string response."


def test_single_multimodal_call_with_regex(engine_setup, test_image):
    """
    Test a single multimodal call for each model using various grammar settings.
    """
    engine_instance: EngineLM = engine_setup["engine"]
    llm_call = MultimodalLLMCall(engine_instance)
    regex_grammar = r'(?:(?P<key>[A-Za-z0-9 _\-\(\)]+): (?P<value>[^;]+);\s*)+|"No key-value pairs found."\s*'
    response = llm_call([test_image, DEFAULT_PROMPT], guided_regex=regex_grammar)
    print(response)

    assert isinstance(response, str), "Expected a string response."


def test_single_multimodal_call_with_pydantic(engine_setup, test_image):
    """
    Test a single multimodal call for each model using various grammar settings.
    """

    from pydantic import BaseModel
    class KeyValuePair(BaseModel):
        key: str
        value: str

    class Pairs(BaseModel):
        answers: List[KeyValuePair]

    engine_instance: EngineLM = engine_setup["engine"]
    llm_call = MultimodalLLMCall(engine_instance)
    response = llm_call([test_image, DEFAULT_PROMPT], guided_json=Pairs)
    print(response)

    assert isinstance(response, str), "Expected a string response."


def test_batched_multimodal_call(engine_setup, test_image):
    """
    Test a batched call for each model.
    """
    engine_instance: EngineLM = engine_setup["engine"]
    llm_call = MultimodalLLMCall(engine_instance)
    prompt = DEFAULT_PROMPT

    batch = [
        [test_image, prompt],
        [test_image, prompt],
    ]
    responses = llm_call(batch)

    print(responses)

    assert isinstance(responses, list), "Expected a list of responses."
    assert all(isinstance(r, str) for r in responses), "All responses must be strings."
