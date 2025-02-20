import os
from typing import Dict, List, Union

from PIL import Image

from marie.engine import EngineLM, get_engine
from marie.engine.config import validate_engine_or_get_default
from marie.engine.engine_utils import as_bytes
from marie.engine.function import Function, FunctionReturnType
from marie.engine.guided import GuidedMode
from marie.logging_core.predefined import default_logger as logger


class MultimodalLLMCall(Function):
    def __init__(self, engine: Union[str, EngineLM], system_prompt: str = None):
        """The MultiModalLM call function. This function will call the LLM with the input (image) and return the response.

        :param engine: engine to use for the LLM call
        :type engine: EngineLM
        :param system_prompt: system prompt to use for the LLM call, default depends on the engine.
        :type system_prompt: Variable, optional
        """
        super().__init__()
        self.engine = validate_engine_or_get_default(engine)
        self.system_prompt = system_prompt

    def forward(
        self,
        inputs: Union[
            List[List[Union[str, bytes, Image.Image]]],
            List[Union[str, bytes, Image.Image]],
        ],
        guided_mode: GuidedMode = None,
        guided_params: Union[List[str], str, Dict] = None,
    ) -> FunctionReturnType:
        """
        The LLM call. This function will call the LLM with the input and return the response.

        :param inputs: list of input variables to the multimodal LLM call. One is an image and the second one is text
        :param guided_params: guided parameters to use for the LLM call
        :param guided_mode: guided mode to use for the LLM call
        :return: response sampled from the LLM

        :example:
        >>> engine = get_engine("qwen_vl_3b")
        >>> target_image = "A byte representation of the image"
        >>> prompt = "What is the capital of France?"
        >>> response = MultimodalLLMCall(engine)([target_image, prompt])
        """

        def validate_input(input_items: List[Union[str, bytes, Image.Image]]):
            for variable in input_items:
                if not isinstance(variable, (str, bytes, Image.Image)):
                    raise ValueError(
                        f"MultimodalLLMCall only accepts str, bytes or PIL Image, got {type(variable)}"
                    )

        if isinstance(inputs[0], list):
            for sublist in inputs:
                validate_input(sublist)
        else:
            validate_input(inputs)

        system_prompt_value = self.system_prompt

        # Make the LLM Call
        response_text = self.engine(
            inputs,
            system_prompt=system_prompt_value,
            guided_mode=guided_mode,
            guided_params=guided_params,
        )

        logger.info(
            f"MultimodalLLMCall function forward",
            extra={
                "text": f"System:{system_prompt_value}\nQuery: {inputs}\nResponse: {response_text}"
            },
        )

        return response_text


from pydantic import BaseModel


class KeyValuePair(BaseModel):
    key: str
    value: str


class Pairs(BaseModel):
    answers: List[KeyValuePair]


if __name__ == "__main__":
    print(Pairs.model_json_schema())

    prompt = f"""
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

    image_path = os.path.expanduser("~/tmp/demo/159861652_2.png")
    image = as_bytes(image_path)
    image = Image.open(image_path).convert("RGB")
    regex_grammar = r'(?:(?P<key>[A-Za-z0-9 _\-\(\)]+): (?P<value>[^;]+);\s*)+|"No key-value pairs found."\s*'

    # https://blog.mlc.ai/2024/11/22/achieving-efficient-flexible-portable-structured-generation-with-regex_grammar
    # https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md
    structured_kv_grammar = r"""
    root    ::= pairs | no_pairs
    pairs   ::= (pair)+
    pair    ::= key ": " value ";"
    key     ::= [^:;]+
    value   ::= [^;]+ | "[MISSING]"
    no_pairs::= "No key-value pairs found."
    """

    engine = get_engine("qwen2_5_vl_7b")
    llm_call = MultimodalLLMCall(engine)
    response = llm_call(
        [image, prompt], guided_mode=GuidedMode.REGEX, guided_params=regex_grammar
    )
    print(response)

    if False:
        print("Batched request")
        batch = [[image, prompt] for _ in range(5)]
        response = llm_call(batch)

        for r in response:
            print('---------------')
            print(r)

        if False:
            for i in range(5):
                print('---------------')
                response = llm_call([image, prompt])
                print(response)
