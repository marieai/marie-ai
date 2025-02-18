from typing import Union

from marie.engine import EngineLM, get_engine
from marie.engine.config import validate_engine_or_get_default
from marie.engine.function import Function
from marie.logging_core.predefined import default_logger as logger


class LLMCall(Function):
    def __init__(self, engine: Union[str, EngineLM], system_prompt: str = None):
        """The simple LLM call function. This function will call the LLM with the input and return the response.

        :param engine: engine to use for the LLM call
        :type engine: EngineLM or str
        :param system_prompt: system prompt to use for the LLM call, default depends on the engine.
        :type system_prompt: str, optional
        """
        super().__init__()
        self.engine = validate_engine_or_get_default(engine)
        self.system_prompt = system_prompt

    def forward(self, input_variable: str) -> str:
        """
        The LLM call. This function will call the LLM with the input and return the response.

        :param input_variable: The input variable (aka prompt) to use for the LLM call.
        :type input_variable: Variable
        :return: response sampled from the LLM

        :example:
        >>> engine = get_engine("qwen_vl_3b")
        >>> llm_call = LLMCall(engine)
        >>> prompt = "What is the capital of France?"
        >>> response = llm_call(prompt)
        """
        # TODO: Should we allow default roles? It will make things less performant.
        system_prompt_value = self.system_prompt

        # Make the LLM Call
        response_text = self.engine(input_variable, system_prompt=system_prompt_value)

        logger.info(
            f"LLMCall function forward",
            extra={
                "text": f"System:{system_prompt_value}\nQuery: {input_variable}\nResponse: {response_text}"
            },
        )

        return response_text


if __name__ == "__main__":

    # some sample text
    document_context = """
        ### **Input Text:**
        Patient Greg Bugaj, born 12/31/1980
        Policy Number: 123456789
        Claim Number: 987654321
        """

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

    {document_context}
    """

    engine = get_engine("qwen2_5_3b")
    llm_call = LLMCall(engine)
    promptXX = "What is the capital of France?"
    response = llm_call(prompt)
    print(response)

    for i in range(10):
        print('---------------')
        response = llm_call(prompt)
        print(response)
