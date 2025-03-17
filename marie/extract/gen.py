import json
import os
import time

import torch
from vllm import LLM
from vllm.sampling_params import SamplingParams

from marie.models.icr.modules.prediction import device


def generate_prompt(labels, text):
    prompt = """
    Extract the entities for the following labels from the given text and provide the results in JSON format.
    - Entities must be extracted exactly as mentioned in the text.
    - Return each entity under its label without creating new labels.
    - Provide a list of entities for each label, ensuring that if no entities are found for a label, an empty list is returned.
    - Accuracy and relevance in your responses are key.

    labels:"""

    for label in labels:
        prompt += f"\n- {label}"

    prompt += """
    JSON Structure:
    {
    """

    for label in labels:
        prompt += f'"{label}": [],\n'

    prompt += "}\n\n"
    prompt += "\n\nTEXT:\n"
    prompt += text
    prompt += "\nEND_OF_TEXT"
    return prompt


# LLaMA3 and Mixtral
def llvm_ner(model_name, prompt):
    # https://www.reddit.com/r/LocalLLaMA/comments/18rryf1/why_is_noone_finetuning_something_like_t5/
    # https://docs.vllm.ai/en/latest/quantization/auto_awq.html
    # https://lightning.ai/lightning-ai/studios/optimized-llm-inference-api-for-mistral-7b-using-vllm?tab=overview

    model_name = "mistralai/Mistral-Nemo-Instruct-2407"
    # mistralai/Mistral-7B-v0.1
    tensor_parallel_size = int(os.environ.get("DEVICES", "1"))
    sampling_params = SamplingParams(max_tokens=200)  # , temperature=0.5

    llm = LLM(
        "mistralai/Mistral-7B-v0.1",
        tensor_parallel_size=tensor_parallel_size,
        dtype=torch.bfloat16,
        gpu_memory_utilization=1.0,
    )

    # llm = LLM(
    #     "TheBloke/Mistral-7B-v0.1-AWQ",
    #     dtype=torch.float16,
    #     quantization="AWQ",
    # )

    # print(
    #     llm.generate("This is me warming up the model", sampling_params=sampling_params)
    # )
    print(f"running inference through  prompt.")
    #
    # llm = vLLM(
    #     model=model_name,
    #     tokenizer_mode="mistral",
    #     load_format="safetensors",
    #     device="cuda",
    # )

    messages = [
        {
            "role": "system",
            "content": "Entity Recognition Expert",
            "role": "user",
            "content": prompt,
        },
    ]

    chat_template = """
    {% for message in messages %}
    {{ message.role }}: {{ message.content }}
    {% endfor %}
    """

    print(messages)
    try:
        t0 = time.perf_counter()

        res = llm.chat(
            messages=messages,
            sampling_params=sampling_params,
            chat_template=chat_template,
        )

        output = res[0]
        print("Response Object:", output)
        t1 = time.perf_counter()
        print(
            f"time : {t1 - t0}  , tokens_generated : {len(output.outputs[0].token_ids)}"
        )
        print("Request ID:", output.request_id)
        print("Prompt:", output.prompt)
        if output.outputs:
            for response in output.outputs:
                print("Response:", response.text)

    except Exception as e:
        print(f"Error during chat: {e}")
        res = {}

    return res


# Sample Text
labels = ["SKILLS", "NAME", "CERTIFICATE", "HOBBIES", "COMPANY", "UNIVERSITY"]

text = "I went to the University of California, Berkeley and graduated with a degree in Computer Science. I have a certificate in Data Science from Coursera. I have worked at Google for 5 years. My hobbies include playing the guitar and reading books. I am proficient in Python, Java, and C++."

# Results
# llama_result = ollama_ner('llama3', text)
prompt = generate_prompt(labels, text)
mixtral_result = llvm_ner("mixtral", prompt)

# print("LLaMA3 Result:", llama_result)
print("Mixtral Result:", mixtral_result)
