import logging
import os

import torch
from transformers import AutoTokenizer
from transformers.utils.logging import (  # Correctly import the DEBUG constant
    DEBUG,
    set_verbosity,
)
from vllm import LLM, SamplingParams

os.environ["VLLM_DISABLE_STATS"] = "true"  # Disable stats reporting for vllm
torch.set_float32_matmul_precision('high')

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

logging.basicConfig(level=logging.DEBUG)
set_verbosity(DEBUG)

# max_tokens is for the maximum length for generation.
sampling_params = SamplingParams(
    temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512
)

# Input the model name or path. Can be GPTQ or AWQ models.
# llm = LLM(model="Qwen/Qwen2.5-7B-Instruct")
llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    max_model_len=4096,  # Shorten context to fit memory
    tensor_parallel_size=1,  # Single GPU optimization
    gpu_memory_utilization=0.8,  # Prevent out-of-memory errors
)

#
# llm = LLM(
#     model="Qwen/Qwen2.5-7B-Instruct",
#     tensor_parallel_size=1,  # No parallelism (for CPU)
#     gpu_memory_utilization=0,  # Forces CPU execution
#     enforce_eager=True  # Ensures CPU execution
# )


# Prepare your prompts
prompt = "Tell me something about large language models. And how they are used in the industry."
messages = [
    {
        "role": "system",
        "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
    },
    {"role": "user", "content": prompt},
]
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

for k in range(5):
    print(f"Generating output {k + 1}...")

    outputs = llm.generate([text], sampling_params)

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


# https://docs.vllm.ai/en/latest/features/reasoning_outputs.html
# https://github.com/Duxiaoman-DI/XuanYuan/blob/main/cli_demo.py
# https://github.com/varunvasudeva1/llm-server-docs?tab=readme-ov-file#ollama-2
