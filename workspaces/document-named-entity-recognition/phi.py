import logging
import os
import subprocess
import time
from threading import Thread

import gradio as gr
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)
from transformers.utils.logging import (  # Correctly import the DEBUG constant
    DEBUG,
    set_verbosity,
)

logging.basicConfig(level=logging.DEBUG)
set_verbosity(DEBUG)

torch.set_float32_matmul_precision('high')

# subprocess.run(
#     "pip install flash-attn --no-build-isolation",
#     env={"FLASH_ATTENTION_SKIP_CUDA_BUILD": "TRUE"},
#     shell=True,
# )

# token = os.environ["HF_TOKEN"]
token = None
# "microsoft/phi-4"
# https://huggingface.co/blog/smolvlm

model_id = "microsoft/Phi-3.5-mini-instruct"  # 10GB
# model_id = "Qwen/Qwen2-7B-Instruct"  # 16GB
# model_id = "Qwen/Qwen2-2B-Instruct"  # 16GB
# model_id = "google/gemma-7b-it"  # 16GB
# model_id = "llava-hf/llava-1.5-7b-hf"  # 16GB
model_id = "Qwen/Qwen2-7B-Instruct"  # 16GB
model_id = "google/gemma-7b-it"  # 16GB
model_id = "Qwen/Qwen2-7B-Instruct"  # 16GB 4-bit quantization 8GBVRAM
model_id = "Qwen/Qwen2.5-7B-Instruct"  # 16GB 4-bit quantization 8GBVRAM
# model_id = "unsloth/Qwen2.5-7B-Instruct"  # 16GB 4-bit quantization 8GBVRAM
# model_id = "Qwen/Qwen2.5-14B-Instruct"  # 13GB vram
# model_id = "Qwen/Qwen2.5-0.5B-Instruct"  # 16GB 4-bit quantization 8GBVRAM
# model_id = "google/gemma-7b-it"  # 16GB
# model_id = "google/gemma-2-2b-it" # 7GB

# model_id = "mistralai/Mistral-7B-v0.3" # mistral-7B-Instruct-v0.3.

# model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     token=token,
#     trust_remote_code=True,
#     torch_dtype=torch.bfloat16
# )

# https://huggingface.co/docs/transformers/main/en/quantization/hqq
from transformers import HqqConfig

# Method 1: all linear layers will use the same quantization config
quant_config = HqqConfig(nbits=8, group_size=64)

# BitsAndBytesConfig int-4 config
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,  # Enable 8-bit quantization
    bnb_optim='cpu',  # Optional: can specify more configuration options like bnb_optim or other tuning params
)
bnb_config = None

# # Load model in 8-bit precision using bitsandbytes
# model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     device_map="auto",  # Automatically use available GPUs
#     torch_dtype=torch.bfloat16,
#     trust_remote_code=True,
#     quantization_config=bnb_config,
#     load_in_4bit=True,  # Enable 8-bit quantization
# )
# # bnb_config = None

# Load model in 8-bit precision using bitsandbytes
if False:
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",  # Automatically use available GPUs
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        quantization_config=bnb_config,
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_use_double_quant=True,
    bnb_8bit_compute_dtype=torch.bfloat16,
)

# bnb_config = None

# # Load model in 4/8-bit precision using bitsandbytes
# model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     device_map="auto",  # Automatically use available GPUs
#     torch_dtype=torch.bfloat16,
#     trust_remote_code=True,
#     quantization_config=bnb_config,
# )
model = None
if False:
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",  # Automatically use available GPUs
        torch_dtype=torch.float16,
        trust_remote_code=True,
        quantization_config=quant_config,
    )
    model = torch.compile(model)
    # model = torchao.autoquant(torch.compile(model, mode='max-autotune'))
    tok = AutoTokenizer.from_pretrained(model_id, token=token)

if False:
    import torch
    from unsloth import FastLanguageModel

    max_seq_length = 2048  # Choose any! We auto support RoPE Scaling internally!
    dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = False  # Use 4bit quantization to reduce memory usage. Can be False.

    model, tok = FastLanguageModel.from_pretrained(
        # Can select any from the below:
        # "unsloth/Qwen2.5-0.5B", "unsloth/Qwen2.5-1.5B", "unsloth/Qwen2.5-3B"
        # "unsloth/Qwen2.5-14B",  "unsloth/Qwen2.5-32B",  "unsloth/Qwen2.5-72B",
        # And also all Instruct versions and Math. Coding verisons!
        model_name=model_id,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    model = FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

terminators = [
    tok.eos_token_id,
]

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(device)}")
else:
    device = torch.device("cpu")
    print("Using CPU")


# model = model.to(device)
# Dispatch Errors


def chat(message, history, temperature, do_sample, max_tokens):
    chat = []
    for item in history:
        chat.append({"role": "user", "content": item[0]})
        if item[1] is not None:
            chat.append({"role": "assistant", "content": item[1]})
    chat.append({"role": "user", "content": message})

    # Tokenize the input and measure tokenization speed
    start_time = time.time()
    messages = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    model_inputs = tok([messages], return_tensors="pt").to(device)
    tokenization_time = time.time() - start_time
    num_input_tokens = model_inputs.input_ids.size(1)

    streamer = TextIteratorStreamer(
        tok, timeout=20.0, skip_prompt=True, skip_special_tokens=True
    )

    max_tokens = 4096
    generate_kwargs = dict(
        model_inputs,
        streamer=streamer,
        max_new_tokens=max_tokens,
        do_sample=False,
        temperature=temperature,
        eos_token_id=terminators,
    )

    # Generate output using the model
    if True:
        start_time = time.perf_counter()
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=4096,
            # max_length=inputs.input_ids.size(1) + 50,  # Input length + max tokens
            # length_penalty=1.5,  # Penalizes longer generations
            num_beams=1,  # 1 = greedy search, > 1 = beam search
            do_sample=False,
            temperature=0.0,  # 1.0 = no change, < 1.0 = more conservative, > 1.0 = more random
            eos_token_id=terminators,  # Stops generation when special token is reached
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        output_text = tok.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        end_time = time.perf_counter()
        generation_time = end_time - start_time
        print("Execution time:", generation_time)

        # Count the total number of generated tokens
        num_output_tokens = len(generated_ids_trimmed[0])

        # Calculate token processing rates
        tokenization_rate = (
            num_input_tokens / tokenization_time if tokenization_time > 0 else 0
        )
        generation_rate = (
            num_output_tokens / generation_time if generation_time > 0 else 0
        )

        # Print results
        print(f"Number of input tokens: {num_input_tokens}")
        print(f"Number of output tokens: {num_output_tokens}")
        print(
            f"Tokenization time: {tokenization_time:.4f} seconds ({tokenization_rate:.2f} tokens/second)"
        )
        print(
            f"Generation time: {generation_time:.4f} seconds ({generation_rate:.2f} tokens/second)"
        )

        print("-" * 50)
        print(output_text)
        splitted = output_text[0].split("\n")
        for s in splitted:
            print(s)
        print("-" * 50)

    if temperature == 0:
        generate_kwargs["do_sample"] = False

    generate_kwargs["do_sample"] = False

    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    partial_text = ""
    for new_text in streamer:
        partial_text += new_text
        yield partial_text

    yield partial_text


demo = gr.ChatInterface(
    fn=chat,
    examples=[["Document extraction."]],
    # multimodal=False,
    additional_inputs_accordion=gr.Accordion(
        label="⚙️ Parameters", open=False, render=False
    ),
    additional_inputs=[
        gr.Slider(
            minimum=0, maximum=1, step=0.1, value=0.9, label="Temperature", render=False
        ),
        gr.Checkbox(label="Sampling", value=True),
        gr.Slider(
            minimum=256,
            maximum=4096,
            step=1,
            value=1024,
            label="Max new tokens",
            render=False,
        ),
    ],
    stop_btn="Stop Generation",
    title="Extract With LLMs",
    description=f"Now Running [{model_id}]",
)
demo.launch()
