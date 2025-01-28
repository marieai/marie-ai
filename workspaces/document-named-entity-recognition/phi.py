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

# subprocess.run(
#     "pip install flash-attn --no-build-isolation",
#     env={"FLASH_ATTENTION_SKIP_CUDA_BUILD": "TRUE"},
#     shell=True,
# )

# token = os.environ["HF_TOKEN"]
token = None
# "microsoft/phi-4"

model_id = "microsoft/Phi-3.5-mini-instruct"  # 10GB
# model_id = "Qwen/Qwen2-7B-Instruct"  # 16GB
# model_id = "Qwen/Qwen2-2B-Instruct"  # 16GB
# model_id = "google/gemma-7b-it"  # 16GB
# model_id = "llava-hf/llava-1.5-7b-hf"  # 16GB
# model_id = "google/gemma-2-2b-it" # 7GB

# model_id = "mistralai/Mistral-7B-v0.3" # mistral-7B-Instruct-v0.3.

# model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     token=token,
#     trust_remote_code=True,
#     torch_dtype=torch.bfloat16
# )

# Configure BitsAndBytesConfig for 8-bit precision
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,  # Enable 8-bit quantization
    bnb_optim='cpu',  # Optional: can specify more configuration options like bnb_optim or other tuning params
)
bnb_config = None

# Load model in 8-bit precision using bitsandbytes
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",  # Automatically use available GPUs
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    quantization_config=bnb_config,
)

tok = AutoTokenizer.from_pretrained(model_id, token=token)
terminators = [
    tok.eos_token_id,
]

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(device)}")
else:
    device = torch.device("cpu")
    print("Using CPU")


model = model.to(device)
# Dispatch Errors


def chat(message, history, temperature, do_sample, max_tokens):
    chat = []
    for item in history:
        chat.append({"role": "user", "content": item[0]})
        if item[1] is not None:
            chat.append({"role": "assistant", "content": item[1]})
    chat.append({"role": "user", "content": message})
    messages = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    model_inputs = tok([messages], return_tensors="pt").to(device)
    streamer = TextIteratorStreamer(
        tok, timeout=20.0, skip_prompt=True, skip_special_tokens=True
    )
    generate_kwargs = dict(
        model_inputs,
        streamer=streamer,
        max_new_tokens=max_tokens,
        do_sample=False,
        temperature=temperature,
        eos_token_id=terminators,
    )

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
