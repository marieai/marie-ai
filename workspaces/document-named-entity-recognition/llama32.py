import time
from threading import Thread

import gradio as gr
import requests
import torch
from gradio import FileData
from PIL import Image
from transformers import (
    AutoProcessor,
    MllamaForConditionalGeneration,
    TextIteratorStreamer,
)

ckpt = "meta-llama/Llama-3.2-11B-Vision-Instruct"
model = MllamaForConditionalGeneration.from_pretrained(
    ckpt, torch_dtype=torch.bfloat16
).to("cuda")
processor = AutoProcessor.from_pretrained(ckpt)


def bot_streaming(message, history, max_new_tokens=4096):

    txt = message["text"]
    ext_buffer = f"{txt}"

    messages = []
    images = []

    for i, msg in enumerate(history):
        if isinstance(msg[0], tuple):
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": history[i + 1][0]},
                        {"type": "image"},
                    ],
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": history[i + 1][1]}],
                }
            )
            images.append(Image.open(msg[0][0]).convert("RGB"))
        elif isinstance(history[i - 1], tuple) and isinstance(msg[0], str):
            # messages are already handled
            pass
        elif isinstance(history[i - 1][0], str) and isinstance(
            msg[0], str
        ):  # text only turn
            messages.append(
                {"role": "user", "content": [{"type": "text", "text": msg[0]}]}
            )
            messages.append(
                {"role": "assistant", "content": [{"type": "text", "text": msg[1]}]}
            )

    # add current message
    if len(message["files"]) == 1:

        if isinstance(message["files"][0], str):  # examples
            image = Image.open(message["files"][0]).convert("RGB")
        else:  # regular input
            image = Image.open(message["files"][0]["path"]).convert("RGB")
        images.append(image)
        messages.append(
            {
                "role": "user",
                "content": [{"type": "text", "text": txt}, {"type": "image"}],
            }
        )
    else:
        messages.append({"role": "user", "content": [{"type": "text", "text": txt}]})

    texts = processor.apply_chat_template(messages, add_generation_prompt=True)

    if images == []:
        inputs = processor(text=texts, return_tensors="pt").to("cuda")
    else:
        inputs = processor(text=texts, images=images, return_tensors="pt").to("cuda")
    streamer = TextIteratorStreamer(
        processor, skip_special_tokens=True, skip_prompt=True
    )

    generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=max_new_tokens)
    generated_text = ""

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    buffer = ""

    for new_text in streamer:
        buffer += new_text
        generated_text_without_prompt = buffer
        time.sleep(0.01)
        yield buffer


demo = gr.ChatInterface(
    fn=bot_streaming,
    title="Multimodal Llama",
    textbox=gr.MultimodalTextbox(max_lines=250, max_plain_text_length=25000),
    additional_inputs=[
        gr.Slider(
            minimum=10,
            maximum=4096,
            value=4096,
            step=10,
            label="Maximum number of new tokens to generate",
        )
    ],
    cache_examples=False,
    description="Try Multimodal Llama by Meta with transformers in this demo. Upload an image, and start chatting about it, or simply try one of the examples below. To learn more about Llama Vision, visit [our blog post](https://huggingface.co/blog/llama32). ",
    stop_btn="Stop Generation",
    fill_height=True,
    multimodal=True,
)

demo.launch(debug=True)
