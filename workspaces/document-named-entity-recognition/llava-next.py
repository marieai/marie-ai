import re
import time
from threading import Thread

import gradio as gr
import torch
from PIL import Image
from transformers import (
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
    TextIteratorStreamer,
)

processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)
model.to("cuda:0")


def bot_streaming(message, history):
    print(message)
    if message["files"]:
        image = message["files"][-1]["path"]
    else:
        # if there's no image uploaded for this turn, look for images in the past turns
        # kept inside tuples, take the last one
        for hist in history:
            if type(hist[0]) == tuple:
                image = hist[0][0]

    if image is None:
        gr.Error("You need to upload an image for LLaVA to work.")
    prompt = f"[INST] <image>\n{message['text']} [/INST]"
    image = Image.open(image).convert("RGB")
    inputs = processor(prompt, image, return_tensors="pt").to("cuda:0")

    streamer = TextIteratorStreamer(processor, **{"skip_special_tokens": True})
    generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=100)
    generated_text = ""

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    text_prompt = f"[INST]  \n{message['text']} [/INST]"

    buffer = ""
    for new_text in streamer:
        buffer += new_text

        generated_text_without_prompt = buffer[len(text_prompt) :]
        time.sleep(0.04)
        yield generated_text_without_prompt


demo = gr.ChatInterface(
    fn=bot_streaming,
    title="LLaVA NeXT",
    examples=[
        {"text": "What is on the flower?", "files": ["./bee.jpg"]},
        {"text": "How to make this pastry?", "files": ["./baklava.png"]},
    ],
    description="Try [LLaVA NeXT](https://huggingface.co/docs/transformers/main/en/model_doc/llava_next) in this demo (more specifically, the [Mistral-7B variant](https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf)). Upload an image and start chatting about it, or simply try one of the examples below. If you don't upload an image, you will receive an error.",
    stop_btn="Stop Generation",
    multimodal=True,
)
demo.launch(debug=True)
