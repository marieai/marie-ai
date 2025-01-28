import io
import os
import subprocess
import uuid
from threading import Thread

import gradio as gr
import numpy as np
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2VLForConditionalGeneration,
    TextIteratorStreamer,
)

# Model and Processor Loading (Done once at startup)
MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
# MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

# Configure BitsAndBytesConfig for 8-bit precision
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Enable 8-bit quantization
    bnb_optim='cpu',  # Optional: can specify more configuration options like bnb_optim or other tuning params
)
# bnb_config = None

# https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct/discussions/10
min_pixels = 256 * 28 * 28
max_pixels = 1280 * 28 * 28
# max_pixels = 1800*28*28
if False:
    model = (
        Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",
            quantization_config=bnb_config,
        )
        .to("cuda")
        .eval()
    )

    processor = AutoProcessor.from_pretrained(
        MODEL_ID, min_pixels=min_pixels, max_pixels=max_pixels, trust_remote_code=True
    )
if True:
    from transformers import (
        AutoProcessor,
        Qwen2_5_VLForConditionalGeneration,
        TextIteratorStreamer,
    )

    # Load model and processor
    ckpt = "Qwen/Qwen2.5-VL-7B-Instruct"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        ckpt, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to("cuda")
    processor = AutoProcessor.from_pretrained(
        ckpt, trust_remote_code=True, min_pixels=min_pixels, max_pixels=max_pixels
    )

DESCRIPTION = f"[{MODEL_ID}]"

image_extensions = Image.registered_extensions()
video_extensions = (
    "avi",
    "mp4",
    "mov",
    "mkv",
    "flv",
    "wmv",
    "mjpeg",
    "wav",
    "gif",
    "webm",
    "m4v",
    "3gp",
)


def identify_and_save_blob(blob_path):
    """Identifies if the blob is an image or video and saves it accordingly."""
    try:
        with open(blob_path, 'rb') as file:
            blob_content = file.read()

            # Try to identify if it's an image
            try:
                Image.open(
                    io.BytesIO(blob_content)
                ).verify()  # Check if it's a valid image
                extension = ".png"  # Default to PNG for saving
                media_type = "image"
            except (IOError, SyntaxError):
                # If it's not a valid image, assume it's a video
                extension = ".mp4"  # Default to MP4 for saving
                media_type = "video"

            # Create a unique filename
            filename = f"temp_{uuid.uuid4()}_media{extension}"
            with open(filename, "wb") as f:
                f.write(blob_content)

            return filename, media_type

    except FileNotFoundError:
        raise ValueError(f"The file {blob_path} was not found.")
    except Exception as e:
        raise ValueError(f"An error occurred while processing the file: {e}")


def process_file_upload(file_path):
    """Process uploaded file and return the file path and image if applicable"""
    if isinstance(file_path, str):
        if file_path.endswith(tuple([i for i, f in image_extensions.items()])):
            return file_path, Image.open(file_path)
        elif file_path.endswith(video_extensions):
            return file_path, None
        else:
            try:
                media_path, media_type = identify_and_save_blob(file_path)
                if media_type == "image":
                    return media_path, Image.open(media_path)
                return media_path, None
            except Exception as e:
                print(e)
                raise ValueError(
                    "Unsupported media type. Please upload an image or video."
                )
    return None, None


def qwen_inference(media_input, text_input=None):
    if isinstance(media_input, str):  # If it's a filepath
        media_path = media_input
        if media_path.endswith(tuple([i for i, f in image_extensions.items()])):
            media_type = "image"
        elif media_path.endswith(video_extensions):
            media_type = "video"
        else:
            try:
                media_path, media_type = identify_and_save_blob(media_input)
                print(media_path, media_type)
            except Exception as e:
                print(e)
                raise ValueError(
                    "Unsupported media type. Please upload an image or video."
                )

    print(media_path)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": media_type,
                    media_type: media_path,
                    **({"fps": 8.0} if media_type == "video" else {}),
                },
                {"type": "text", "text": text_input},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    streamer = TextIteratorStreamer(
        processor, skip_prompt=True, **{"skip_special_tokens": True}
    )
    generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=1024)

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    buffer = ""
    for new_text in streamer:
        buffer += new_text
        yield buffer


css = """
  #output {
    height: 500px; 
    overflow: auto; 
    border: 1px solid #ccc; 
  }
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Tab(label="Image Input"):
        with gr.Row():
            with gr.Column():
                input_media = gr.File(label="Upload Image to analyze", type="filepath")
                preview_image = gr.Image(label="Preview", visible=True)
                text_input = gr.Textbox(label="Text Input (optional)")
                submit_btn = gr.Button(value="Submit")
            with gr.Column():
                output_text = gr.Textbox(label="Output Text")

        input_media.change(
            fn=process_file_upload,
            inputs=[input_media],
            outputs=[input_media, preview_image],
        )

        submit_btn.click(qwen_inference, [input_media, text_input], [output_text])

demo.launch(debug=True)
