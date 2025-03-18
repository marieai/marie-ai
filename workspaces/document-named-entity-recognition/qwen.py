import ast
import io
import json
import os
import random
import subprocess
import uuid
from threading import Thread

import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageColor, ImageDraw, ImageFont
from qwen_vl_utils import process_vision_info, smart_resize
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2VLForConditionalGeneration,
    TextIteratorStreamer,
)

additional_colors = [
    colorname for (colorname, colorcode) in ImageColor.colormap.items()
]

MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

# Configure BitsAndBytesConfig for 8-bit precision
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Enable 8-bit quantization
)
# bnb_config = None

# https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct/discussions/10
min_pixels = 512 * 28 * 28
max_pixels = 2048 * 28 * 28

from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    TextIteratorStreamer,
)

# Load model and processor
# ckpt = "Qwen/Qwen2.5-VL-3B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    # quantization_config=bnb_config,
    attn_implementation="flash_attention_2",
    quantization_config=bnb_config,
).to("cuda")

processor = AutoProcessor.from_pretrained(
    MODEL_ID, trust_remote_code=True, min_pixels=min_pixels, max_pixels=max_pixels
)

DESCRIPTION = f"[{MODEL_ID}]"

image_extensions = Image.registered_extensions()
video_extensions = ()


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
                raise ValueError("The file is not a valid image.")

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


def parse_json(json_output):
    # Parsing out the markdown fencing
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(
                lines[i + 1 :]
            )  # Remove everything before "```json"
            json_output = json_output.split("```")[
                0
            ]  # Remove everything after the closing "```"
            break  # Exit the loop once "```json" is found
    return json_output


def plot_bounding_boxes(im, bounding_boxes, input_width, input_height):
    """
    Plots bounding boxes on an image with markers for each a name, using PIL, normalized coordinates, and different colors.

    Args:
        img_path: The path to the image file.
        bounding_boxes: A list of bounding boxes containing the name of the object
         and their positions in normalized [y1 x1 y2 x2] format.
    """

    # Load the image
    img = im
    width, height = img.size
    print(img.size)
    draw = ImageDraw.Draw(img)

    # Define a list of colors
    colors = [
        'red',
        'green',
        'blue',
        'yellow',
        'orange',
        'pink',
        'purple',
        'brown',
        'gray',
        'beige',
        'turquoise',
        'cyan',
        'magenta',
        'lime',
        'navy',
        'maroon',
        'teal',
        'olive',
        'coral',
        'lavender',
        'violet',
        'gold',
        'silver',
    ] + additional_colors

    # Parsing out the markdown fencing
    bounding_boxes = parse_json(bounding_boxes)
    font = ImageFont.truetype("NotoSansCJK-Regular.ttc", size=14)
    try:
        json_output = ast.literal_eval(bounding_boxes)
    except Exception as e:
        end_idx = bounding_boxes.rfind('"}') + len('"}')
        truncated_text = bounding_boxes[:end_idx] + "]"
        json_output = ast.literal_eval(truncated_text)

    # Iterate over the bounding boxes
    for i, bounding_box in enumerate(json_output):
        # Select a color from the list
        color = colors[i % len(colors)]

        # Convert normalized coordinates to absolute coordinates
        abs_y1 = int(bounding_box["bbox_2d"][1] / input_height * height)
        abs_x1 = int(bounding_box["bbox_2d"][0] / input_width * width)
        abs_y2 = int(bounding_box["bbox_2d"][3] / input_height * height)
        abs_x2 = int(bounding_box["bbox_2d"][2] / input_width * width)

        if abs_x1 > abs_x2:
            abs_x1, abs_x2 = abs_x2, abs_x1

        if abs_y1 > abs_y2:
            abs_y1, abs_y2 = abs_y2, abs_y1

        # Draw the bounding box
        draw.rectangle(((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=4)

        # Draw the text
        if "label" in bounding_box:
            draw.text(
                (abs_x1 + 8, abs_y1 + 6), bounding_box["label"], fill=color, font=font
            )

    # Display the image
    img.show()


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
    image = Image.open(media_path)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": media_type,
                    media_type: media_path,
                },
                {"type": "text", "text": text_input},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=[image],
        videos=None,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    input_height = inputs['image_grid_thw'][0][1] * 14
    input_width = inputs['image_grid_thw'][0][2] * 14

    print('input_height =', input_height)
    print('input_width =', input_width)

    streamer = TextIteratorStreamer(
        processor, skip_prompt=True, **{"skip_special_tokens": True}
    )
    generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=2048)

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    buffer = ""
    for new_text in streamer:
        buffer += new_text
        yield buffer

    print("done")
    print(buffer)

    json_output = parse_json(buffer)
    print(json_output)
    plot_bounding_boxes(image, buffer, input_width, input_height)


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

# https://github.com/QwenLM/Qwen2.5-VL/issues/721
# https://github.com/QwenLM/Qwen2.5-VL/blob/main/cookbooks/spatial_understanding.ipynb
# Outline the position of  each table column header and output all the coordinates in JSON format with table header text as label.

# https://www.linkedin.com/pulse/advancing-document-ai-table-column-detection-made-simple-vaghela-px6ac
# https://github.com/VikParuchuri/marker/tree/master/marker
# python table_recognition.py --debug --images /home/greg/dev/rms/grapnel-g5/assets/TID-100985/ANNOTATED/OUT/tables/fragments/5_0-1200.png --skip_table_detection
