import ast
import base64
import io
import os
import uuid
from typing import List

import gradio as gr
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image, ImageColor, ImageDraw, ImageFont

load_dotenv()  # Loads environment variables from a .env

api_key = os.getenv("OPENAI_API_KEY", "EMPTY")
base_url = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")

print(f"API Key: {api_key}")
print(f"Base URL: {base_url}")

additional_colors = [
    colorname for (colorname, colorcode) in ImageColor.colormap.items()
]

# https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct/discussions/10
min_pixels = 512 * 28 * 28
max_pixels = 2048 * 28 * 28
import requests

# Default model list fallback
DEFAULT_MODEL_LIST = [
    "table_extractionXXX-NOT-AVAILABLE",
    "qwen_v2_5_vlXXX-NOT-AVAILABLE",
]


def get_model_list():
    """Fetch model list from endpoint, fallback to default if unavailable."""
    try:
        url = base_url.rstrip("/") + "/models"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            # Try to extract model names from OpenAI-compatible response
            if isinstance(data, dict) and "data" in data:
                return [m["id"] for m in data["data"] if "id" in m]
            # Or just a list of strings
            if isinstance(data, list):
                return data
        print(f"Model endpoint returned status {resp.status_code}, using default list.")
    except Exception as e:
        print(f"Could not fetch model list from endpoint: {e}")
    return DEFAULT_MODEL_LIST


DESCRIPTION = f"[QWEN 2.5VL API]"

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


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def estimate_token_count(prompt: str, model_name: str = "gpt-3.5-turbo-0301") -> int:
    """
    Estimates the number of tokens in the given prompt using tiktoken.

    Args:
        prompt: The text input for which to estimate token count.
        model_name: The model name from OpenAI to select the tokenizer. Defaults to "gpt-3.5-turbo-0301".

    Returns:
        The estimated number of tokens in the prompt.
    """
    try:
        tokenizer = tiktoken.encoding_for_model(model_name)
        tokens = tokenizer.encode(prompt)
        return len(tokens)
    except Exception as e:
        raise ValueError(f"An error occurred while estimating tokens: {e}")


def _b64_png(image_path: str) -> str:
    # RGB + modest PNG optimization to keep payload smaller without quality loss for OCR
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        buf = io.BytesIO()
        im.save(buf, format="PNG", optimize=True)
        return base64.b64encode(buf.getvalue()).decode("utf-8")


# Qwen/Qwen2.5-VL-7B-Instruct  Qwen/Qwen2.5-VL-72B-Instruct-AWQ"
def inference_with_api(
    image_path,
    prompt,
    sys_prompt="You are a helpful assistant.",
    model_id="qwen_v2_5_vl",
    min_pixels=1280 * 28 * 28,
    max_pixels=2400 * 28 * 28,
    temperature=0.1,
    top_p=0.2,
    enable_thinking=True,
):
    min_pixels = 512 * 28 * 28  # ~0.4 MP
    max_pixels = 1536 * 28 * 28  # ~1.2 MP

    # base64_image = encode_image(image_path)
    with io.BytesIO() as buffer:
        Image.open(image_path).convert("RGB").save(buffer, format="PNG")
        bytes_data = buffer.getvalue()

    image_type = "png"
    # base64_image = base64.b64encode(bytes_data).decode('utf-8')
    base64_image = _b64_png(image_path)
    estimated_tokens = estimate_token_count(prompt)
    estimated_tokens = 8192 * 2  #  estimated_tokens + 512  #
    # --max-batch-prefill-tokens 4096 --max-total-tokens 4096 --max-input-tokens 2048

    print('estimated_tokens : ', estimated_tokens)

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )
    # sys_prompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": sys_prompt},
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}",
                    },
                    "min_pixels": min_pixels,
                    "max_pixels": max_pixels,
                },
                {"type": "text", "text": prompt},
            ],
        },
    ]

    # store for debugging
    temp_dir = "/tmp/openai_messages"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, f"qwen-api_messages.json")
    try:
        import json

        with open(temp_file_path, "w") as temp_file:
            json.dump(messages, temp_file, indent=4)
        print(f"Messages saved for analysis to: {temp_file_path}")
    except Exception as e:
        print(f"Failed to save messages to {temp_file_path}: {e}")

    # 1. **temperature=0.0** – Minimizes randomness by always picking the highest probability token.
    # 2. **top_p=1.0** – Disables nucleus sampling, ensuring no additional probability mass is truncated.
    # 3. **frequency_penalty=0.0** and **presence_penalty=0.0** – Ensures no penalization that would otherwise alter token probabilities

    # Predictable + Complete
    # Parameter	Typical Value	Why
    # temperature	0.1–0.2	Keeps JSON completion stable; avoids deterministic freezing
    # top-p	0.1–0.3	Smooths token choice; resists OCR noise
    # penalties	0 / 0	Keeps schema identical every run

    max_tokens: int = estimated_tokens  # 2048
    stop: List[str] = []

    completion = client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        max_tokens=max_tokens,
        stop=stop,
        seed=42,  # ensures deterministic output across identical inputs
        # enable_thinking=enable_thinking,
        # THIS DOES NOT WORK YET
        extra_body={
            "chat_template_kwargs": {
                "thinking": enable_thinking,
                "enable_thinking": enable_thinking,
            }
        },
    )

    return completion.choices[0].message.content


def qwen_inference(
    media_input,
    text_input=None,
    system_prompt=None,
    model_id="qwen_v2_5_vl",
    temperature=0.1,
    top_p=0.2,
    enable_thinking=True,
):
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

    # Use provided system prompt or default
    if system_prompt is None or system_prompt.strip() == "":
        system_prompt = (
            "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        )

    result = inference_with_api(
        media_path,
        text_input,
        sys_prompt=system_prompt,
        model_id=model_id,
        temperature=temperature,
        top_p=top_p,
        enable_thinking=enable_thinking,
    )
    print('result')
    print(result)
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

    print("done")
    return result
    # print(buffer)
    #
    # json_output = parse_json(buffer)
    # print(json_output)
    # plot_bounding_boxes(image, buffer, input_width, input_height)


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
                # Model selection
                model_list = get_model_list()
                model_id = gr.Dropdown(
                    choices=model_list,
                    value=model_list[0] if model_list else "qwen_v2_5_vl",
                    label="Model",
                    info="Select the model to use",
                )
                system_prompt = gr.Textbox(
                    label="System Prompt",
                    value="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
                    lines=3,
                    placeholder="Enter system prompt here...",
                )
                text_input = gr.Textbox(label="Text Input (optional)")

                # Model parameters
                with gr.Row():
                    temperature = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=0.1,
                        step=0.1,
                        label="Temperature",
                        info="Controls randomness (0.0 = deterministic, higher = more random)",
                    )
                    top_p = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.2,
                        step=0.1,
                        label="Top-p",
                        info="Controls diversity (lower = more focused)",
                    )

                enable_thinking = gr.Checkbox(
                    label="Enable Thinking",
                    value=True,
                    info="Allow the model to show reasoning process",
                )

                submit_btn = gr.Button(value="Submit")
            with gr.Column():
                output_text = gr.Textbox(label="Output Text")

        input_media.change(
            fn=process_file_upload,
            inputs=[input_media],
            outputs=[input_media, preview_image],
        )

        submit_btn.click(
            qwen_inference,
            [
                input_media,
                text_input,
                system_prompt,
                model_id,
                temperature,
                top_p,
                enable_thinking,
            ],
            [output_text],
        )

demo.launch(debug=True, share=False)

# https://github.com/QwenLM/Qwen2.5-VL/issues/721
# https://github.com/QwenLM/Qwen2.5-VL/blob/main/cookbooks/spatial_understanding.ipynb
# Outline the position of  each table column header and output all the coordinates in JSON format with table header text as label.

# https://www.linkedin.com/pulse/advancing-document-ai-table-column-detection-made-simple-vaghela-px6ac
# https://github.com/VikParuchuri/marker/tree/master/marker
# python table_recognition.py --debug --images /home/greg/dev/rms/grapnel-g5/assets/TID-100985/ANNOTATED/OUT/tables/fragments/5_0-1200.png --skip_table_detection
# Converted
# https://github.com/RapidAI/TableStructureRec/blob/main/demo_wired.py
