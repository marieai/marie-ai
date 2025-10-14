import ast
import base64
import io
import os
import uuid
from typing import List, Optional, Tuple

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
MODEL_ID = "qwen_v2_5_vl"
DESCRIPTION = (
    f"[QWEN 2.5VL API] — Two-image mode: Image A (extract), Image B (structural hints)"
)

image_extensions = Image.registered_extensions()
video_extensions = ()

# ---------- Two-image default prompt ----------
TWO_IMAGE_PROMPT = """You are given TWO inputs:
- Image A (FIRST image): the page to EXTRACT from.
- Image B (SECOND image): a STRUCTURAL HINT image (column guides, header geometry).

Hard Rules:
1) Read ALL text/numbers ONLY from Image A.
2) Use Image B ONLY for geometry: column boundaries & header alignment.
3) If B conflicts with A, follow A. If B is blank/misaligned, ignore B.
4) No hallucinations. Missing = blank. Output Markdown only.

### Service Lines (table)
Headers (use EXACT names): DATES_OF_SERVICE | PROCEDURE_DESCRIPTION | BILLED_AMOUNT | DISCOUNT | ALLOWED_AMOUNT | DEDUCTIBLE | COINSURANCE | COPAY | BALANCE_PAYABLE | REMARKS_CODE
- Monetary fields: $ and exactly two decimals.
- Use B only to align columns. All values must come from A.
"""

# ----------------- Helpers -----------------


def identify_and_save_blob(blob_path: str) -> Tuple[str, str]:
    """Identifies if the blob is an image; saves to a temp file and returns (filename, media_type)."""
    try:
        with open(blob_path, 'rb') as file:
            blob_content = file.read()
            try:
                Image.open(io.BytesIO(blob_content)).verify()
                extension = ".png"  # Default to PNG for saving
                media_type = "image"
            except (IOError, SyntaxError):
                raise ValueError("The file is not a valid image.")

            filename = f"temp_{uuid.uuid4()}_media{extension}"
            with open(filename, "wb") as f:
                f.write(blob_content)
            return filename, media_type
    except FileNotFoundError:
        raise ValueError(f"The file {blob_path} was not found.")
    except Exception as e:
        raise ValueError(f"An error occurred while processing the file: {e}")


def process_file_upload(file_path) -> Tuple[Optional[str], Optional[Image.Image]]:
    """Process uploaded file and return the (file path, PIL.Image) if it's an image."""
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
                raise ValueError("Unsupported media type. Please upload an image.")
    return None, None


def parse_json(json_output: str) -> str:
    # Parsing out the markdown fencing
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i + 1 :])
            json_output = json_output.split("```")[0]
            break
    return json_output


def plot_bounding_boxes(im, bounding_boxes, input_width, input_height):
    img = im
    width, height = img.size
    draw = ImageDraw.Draw(img)
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

    bounding_boxes = parse_json(bounding_boxes)
    font = ImageFont.truetype("NotoSansCJK-Regular.ttc", size=14)
    try:
        json_output = ast.literal_eval(bounding_boxes)
    except Exception:
        end_idx = bounding_boxes.rfind('"}') + len('"}')
        truncated_text = bounding_boxes[:end_idx] + "]"
        json_output = ast.literal_eval(truncated_text)

    for i, bounding_box in enumerate(json_output):
        color = colors[i % len(colors)]
        abs_y1 = int(bounding_box["bbox_2d"][1] / input_height * height)
        abs_x1 = int(bounding_box["bbox_2d"][0] / input_width * width)
        abs_y2 = int(bounding_box["bbox_2d"][3] / input_height * height)
        abs_x2 = int(bounding_box["bbox_2d"][2] / input_width * width)
        if abs_x1 > abs_x2:
            abs_x1, abs_x2 = abs_x2, abs_x1
        if abs_y1 > abs_y2:
            abs_y1, abs_y2 = abs_y2, abs_y1
        draw.rectangle(((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=4)
        if "label" in bounding_box:
            draw.text(
                (abs_x1 + 8, abs_y1 + 6), bounding_box["label"], fill=color, font=font
            )
    img.show()


def encode_image_to_data_url(image_path: str) -> str:
    with io.BytesIO() as buffer:
        Image.open(image_path).convert("RGB").save(buffer, format="PNG")
        bytes_data = buffer.getvalue()
    b64 = base64.b64encode(bytes_data).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def estimate_token_count(prompt: str, model_name: str = "gpt-3.5-turbo-0301") -> int:
    try:
        tokenizer = tiktoken.encoding_for_model(model_name)
        tokens = tokenizer.encode(prompt)
        return len(tokens)
    except Exception as e:
        raise ValueError(f"An error occurred while estimating tokens: {e}")


# ----------------- OpenAI-style Inference -----------------


def build_user_content_two_images(
    prompt_text: str,
    img_a_path: str,
    img_b_path: Optional[str],
    min_px: int,
    max_px: int,
):
    """Order matters: Image A first (content), Image B second (hints)."""
    parts = []
    # Put the textual instructions first
    parts.append({"type": "text", "text": prompt_text})

    # Image A (must exist)
    data_url_a = encode_image_to_data_url(img_a_path)
    parts.append(
        {
            "type": "image_url",
            "image_url": {"url": data_url_a},
            "min_pixels": min_px,
            "max_pixels": max_px,
        }
    )

    # Optional Image B (structural hints)
    if img_b_path:
        data_url_b = encode_image_to_data_url(img_b_path)
        parts.append(
            {
                "type": "image_url",
                "image_url": {"url": data_url_b},
                "min_pixels": min_px,
                "max_pixels": max_px,
            }
        )
    return parts


def inference_with_api_two_images(
    image_a_path: str,
    image_b_path: Optional[str],
    prompt: str,
    sys_prompt: str = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
    model_id: str = MODEL_ID,
    min_px: int = 1280 * 28 * 28,
    max_px: int = 2400 * 28 * 28,
) -> str:
    # Token estimate (loose bound)
    estimated_tokens = max(8192, estimate_token_count(prompt) + 512)
    print('estimated_tokens : ', estimated_tokens)

    client = OpenAI(api_key=api_key, base_url=base_url)

    user_content = build_user_content_two_images(
        prompt_text=prompt,
        img_a_path=image_a_path,
        img_b_path=image_b_path,
        min_px=min_px,
        max_px=max_px,
    )

    messages = [
        {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
        {"role": "user", "content": user_content},
    ]

    # Store for debugging
    temp_dir = "/tmp/openai_messages"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, "qwen-api_messages_2img.json")
    try:
        import json

        with open(temp_file_path, "w") as temp_file:
            json.dump(messages, temp_file, indent=4)
        print(f"Messages saved for analysis to: {temp_file_path}")
    except Exception as e:
        print(f"Failed to save messages to {temp_file_path}: {e}")

    completion = client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=0.0,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        max_tokens=estimated_tokens,
        stop=[],
    )
    return completion.choices[0].message.content


# Backward-compatible single-image (used if B missing)
def inference_with_api_single_image(
    image_path: str,
    prompt: str,
    sys_prompt: str = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
    model_id: str = MODEL_ID,
    min_px: int = 1280 * 28 * 28,
    max_px: int = 2400 * 28 * 28,
) -> str:
    client = OpenAI(api_key=api_key, base_url=base_url)
    data_url = encode_image_to_data_url(image_path)
    messages = [
        {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": data_url},
                    "min_pixels": min_px,
                    "max_pixels": max_px,
                },
            ],
        },
    ]
    completion = client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=0.0,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        max_tokens=8192,
        stop=[],
    )
    return completion.choices[0].message.content


# ----------------- Gradio Pipeline -----------------


def qwen_inference(media_input_a, media_input_b, text_input: Optional[str] = None):
    """
    media_input_a: Image A (content to extract)
    media_input_b: Image B (structural hints) - optional
    text_input: optional extra prompt (defaults to TWO_IMAGE_PROMPT)
    """
    # Resolve Image A
    if isinstance(media_input_a, str):
        path_a, img_a = process_file_upload(media_input_a)
    else:
        path_a, img_a = None, None

    if not path_a:
        raise ValueError("Please upload Image A (content).")

    # Resolve Image B (optional)
    path_b, img_b = (None, None)
    if isinstance(media_input_b, str) and len(media_input_b) > 0:
        path_b, img_b = process_file_upload(media_input_b)

    prompt = (
        text_input.strip() if text_input and text_input.strip() else TWO_IMAGE_PROMPT
    )

    if path_b:
        # Two-image mode
        result = inference_with_api_two_images(
            image_a_path=path_a,
            image_b_path=path_b,
            prompt=prompt,
            model_id=MODEL_ID,
            min_px=min_pixels,
            max_px=max_pixels,
        )
    else:
        # Fallback single-image mode (A only)
        result = inference_with_api_single_image(
            image_path=path_a,
            prompt=prompt,
            model_id=MODEL_ID,
            min_px=min_pixels,
            max_px=max_pixels,
        )

    print('result')
    print(result)
    return result


# ----------------- UI -----------------

css = """
  #output {
    height: 500px;
    overflow: auto;
    border: 1px solid #ccc;
  }
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Tab(label="Two-Image Input"):
        with gr.Row():
            with gr.Column():
                input_media_a = gr.File(
                    label="Upload Image A (content to EXTRACT)", type="filepath"
                )
                preview_image_a = gr.Image(label="Preview A", visible=True)
                input_media_b = gr.File(
                    label="Upload Image B (STRUCTURAL HINTS)", type="filepath"
                )
                preview_image_b = gr.Image(label="Preview B", visible=True)
                text_input = gr.Textbox(
                    label="Prompt (optional; defaults to two-image EOB template)",
                    value=TWO_IMAGE_PROMPT,
                    lines=12,
                )
                submit_btn = gr.Button(value="Submit")
            with gr.Column():
                output_text = gr.Textbox(
                    label="Output Text", elem_id="output", lines=24
                )

        input_media_a.change(
            fn=process_file_upload,
            inputs=[input_media_a],
            outputs=[input_media_a, preview_image_a],
        )
        input_media_b.change(
            fn=process_file_upload,
            inputs=[input_media_b],
            outputs=[input_media_b, preview_image_b],
        )
        submit_btn.click(
            qwen_inference, [input_media_a, input_media_b, text_input], [output_text]
        )

demo.launch(debug=True, share=False)
