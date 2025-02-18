import base64
import io
from typing import List, Union

import torch
from PIL import Image


def is_jpeg(data):
    jpeg_signature = b'\xFF\xD8\xFF'
    return data.startswith(jpeg_signature)


def is_png(data):
    png_signature = b'\x89\x50\x4E\x47\x0D\x0A\x1A\x0A'
    return data.startswith(png_signature)


def get_image_type_from_bytes(data):
    if is_jpeg(data):
        return "jpeg"
    elif is_png(data):
        return "png"
    else:
        raise ValueError("Image type not supported, only jpeg and png supported.")


def as_bytes(image_path: str) -> bytes:
    """
    Open an image from image_path, convert it to an RGB PNG, and return its bytes.
    """
    with io.BytesIO() as buffer:
        Image.open(image_path).convert("RGB").save(buffer, format="PNG")
        return buffer.getvalue()


def extract_vision_info(conversations: list[dict] | list[list[dict]]) -> list[dict]:
    vision_infos = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if (
                        "image" in ele
                        or "image_url" in ele
                        or "video" in ele
                        or ele["type"] in ("image", "image_url", "video")
                    ):
                        vision_infos.append(ele)
    return vision_infos


def extract_text_info(conversations: list[dict] | list[list[dict]]) -> list[dict]:
    text_infos = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if "text" in ele:
                        text_infos.append(ele)
    return text_infos


def fetch_image(
    ele: dict[str, str | Image.Image],
) -> Image.Image:
    if "image" in ele:
        image = ele["image"]
    else:
        image = ele["image_url"]
    image_obj = None
    if isinstance(image, Image.Image):
        image_obj = image
    elif image.startswith("file://"):
        image_obj = Image.open(image[7:])
    elif image.startswith("data:image"):
        if "base64," in image:
            _, base64_data = image.split("base64,", 1)
            data = base64.b64decode(base64_data)
            image_obj = Image.open(io.BytesIO(data))
    else:
        image_obj = Image.open(image)
    if image_obj is None:
        raise ValueError(
            f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}"
        )
    image = image_obj.convert("RGB")

    return image


def process_vision_info(
    conversations: list[dict] | list[list[dict]],
) -> tuple[list[Image.Image] | None, list[torch.Tensor | list[Image.Image]] | None]:
    """lifted  from qwen_vl_utils import process_vision_info"""
    ## Read images from messages, we don't do video for now
    vision_infos = extract_vision_info(conversations)
    image_inputs = []
    for vision_info in vision_infos:
        if "image" in vision_info or "image_url" in vision_info:
            image_inputs.append(fetch_image(vision_info))
        else:
            raise ValueError("image, image_url or video should in content.")

    return image_inputs, None


def open_ai_like_formatting(
    content: List[Union[str, bytes | Image.Image]], remote: bool = False
) -> List[dict]:
    """Helper function to format a list of strings and bytes into a list of dictionaries to pass as messages to the API."""
    formatted_content = []
    for item in content:
        if isinstance(item, Image.Image):
            if remote:
                with io.BytesIO() as buffer:
                    Image.open(item).convert("RGB").save(buffer, format="PNG")
                    bytes_data = buffer.getvalue()
                image_type = get_image_type_from_bytes(bytes_data)
                base64_image = base64.b64encode(bytes_data).decode('utf-8')
                formatted_content.append(
                    {
                        "type": "image",
                        "image": f"data:image/{image_type};base64,{base64_image}",
                        "image_urlXXX": {
                            "url": f"data:image/{image_type};base64,{base64_image}"
                        },
                    }
                )
            else:
                formatted_content.append(
                    {
                        "type": "image",
                        "image": item,
                    }
                )
        elif isinstance(item, bytes):
            # For now, bytes are assumed to be images
            image_type = get_image_type_from_bytes(item)
            base64_image = base64.b64encode(item).decode('utf-8')
            formatted_content.append(
                {
                    "type": "image",
                    "image_url": f"data:image/{image_type};base64,{base64_image}",
                    "image_urlXXX": {
                        "url": f"data:image/{image_type};base64,{base64_image}"
                    },
                }
            )
        elif isinstance(item, str):
            formatted_content.append({"type": "text", "text": item})
        else:
            raise ValueError(f"Unsupported input type: {type(item)}")
    return formatted_content


def convert_openai_to_transformers_format(conversation):
    """
    Converts OpenAI-style conversation format into a structure compatible with
    Transformers' `apply_chat_template()`.
    """
    transformed_conversation = []

    for message in conversation:
        role = message["role"]
        content = message["content"]
        if isinstance(content, list):
            content_text = "\n".join(
                item["text"] if isinstance(item, dict) and "text" in item else str(item)
                for item in content
            )
        else:
            content_text = str(content)
        transformed_conversation.append({"role": role, "content": content_text})

    return transformed_conversation


def force_download(model_name):
    """
    Force download the model and tokenizer.
    For models with special configurations, dynamically detect and use the appropriate class.
    """
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    from transformers.models.llava_next import (
        LlavaNextConfig,
        LlavaNextForConditionalGeneration,
    )

    try:
        print(f"Attempting to load model: {model_name}")
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

        # Handle LlavaNextConfig specifically
        if isinstance(config, LlavaNextConfig):
            print(
                f"Detected LlavaNextConfig, using LlavaNextForConditionalGeneration for {model_name}"
            )
            model = LlavaNextForConditionalGeneration.from_pretrained(
                model_name, trust_remote_code=True
            )
        else:
            model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print(f"Model and tokenizer for '{model_name}' loaded successfully.")
        return model, tokenizer

    except Exception as e:
        print(f"Error loading the model '{model_name}': {e}")
        raise
