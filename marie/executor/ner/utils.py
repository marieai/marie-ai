import io
import os
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFont

from marie.utils.utils import ensure_exists


def unnormalize_box(bbox, width, height):
    return [
        width * (bbox[0] / 1000),
        height * (bbox[1] / 1000),
        width * (bbox[2] / 1000),
        height * (bbox[3] / 1000),
    ]


def normalize_bbox(bbox, size):
    return [
        int(1000 * bbox[0] / size[0]),
        int(1000 * bbox[1] / size[1]),
        int(1000 * bbox[2] / size[0]),
        int(1000 * bbox[3] / size[1]),
    ]


def iob_to_label(label):
    label = label[2:]
    if not label:
        return "other"
    return label


def get_random_color():
    return (
        np.random.randint(50, 255),
        np.random.randint(50, 255),
        np.random.randint(50, 255),
        70,
    )


def draw_box(draw, box, text, fill_color, font):
    # in  xywh -> x1y1x2y2
    draw.rectangle(
        [(box[0], box[1]), (box[0] + box[2], box[1] + box[3])],
        outline="red",
        fill=fill_color,
        width=1,
    )
    if text is not None:
        draw.text(
            (box[0] + 10, box[1] - 10),
            text=f"{text}",
            fill="red",
            font=font,
            width=1,
        )


def visualize_prediction(
    output_filename,
    frame,
    true_predictions,
    true_boxes,
    true_scores,
    label2color,
    fmt: str = 'PNG',
    dpi: Tuple[int, int] = None,
    box_format="xyxy",
):
    if not isinstance(frame, PIL.Image.Image):
        raise ValueError("frame should be a PIL image")
    image = frame.copy()

    # https://stackoverflow.com/questions/54165439/what-are-the-exact-color-names-available-in-pils-imagedraw
    # label2color = get_label_colors()
    draw = ImageDraw.Draw(image, "RGBA")
    font = get_font(10)

    for prediction, box, score in zip(true_predictions, true_boxes, true_scores):
        # don't draw other
        label = prediction[2:]
        if not label:
            continue
        # xywh -> x1y1x2y2
        # if box_format == "xywh":
        box = [box[0], box[1], box[0] + box[2], box[1] + box[3]]

        predicted_label = iob_to_label(prediction).lower()
        draw.rectangle(
            (box[0], box[1], box[2], box[3]),
            outline=label2color[predicted_label.lower()],
            width=1,
        )
        draw.text(
            (box[0] + 10, box[1] - 10),
            text=f"{predicted_label}",
            fill="red",
            font=font,
            width=1,
        )

    # image.show()
    image.save(output_filename)
    del draw

    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format=fmt, dpi=dpi)
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr


def visualize_extract_kv(output_filename, frame, kv_results):
    """Visualize KV prediction"""
    image = frame.copy()
    draw_rbga = ImageDraw.Draw(image, "RGBA")
    font = get_font(10)

    def __draw(
        a_box,
    ):
        draw_box(
            draw_rbga,
            a_box,
            None,
            get_random_color(),
            font,
        )

    for i, kv in enumerate(kv_results):
        if "question" in kv["value"]:
            __draw(kv["value"]["question"]["bbox"])

        if "answer" in kv["value"]:
            __draw(kv["value"]["answer"]["bbox"])

    image.save(output_filename)
    # FIXME(flake8) :  F821 undefined name 'draw'
    # del draw


def get_font(size):
    try:
        font = ImageFont.truetype(os.path.join("./assets/fonts", "FreeSans.ttf"), size)
    except Exception as ex:
        font = ImageFont.load_default()

    return font


def visualize_icr(
    frames: Union[np.ndarray, List[Image.Image]],
    results: List[Dict[str, Any]],
    color: Tuple[int, int, int, int] = (150, 255, 150, 150),
    font_size: int = 15,
    padding: int = 8,
    cache_image: bool = False,
    filename_prefix: str = "visualize_icr",
) -> PIL.Image:
    """Visualize ICR results
    :param frames:  Image or list of images
    :param results: ICR results
    :param color: Color of overlay boxes
    :param font_size: Font size of the text
    :param padding: Padding around the text
    :param cache_image: Default False. If True, image will be cached in `/tmp/tensors/`
    :param filename_prefix:  cached image name {filename_prefix}_{page_idx}.png (only used if cache_image=True)
    :return: Image with Overlay boxes containing ICR results
    """
    assert len(frames) == len(results)
    ensure_exists("/tmp/tensors/")

    for page_idx, (image, result) in enumerate(zip(frames, results)):
        # convert from numpy to PIL
        img = image.copy()
        # we can have frames as both PIL and CV images
        if not isinstance(img, Image.Image):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            viz_img = Image.fromarray(img)
        else:
            viz_img = img

        font = get_font(font_size)

        words_all = []
        words = np.array(result["words"])

        for i, item in enumerate(words):
            box = item["box"]
            text = item["text"]
            words_all.append(text)

            # get text dimensions
            text_size = font.getbbox(text)
            text_w = text_size[2] - text_size[0]
            text_h = text_size[3] - text_size[1]
            button_size = (text_w + padding, text_h + padding)

            # create image with correct size and black background
            button_img = Image.new("RGBA", button_size, color=color)
            # put text on button with 10px margins
            button_draw = ImageDraw.Draw(button_img, "RGBA")
            button_draw.text(
                (padding // 2, padding // 2),
                text=text,
                font=font,
                stroke_width=0,
                fill=(0, 0, 0, 0),
                width=1,
            )
            # put button on source image in position (0, 0)
            viz_img.paste(button_img, (box[0], box[1]))

        if cache_image:
            viz_img.save(f"/tmp/tensors/{filename_prefix}_{page_idx}.png")

        return viz_img
