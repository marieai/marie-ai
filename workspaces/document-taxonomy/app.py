import copy

import gradio as gr
import numpy as np
import torch as torch
from PIL import Image
from verbalizers import verbalizers

from marie.boxes import BoxProcessorUlimDit, PSMode
from marie.boxes.dit.ulim_dit_box_processor import visualize_bboxes
from marie.document import TrOcrProcessor
from marie.executor.ner.utils import normalize_bbox
from marie.utils.docs import frames_from_file
from marie.utils.json import to_json
from marie.utils.utils import batchify

use_cuda = torch.cuda.is_available()

prefix_text = "0"
import os
from time import time
from typing import Dict, List, Tuple

import torch
from loguru import logger
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# TODO: Move this to document_taxonomy package
MODEL_ID = os.path.expanduser(
    "~/dev/flan-t5-text-classifier/flan-t5-eob-classification-taxonomy/checkpoint-1560"
)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=False
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID, quantization_config=quantization_config
)
model.to(device)
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")

# 15-25% faster inference with high precision
torch.set_float32_matmul_precision('high')
model.eval()

label2id = {"TABLE": 0, "SECTION": 1, "CODES": 2, "OTHER": 3}
id2label = {id: label for label, id in label2id.items()}


def create_chunks(metadata, tokenizer) -> List[Dict]:
    """
    Divides a document into chunks based on max token length.
    """
    # For an LLM, the context window should "have text around" the target point, meaning it should include both text before and after the current point of focus (the line)
    chunks = []
    max_token_length = tokenizer.model_max_length
    # adding spatial context
    lines = verbalizers("SPATIAL_FORMAT", metadata)

    for idx, line in enumerate(lines):
        line_text = line["text"]
        line_bbox_xywh = [int(x) for x in line["bbox"]]
        x, y = line_bbox_xywh[0], line_bbox_xywh[1]
        line["text"] = line_text + f" {x}|{y}"

    for idx, line in enumerate(lines):
        line_id = line["line"]
        chunk_size_start = 0
        chunk_size_end = 0
        chunk_idx = 0
        token_length = 0
        prompt = ""
        q = ""
        c = ""

        while token_length <= max_token_length:
            start = max(0, idx - chunk_size_start)
            end = min(len(lines), idx + chunk_size_end)
            source_row = lines[idx]
            selected_rows = lines[start:end]

            q = source_row["text"]
            c = "\n".join([r["text"] for r in selected_rows])
            current_prompt = f"""classify: {q}\ncontext: {c}\n"""

            tokens = tokenizer(
                current_prompt, return_tensors="pt", add_special_tokens=False
            )
            token_count = len(tokens["input_ids"][0])
            if token_count > max_token_length:
                break

            if chunk_idx % 2 == 0:
                chunk_size_start += 1
            else:
                chunk_size_end += 1

            chunk_idx += 1
            prompt = current_prompt
            token_length = token_count

            if start == 0 and end == len(lines):
                break
        chunks.append(
            {"line_id": line_id, "question": q, "context": c, "prompt": prompt}
        )
    return chunks


def classify(texts_to_classify: List[str]) -> List[Tuple[str, float]]:
    """Classify a list of texts using the model."""

    start = time()
    inputs = tokenizer(
        texts_to_classify,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding=True,
    ).to(device)

    # inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        outputs = model(**inputs)
    logger.debug(
        f"Classification of {len(texts_to_classify)} examples took {time() - start} seconds"
    )

    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=-1)

    # Get the top class and the corresponding probability (certainty) for each input text
    confidences, predicted_classes = torch.max(probs, dim=1)
    predicted_classes = (
        predicted_classes.cpu().numpy()
    )  # Move to CPU for numpy conversion if needed
    confidences = confidences.cpu().numpy()  # Same here

    predicted_labels = [id2label[class_id] for class_id in predicted_classes]
    return list(zip(predicted_labels, confidences))


def build_ocr_engine():
    text_layout = None
    box_processor = BoxProcessorUlimDit(
        models_dir="../../model_zoo/unilm/dit/text_detection",
        cuda=use_cuda,
    )

    icr_processor = TrOcrProcessor(models_dir="../../model_zoo/trocr", cuda=use_cuda)

    return box_processor, icr_processor, text_layout


box_processor, icr_processor, text_layout = build_ocr_engine()


def group_taxonomies_by_label(lines: List[Dict]) -> List[Dict]:
    """
    Groups contiguous lines with the same label into taxonomy groups.
    """
    if len(lines) == 0:
        return []

    grouped_lines = []
    current_group = {"label": lines[0]["taxonomy"]["label"], "lines": [lines[0]]}

    for line in lines[1:]:
        if line["taxonomy"]["label"] == current_group["label"]:
            current_group["lines"].append(line)
        else:
            grouped_lines.append(current_group)
            current_group = {"label": line["taxonomy"]["label"], "lines": [line]}

    grouped_lines.append(current_group)  # Add the last group

    for group in grouped_lines:
        print(f"Group: {group['label']}")
        group_size = len(group["lines"])
        total_score = 0
        min_x, min_y, max_x, max_y = (
            float('inf'),
            float('inf'),
            float('-inf'),
            float('-inf'),
        )
        for line in group["lines"]:
            score = line['taxonomy']['score']
            total_score += score
            score = f"{score:.4f}"
            line_info = f"Line {line['line']}: {score} > {line['text']}"
            print(line_info)
            bbox = line['bbox']
            min_x = min(min_x, bbox[0])
            min_y = min(min_y, bbox[1])
            max_x = max(max_x, bbox[0] + bbox[2])
            max_y = max(max_y, bbox[1] + bbox[3])
        average_score = total_score / group_size
        print(f"Average Score for Group '{group['label']}': {average_score:.4f}")
        group['bbox'] = [min_x, min_y, max_x - min_x, max_y - min_y]
        group['score'] = average_score
        print(f"Bounding Box for Group '{group['label']}': {group['bbox']}")

    return grouped_lines


def process_taxonomies(result: dict, batch_size: int = 16):
    print("Processing taxonomies")
    chunks = create_chunks(result, tokenizer)
    num_batches = len(chunks) // batch_size + (len(chunks) % batch_size > 0)
    batched_chunks = batchify(chunks, batch_size)
    print("num_batches", num_batches)

    for idx, batch in enumerate(batched_chunks):
        print(f"Processing batch {idx + 1}/{num_batches}")
        texts = [chunk["prompt"] for chunk in batch]
        predictions = classify(texts)
        print(predictions)
        for chunk, prediction in zip(batch, predictions):
            chunk["prediction"] = prediction
            for line in result["lines"]:
                if line["line"] == chunk["line_id"]:
                    label = chunk["prediction"][0]
                    score = chunk["prediction"][1]
                    line["taxonomy"] = {"label": label, "score": score}
                    break

    taxonomy_groups = group_taxonomies_by_label(result["lines"])
    return taxonomy_groups


def process_image(filename):
    print("Processing image : ", filename)
    image = Image.open(filename).convert("RGB")
    # image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    (
        boxes,
        fragments,
        lines,
        _,
        lines_bboxes,
    ) = box_processor.extract_bounding_boxes("gradio", "field", image, PSMode.SPARSE)

    result, overlay_image = icr_processor.recognize(
        "gradio ", "00000", image, boxes, fragments, lines, return_overlay=True
    )

    # get boxes and words from the result
    words = []
    boxes_norm = []

    for word in result["words"]:
        x, y, w, h = word["box"]
        w_box = [x, y, x + w, y + h]
        words.append(word["text"])
        boxes_norm.append(normalize_bbox(w_box, (image.size[0], image.size[1])))

    print("len words", len(words))
    print("len boxes", len(boxes_norm))
    # layout_boxes = text_layout(words, boxes_norm)
    # print(layout_boxes)

    bboxes_img = visualize_bboxes(image, boxes, format="xywh")
    lines_img = visualize_bboxes(overlay_image, lines_bboxes, format="xywh")

    taxonomy_groups = process_taxonomies(result, batch_size=16)
    taxonomy_line_bboxes = [group["bbox"] for group in taxonomy_groups]
    taxonomy_labels = [group["label"] for group in taxonomy_groups]
    image_width = image.size[0]
    taxonomy_line_bboxes = [
        [0, bbox[1], image_width, bbox[3]] for bbox in taxonomy_line_bboxes
    ]

    taxonomy_overlay_image = visualize_bboxes(
        image, np.array(taxonomy_line_bboxes), format="xywh", labels=taxonomy_labels
    )
    return bboxes_img, overlay_image, taxonomy_overlay_image, to_json(result)


def image_to_gallery(image_src):
    # image_file will be of tempfile._TemporaryFileWrapper type
    filename = image_src.name
    frames = frames_from_file(filename)
    return frames


def print_textbox(x):
    print(x)
    global prefix_text
    prefix_text = x


def interface():
    article = """
         # Document Taxonomy     
        """

    def gallery_click_handler(src_gallery, evt: gr.SelectData):
        selection = src_gallery[evt.index]

        print("selection", selection)
        # filename = selection["name"]
        filename = selection[0]
        return process_image(filename)

    with gr.Blocks() as iface:
        gr.Markdown(article)

        with gr.Row(variant="compact"):
            with gr.Column():
                src = gr.components.File(
                    type="filepath",  # Corrected type parameter
                    label="Multi-page TIFF/PDF file",
                    file_count="single",
                )
                with gr.Row():
                    btn_reset = gr.Button("Clear")
                    btn_grid = gr.Button("Image-Grid", variant="primary")

            with gr.Column():
                chk_filter_results = gr.Checkbox(
                    label="Filter results",
                    value=False,
                    interactive=True,
                )

                gr.Number(
                    label="Threshold",
                    minimum=0,
                    maximum=1,
                    step=0.1,
                    value=0.95,
                    interactive=True,
                    precision=2,
                )

                # chk_apply_overlay.change(
                #     lambda x: update_overlay(x),
                #     inputs=[chk_apply_overlay],
                #     outputs=[],
                # )

        with gr.Row():
            gallery = gr.Gallery(
                label="Image frames",
                show_label=False,
                elem_id="gallery",
                interactive=True,
            )

        btn_grid.click(image_to_gallery, inputs=[src], outputs=gallery)
        btn_reset.click(lambda: src.clear())

        with gr.Row():
            with gr.Column():
                boxes = gr.components.Image(type="pil", label="boxes")
            with gr.Column():
                lines = gr.components.Image(type="pil", label="lines")
        with gr.Row():
            with gr.Column():
                icr = gr.components.Image(type="pil", label="icr")

        with gr.Row():
            with gr.Column():
                results = gr.components.JSON()

        gallery.select(
            gallery_click_handler,
            inputs=[gallery],
            outputs=[boxes, icr, lines, results],
        )

    iface.launch(debug=True, share=False, server_name="0.0.0.0")


if __name__ == "__main__":
    import torch

    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = False
    # torch._dynamo.config.suppress_errors = False

    interface()
