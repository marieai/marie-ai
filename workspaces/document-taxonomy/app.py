import gradio as gr
import torch as torch
from PIL import Image

from marie.boxes import BoxProcessorUlimDit, PSMode
from marie.boxes.dit.ulim_dit_box_processor import visualize_bboxes
from marie.document import TrOcrProcessor
from marie.executor.ner.utils import normalize_bbox
from marie.utils.docs import frames_from_file
from marie.utils.json import to_json

use_cuda = torch.cuda.is_available()

prefix_text = "0"
import os
from time import time
from typing import List, Tuple

import torch
from loguru import logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# TODO: Move this to document_taxonomy package
MODEL_ID = os.path.expanduser(
    "~/dev/flan-t5-text-classifier/flan-t5-eob-classification-taxonomy"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
model.to(device)
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")

# 15-25% faster inference with high precision
torch.set_float32_matmul_precision('high')
model.eval()

label2id = {"TABLE": 0, "SECTION": 1, "CODES": 2, "OTHER": 3}
id2label = {id: label for label, id in label2id.items()}


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


def process_taxonomies(result: dict):
    print("Processing taxonomies")
    print(result)


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

    process_taxonomies(result)

    return bboxes_img, overlay_image, lines_img, to_json(result)


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
