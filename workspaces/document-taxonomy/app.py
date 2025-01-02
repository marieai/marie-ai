import os

import gradio as gr
import numpy as np
import torch as torch
from docarray import DocList
from PIL import Image

from marie.boxes import BoxProcessorUlimDit, PSMode
from marie.boxes.dit.ulim_dit_box_processor import visualize_bboxes
from marie.components.document_taxonomy.transformers import TransformersDocumentTaxonomy
from marie.document import TrOcrProcessor
from marie.executor.ner.utils import normalize_bbox
from marie.utils.docs import docs_from_image, frames_from_file

use_cuda = torch.cuda.is_available()

# TODO: Move this to document_taxonomy package
MODEL_ID = os.path.expanduser(
    "~/dev/flan-t5-text-classifier/flan-t5-eob-classification-taxonomy-trainer/R2-0.973529761794621"
)
max_input_length = 512

# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True, bnb_4bit_use_double_quant=False
# )
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

label2id = {"TABLE": 0, "SECTION": 1, "CODES": 2, "OTHER": 3}
id2label = {id: label for label, id in label2id.items()}


def build_ocr_engine():
    text_layout = None
    box_processor = BoxProcessorUlimDit(
        models_dir="../../model_zoo/unilm/dit/text_detection",
        cuda=use_cuda,
    )

    icr_processor = TrOcrProcessor(models_dir="../../model_zoo/trocr", cuda=use_cuda)

    return box_processor, icr_processor, text_layout


box_processor, icr_processor, text_layout = build_ocr_engine()


def process_taxonomies(documents: DocList, result: dict):
    print("Processing taxonomies")

    results = TransformersDocumentTaxonomy(model_name_or_path=MODEL_ID, batch_size=16)
    output = results.run(documents, result)

    print("results", results)
    print("output", output)

    taxonomy_groups = []
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

    bboxes_img = visualize_bboxes(image, boxes, format="xywh")
    lines_img = visualize_bboxes(overlay_image, lines_bboxes, format="xywh")
    frames = [image]
    documents = docs_from_image(frames)

    taxonomy_groups = process_taxonomies(documents, result, batch_size=16)
    taxonomy_line_bboxes = [group["bbox"] for group in taxonomy_groups]
    taxonomy_labels = [group["label"] for group in taxonomy_groups]
    image_width = image.size[0]
    taxonomy_line_bboxes = [
        [0, bbox[1], image_width, bbox[3]] for bbox in taxonomy_line_bboxes
    ]

    taxonomy_overlay_image = visualize_bboxes(
        image, np.array(taxonomy_line_bboxes), format="xywh", labels=taxonomy_labels
    )
    return bboxes_img, overlay_image, taxonomy_overlay_image, None  # to_json(result)


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
