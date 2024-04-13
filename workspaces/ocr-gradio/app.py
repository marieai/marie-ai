import tempfile

import cv2
import gradio as gr
import numpy as np
import torch as torch
from PIL import Image

from marie.boxes import BoxProcessorUlimDit, PSMode
from marie.boxes.dit.ulim_dit_box_processor import visualize_bboxes
from marie.document import TrOcrProcessor
from marie.document.layoutreader import TextLayout
from marie.executor.ner.utils import normalize_bbox
from marie.renderer import TextRenderer
from marie.utils.docs import frames_from_file
from marie.utils.json import to_json

use_cuda = torch.cuda.is_available()


def build_ocr_engine():
    text_layout = TextLayout(
        "../../model_zoo/unilm/layoutreader/layoutreader-base-readingbank"
    )

    box_processor = BoxProcessorUlimDit(
        models_dir="../../model_zoo/unilm/dit/text_detection",
        cuda=use_cuda,
    )

    icr_processor = TrOcrProcessor(models_dir="../../model_zoo/trocr", cuda=use_cuda)

    return box_processor, icr_processor, text_layout


box_processor, icr_processor, text_layout = build_ocr_engine()


def to_text(result):
    """
    Create a text representation of the result

    :param result:
    :return:
    """
    # create a fake frames array from metadata in the results, this is needed for the renderer for sizing
    frames = []

    meta = result["meta"]["imageSize"]
    width = meta["width"]
    height = meta["height"]
    frames.append(np.zeros((height, width, 3), dtype=np.uint8))

    # write to temp file and read it back
    tmp_file = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix='.txt')

    renderer = TextRenderer(config={"preserve_interword_spaces": False})
    renderer.render(
        frames,
        [result],
        output_file_or_dir=tmp_file.name,
    )
    tmp_file.close()

    with open(tmp_file.name, "r") as f:
        return f.read()


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

    print(result)
    # get boxes and words from the result
    words = []
    boxes_norm = []
    # boxes = [word["box"] for word in words]

    for word in result["words"]:
        x, y, w, h = word["box"]
        w_box = [x, y, x + w, y + h]
        words.append(word["text"])
        boxes_norm.append(normalize_bbox(w_box, (image.size[0], image.size[1])))

    print("len words", len(words))
    print("len boxes", len(boxes_norm))
    layout_boxes = text_layout(words, boxes_norm)
    print(layout_boxes)

    bboxes_img = visualize_bboxes(image, boxes, format="xywh")
    lines_img = visualize_bboxes(overlay_image, lines_bboxes, format="xywh")

    return bboxes_img, overlay_image, lines_img, to_json(result), to_text(result)


def image_to_gallery(image_src):
    # image_file will be of tempfile._TemporaryFileWrapper type
    filename = image_src.name
    frames = frames_from_file(filename)
    return frames


def interface():
    article = """
         # Bounding Boxes and Intelligent Character Recognition(OCR)         
        """

    # [Dit: For textbox detection and  TROCR: Transformer-based OCR and ICR]

    def gallery_click_handler(src_gallery, evt: gr.SelectData):
        selection = src_gallery[evt.index]
        filename = selection["name"]
        return process_image(filename)

    with gr.Blocks() as iface:
        gr.Markdown(article)

        with gr.Row(variant="compact"):
            srcXXX = gr.components.Image(
                type="pil", source="upload", image_mode="L", label="Single page image"
            )
            src = gr.components.File(
                type="file", source="upload", label="Multi-page TIFF/PDF file"
            )

        with gr.Row():
            btn_reset = gr.Button("Clear")
            btn_submit = gr.Button("Process", variant="primary")
            btn_grid = gr.Button("Image-Grid", variant="primary")

        with gr.Row(live=True):
            gallery = gr.Gallery(
                label="Image frames",
                show_label=False,
                elem_id="gallery",
                interactive=True,
            ).style(columns=4, object_fit="contain", height="auto")

        btn_grid.click(image_to_gallery, inputs=[src], outputs=gallery)
        # btn_submit.click(process_image, inputs=[src], outputs=[fake, blended])
        btn_reset.click(lambda: src.clear())

        #
        # with gr.Row():
        #     src = gr.Image(type="pil", source="upload")
        #
        # with gr.Row():
        #     btn_reset = gr.Button("Clear")
        #     btn_submit = gr.Button("Submit", variant="primary")

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
                txt = gr.components.Textbox(label="text", max_lines=100)

        with gr.Row():
            with gr.Column():
                results = gr.components.JSON()

        # btn_submit.click(
        #     process_image, inputs=[src], outputs=[boxes, lines, results, txt]
        # )

        gallery.select(
            gallery_click_handler,
            inputs=[gallery],
            outputs=[boxes, icr, lines, results, txt],
        )

    iface.launch(debug=True, share=False, server_name="0.0.0.0")


if __name__ == "__main__":
    import torch

    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = False
    # torch._dynamo.config.suppress_errors = False

    interface()
