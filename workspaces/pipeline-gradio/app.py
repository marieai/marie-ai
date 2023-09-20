import gradio as gr
import numpy as np
import torch as torch
from PIL import Image

from effect import (
    blur,
    morphology,
    pepper,
    salt,
    bleed_through,
)
from marie.boxes import BoxProcessorUlimDit, PSMode
from marie.boxes.dit.ulim_dit_box_processor import visualize_bboxes
from marie.document import TrOcrProcessor
from marie.ocr.mock_ocr_engine import MockOcrEngine
from marie.overlay.overlay import OverlayProcessor
from marie.utils.utils import ensure_exists

use_cuda = torch.cuda.is_available()

box_processor = BoxProcessorUlimDit(
    models_dir="../../model_zoo/unilm/dit/text_detection",
    cuda=use_cuda,
)
icr_processor = (
    MockOcrEngine()
)  # TrOcrIcrProcessor(models_dir="../../model_zoo/trocr", cuda=use_cuda)

icr_processor = TrOcrProcessor(models_dir="../../model_zoo/trocr", cuda=use_cuda)

overlay_processor = OverlayProcessor(work_dir=ensure_exists("/tmp/form-segmentation"))

selected_scale = 100
selected_augmentation = "none"
selected_overlay = True


def update_overlay(value):
    global selected_overlay
    selected_overlay = value


def update_scale(value):
    global selected_scale
    selected_scale = value


def update_augmentation(value):
    global selected_augmentation
    selected_augmentation = value


def process_image(image):
    # apply scale transformation
    print(f"Scale: {selected_scale}")
    print(f"Augmentation: {selected_augmentation}")
    print(f"Overlay: {selected_overlay}")

    scale = selected_scale
    aug = selected_augmentation
    has_overlay = selected_overlay

    if scale != 100:
        image = image.resize(
            (int(image.width * int(scale) / 100), int(image.height * int(scale) / 100))
        )

    # apply augmentation transformation

    if aug != "none":
        src = np.array(image)
        if aug == "dilate":
            converted = morphology(
                src, operation="open", kernel_shape=(3, 3), kernel_type="ones"
            )
        elif aug == "erode":
            converted = morphology(
                src, operation="erode", kernel_shape=(3, 3), kernel_type="ones"
            )
        elif aug == "close":
            converted = morphology(
                src, operation="close", kernel_shape=(3, 3), kernel_type="ones"
            )
        elif aug == "open":
            converted = morphology(
                src, operation="open", kernel_shape=(3, 3), kernel_type="ones"
            )
        elif aug == "blur":
            converted = blur(src)
        elif aug == "pepper":
            converted = pepper(src)
        elif aug == "salt":
            converted = salt(src)
        elif aug == "bleed":
            converted = bleed_through(src)

        image = Image.fromarray(converted)

    # save image to disk and pass to overlay processor
    src_img_path = "/tmp/segment.png"
    image.save(src_img_path)

    docId = "segment"
    if has_overlay:
        real, fake, blended = overlay_processor.segment(docId, src_img_path)
    else:
        real, fake, blended = image, image, image

    # pil image from opencv
    # image = Image.fromarray(blended)
    image = blended
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

    bboxes_img = visualize_bboxes(image, boxes, format="xywh")
    lines_img = visualize_bboxes(overlay_image, lines_bboxes, format="xywh")

    return fake, blended, bboxes_img, lines_img, result


def interface():
    article = """
         # Document processing pipeline
        """

    with gr.Blocks() as iface:
        gr.Markdown(article)

        with gr.Row():
            with gr.Column():
                # add gradio checkbox for "show overlay"
                chk_apply_overlay = gr.Checkbox(
                    label="Apply overlay transformation",
                    default=True,
                    interactive=True,
                )

                chk_apply_overlay.change(
                    lambda x: update_overlay(x),
                    inputs=[chk_apply_overlay],
                    outputs=[],
                )

                chk_scale = gr.Radio(
                    label="Scale (percent)",
                    choices=["100", "75", "50"],
                    value="100",
                    interactive=True,
                )

                chk_scale.change(
                    lambda x: update_scale(x), inputs=[chk_scale], outputs=[]
                )

                with gr.Row():
                    # chk_kernel_type = gr.Radio(
                    #     label="Kernel type",
                    #     choices=["ones", "upper", "lower", "x", "plus", "ellipse"],
                    #     value="none",
                    #     interactive=True,
                    # )

                    chk_augment = gr.Radio(
                        label="Image augmentation",
                        choices=[
                            "none",
                            "dilate",
                            "erode",
                            "close",
                            "open",
                            "blur",
                            "pepper",
                            # "salt", # too salty
                            "bleed",
                        ],
                        value="none",
                        interactive=True,
                    )

                    chk_augment.change(
                        lambda x: update_augmentation(x),
                        inputs=[chk_augment],
                        outputs=[],
                    )

            with gr.Column():
                with gr.Row():
                    src = gr.components.Image(
                        type="pil", source="upload", image_mode="L", label="source"
                    )

                with gr.Row():
                    # with gr.Column():
                    #     btn_reset = gr.Button("Clear")
                    with gr.Column():
                        btn_submit = gr.Button("Submit", variant="primary")

        with gr.Row():
            with gr.Column():
                fake = gr.components.Image(type="numpy", label="overlay")
            with gr.Column():
                blended = gr.components.Image(type="numpy", label="blended")

        with gr.Row():
            with gr.Column():
                bboxes_img = gr.components.Image(type="numpy", label="bboxes")
            with gr.Column():
                lines_img = gr.components.Image(type="numpy", label="lines")

        # with gr.Row():
        #     with gr.Column():
        #         results = gr.components.JSON()

        btn_submit.click(
            process_image,
            inputs=[src],
            outputs=[fake, blended, bboxes_img, lines_img],
        )
    iface.launch(debug=True, share=True, server_name="0.0.0.0", show_api=False)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = False

    interface()
