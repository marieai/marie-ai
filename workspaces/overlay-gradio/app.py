import os

import gradio as gr
from PIL import Image

from marie.overlay.overlay import OverlayProcessor
from marie.utils.utils import ensure_exists

overlay_processor = OverlayProcessor(work_dir=ensure_exists("/tmp/form-segmentation"))


def process_image(image):
    src_img_path = "/tmp/segment.png"
    image.save(src_img_path)
    docId = "segment"
    real, fake, blended = overlay_processor.segment(docId, src_img_path)

    return fake, blended


def interface():
    article = """
         # Document Overlay via Pix2Pix
        """

    with gr.Blocks() as iface:
        gr.Markdown(article)
        with gr.Row():
            src = gr.components.Image(
                type="pil", source="upload", image_mode="L", label="source"
            )

        with gr.Row():
            btn_reset = gr.Button("Clear")
            btn_submit = gr.Button("Submit", variant="primary")

        with gr.Row():
            with gr.Column():
                fake = gr.components.Image(type="pil", label="fake")
            with gr.Column():
                blended = gr.components.Image(type="pil", label="blended")

        btn_submit.click(process_image, inputs=[src], outputs=[fake, blended])
    iface.launch(debug=True, share=True, server_name="0.0.0.0")


if __name__ == "__main__":
    import torch

    img = Image.open(
        "/tmp/generators/1e93b981833c33a5268e664453229d55/burst/PID_179_8268_0_169750819_00022.tif"
    )
    img = img.convert("RGB")

    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = False

    # print(torch._dynamo.list_backends())e
    interface()
