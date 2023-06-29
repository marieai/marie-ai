import gradio as gr
from PIL import Image

from marie.boxes import BoxProcessorUlimDit, PSMode
from marie.boxes.dit.ulim_dit_box_processor import visualize_bboxes

box = BoxProcessorUlimDit(
    models_dir="../../model_zoo/unilm/dit/text_detection",
    cuda=True,
)

sel_bbox_optimization = False
sel_content_aware = False


def update_content_aware(value):
    global sel_content_aware
    sel_content_aware = value


def update_bbox_optimzation(value):
    global sel_bbox_optimization
    sel_bbox_optimization = value


def process_image(image):
    print("Processing image")
    print(f"Content Aware: {sel_content_aware}")
    print(f"BBox Optimization: {sel_bbox_optimization}")

    (boxes, fragments, lines, _, lines_bboxes,) = box.extract_bounding_boxes(
        "gradio",
        "field",
        image,
        PSMode.SPARSE,
        sel_content_aware,
        sel_bbox_optimization,
    )

    bboxes_img = visualize_bboxes(image, boxes, format="xywh")
    lines_img = visualize_bboxes(image, lines_bboxes, format="xywh")
    return bboxes_img, lines_img


def interface():
    article = """
         # DiT for Text Detection
         [DiT: Self-supervised Pre-training for Document Image Transformer](https://github.com/microsoft/unilm/tree/master/dit/text_detection)
        """

    with gr.Blocks() as iface:
        gr.Markdown(article)

        with gr.Row():
            with gr.Column():
                src = gr.Image(type="pil", source="upload")
            with gr.Column():
                chk_apply_bbox_optimization = gr.Checkbox(
                    label="Bounding Box optimization",
                    default=True,
                    interactive=True,
                )

                chk_apply_bbox_optimization.change(
                    lambda x: update_bbox_optimzation(x),
                    inputs=[chk_apply_bbox_optimization],
                    outputs=[],
                )

                chk_apply_content_aware = gr.Checkbox(
                    label="Content aware transformation",
                    default=True,
                    interactive=True,
                )

                chk_apply_content_aware.change(
                    lambda x: update_content_aware(x),
                    inputs=[chk_apply_content_aware],
                    outputs=[],
                )

        with gr.Row():
            btn_reset = gr.Button("Clear")
            btn_submit = gr.Button("Submit", variant="primary")

        with gr.Row():
            with gr.Column():
                boxes = gr.components.Image(type="pil", label="boxes")
            with gr.Column():
                lines = gr.components.Image(type="pil", label="lines")

        btn_submit.click(process_image, inputs=[src], outputs=[boxes, lines])

    iface.launch(debug=True, share=True, server_name="0.0.0.0")


if __name__ == "__main__":
    import torch

    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = False
    # torch._dynamo.config.suppress_errors = False

    interface()
