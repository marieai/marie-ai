import gradio as gr
from PIL import Image

from marie.boxes import BoxProcessorUlimDit, PSMode
from marie.boxes.dit.ulim_dit_box_processor import visualize_bboxes
from marie.components.document_registration.unilm_dit import (
    UnilmDocumentBoundaryRegistration,
)
from marie.utils.docs import docs_from_image

processor = UnilmDocumentBoundaryRegistration(
    model_name_or_path="../../model_zoo/unilm/dit/object_detection/document_boundary",
    use_gpu=True,
)


def process_image(image):
    print("Processing image")
    documents = docs_from_image(image)
    print("Documents: ", len(documents))
    results = processor.run(documents)

    print("Results: ", results)

    # bboxes_img = visualize_bboxes(image, boxes, format="xywh")
    # lines_img = visualize_bboxes(image, lines_bboxes, format="xywh")
    # return bboxes_img, lines_img


def interface():
    article = """
         # DiT for Object Detection - Document Boundary Registration
         [DiT: Self-supervised Pre-training for Document Image Transformer](https://github.com/microsoft/unilm/tree/master/dit/obj)
        """

    with gr.Blocks() as iface:
        gr.Markdown(article)

        with gr.Row():
            src = gr.Image(type="pil", source="upload")
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
