import gradio as gr

from marie.components.document_registration.datamodel import DocumentBoundaryPrediction
from marie.components.document_registration.unilm_dit import (
    UnilmDocumentBoundaryRegistration,
)
from marie.utils.docs import docs_from_image

processor = UnilmDocumentBoundaryRegistration(
    model_name_or_path="../../model_zoo/unilm/dit/object_detection/document_boundary",
    use_gpu=True,
    debug_visualization=True,
)

registration_method = "absolute"


def update_registration_method(value):
    global registration_method
    registration_method = value


def process_image(
    image, margin_width, margin_height, registration_point_x, registration_point_y
):
    print(f"Registration Method: {registration_method}")
    print(f"Registration Point X: {registration_point_x}")
    print(f"Registration Point Y: {registration_point_y}")
    print(f"Margin Width: {margin_width}")
    print(f"Margin Height: {margin_height}")

    documents = docs_from_image(image)
    results = processor.run(
        documents,
        registration_method,
        (registration_point_x, registration_point_y),
        margin_width,
        margin_height,
    )
    result = results[0]
    boundary: DocumentBoundaryPrediction = result.tags["document_boundary"]

    return boundary.aligned_image, boundary.visualization_image


def interface():
    article = """
         # DiT for Object Detection - Document Boundary Registration
         [DiT: Self-supervised Pre-training for Document Image Transformer](https://github.com/microsoft/unilm/tree/master/dit/obj)
        """

    with gr.Blocks() as iface:
        gr.Markdown(article)

        with gr.Row():
            with gr.Column():
                src = gr.Image(type="pil", source="upload")

            with gr.Column():
                chk_registration_method = gr.Radio(
                    ["absolute", "fit_to_page"],
                    label="Registration Method",
                )

                chk_registration_method.change(
                    lambda x: update_registration_method(x),
                    inputs=[chk_registration_method],
                    outputs=[],
                )

                slider_mw = gr.Slider(
                    label="Margin Width",
                    min=0,
                    max=100,
                    step=1,
                    value=5,
                    interactive=True,
                )
                slider_mh = gr.Slider(
                    label="Margin Height",
                    min=0,
                    max=100,
                    step=1,
                    value=5,
                    interactive=True,
                )

                slider_rpx = gr.Slider(
                    label="Registration Point X",
                    min=0,
                    max=100,
                    step=1,
                    value=10,
                    interactive=True,
                )
                slider_rpy = gr.Slider(
                    label="Registration Point Y",
                    min=0,
                    max=100,
                    step=1,
                    value=10,
                    interactive=True,
                )

        with gr.Row():
            btn_reset = gr.Button("Clear")
            btn_submit = gr.Button("Submit", variant="primary")

        with gr.Row():
            with gr.Column():
                aligned_img = gr.components.Image(type="pil", label="Aligned Image")
            with gr.Column():
                visualization_img = gr.components.Image(
                    type="pil", label="Visualization Image"
                )

        btn_submit.click(
            process_image,
            inputs=[src, slider_mw, slider_mh, slider_rpx, slider_rpy],
            outputs=[aligned_img, visualization_img],
        )
    iface.launch(debug=True, share=True, server_name="0.0.0.0")


if __name__ == "__main__":
    import torch

    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = False
    # torch._dynamo.config.suppress_errors = False

    interface()
