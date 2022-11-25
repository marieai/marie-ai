import gradio as gr
from PIL import Image

from marie.boxes import BoxProcessorUlimDit, PSMode
from marie.boxes.dit.ulim_dit_box_processor import visualize_bboxes
from marie.utils.image_utils import hash_file
from marie.utils.json import to_json

box = BoxProcessorUlimDit(
    models_dir="./model_zoo/unilm/dit/text_detection",
    cuda=True,
)


def process_image(image):
    (
        boxes,
        fragments,
        lines,
        _,
        lines_bboxes,
    ) = box.extract_bounding_boxes("gradio", "field", image, PSMode.SPARSE)

    bboxes_img = visualize_bboxes(image, boxes, format="xywh")
    lines_img = visualize_bboxes(image, lines_bboxes, format="xywh")
    return bboxes_img, lines_img  # to_json(boxes)


def interface():
    title = "DiT for Text Detection"
    description = """<p></p>"""

    article = (
        "<p style='text-align: center'><a href='https://github.com/microsoft/unilm/tree/master/dit/text_detection' target='_blank'>"
        "DiT: Self-supervised Pre-training for Document Image Transformer</a> "
        "</p>"
    )
    examples = []

    iface = gr.Interface(
        fn=process_image,
        inputs=[
            gr.inputs.Image(type="pil"),
        ],
        outputs=[
            gr.outputs.Image(type="pil", label="boxes"),
            gr.outputs.Image(type="pil", label="lines"),
            # gr.outputs.JSON(),
        ],
        title=title,
        description=description,
        article=article,
        examples=examples,
        theme="default",
        # css=".footer{display:none !important}",
        live=False,
    )

    iface.launch(debug=True, share=True, server_name="0.0.0.0")


if __name__ == "__main__":
    interface()
