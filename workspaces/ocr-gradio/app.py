import gradio as gr
import torch as torch

from marie.boxes import BoxProcessorUlimDit, PSMode
from marie.boxes.dit.ulim_dit_box_processor import visualize_bboxes
from marie.document import TrOcrProcessor
from marie.ocr.mock_ocr_engine import MockOcrEngine

use_cuda = torch.cuda.is_available()

box_processor = BoxProcessorUlimDit(
    models_dir="../../model_zoo/unilm/dit/text_detection",
    cuda=use_cuda,
)
icr_processor = (
    MockOcrEngine()
)  # TrOcrIcrProcessor(models_dir="../../model_zoo/trocr", cuda=use_cuda)

icr_processor = TrOcrProcessor(models_dir="../../model_zoo/trocr", cuda=use_cuda)


def process_image(image):
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
    bboxes_img = visualize_bboxes(image, boxes, format="xywh")
    lines_img = visualize_bboxes(overlay_image, lines_bboxes, format="xywh")

    return bboxes_img, lines_img


def interface():
    article = """
         # Bounding Boxes and ICR         
        """
    # [Dit: For textbox detection and  TROCR: Transformer-based OCR and ICR]

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
                lines = gr.components.Image(type="pil", label="icr")

        btn_submit.click(process_image, inputs=[src], outputs=[boxes, lines])

    iface.launch(debug=True, share=False, server_name="0.0.0.0")


if __name__ == "__main__":
    import torch

    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = False
    # torch._dynamo.config.suppress_errors = False

    interface()
