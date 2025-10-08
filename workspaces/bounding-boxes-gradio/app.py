import uuid

import gradio as gr
from PIL import Image

from marie.boxes import BoxProcessorUlimDit, PSMode
from marie.boxes.dit.ulim_dit_box_processor import visualize_bboxes
from marie.models.utils import setup_torch_optimizations
from marie.utils.json import store_json_object


class GradioBoxProcessor:
    def __init__(self):
        self.sel_content_aware = True
        self.sel_bbox_optimization = True
        self.sel_bbox_refinement = True
        self.sel_bbox_strict_line_merge = False

        self.box = BoxProcessorUlimDit(
            models_dir="../../model_zoo/unilm/dit/text_detection",
            cuda=True,
        )

    def update_bbox_refinement(self, value):
        self.sel_bbox_refinement = value

    def update_content_aware(self, value):
        self.sel_content_aware = value

    def update_bbox_optimization(self, value):
        self.sel_bbox_optimization = value

    def update_bbox_strict_line_merge(self, value):
        self.sel_bbox_strict_line_merge = value

    def process_image(self, image):
        print("Processing image")
        print(f"Content Aware: {self.sel_content_aware}")
        print(f"BBox Optimization: {self.sel_bbox_optimization}")
        print(f"BBox Refinement: {self.sel_bbox_refinement}")
        print(f"BBox Strict Line Merge: {self.sel_bbox_strict_line_merge}")

        (
            boxes,
            fragments,
            lines,
            _,
            lines_bboxes,
        ) = self.box.extract_bounding_boxes(
            "gradio",
            "field",
            image,
            PSMode.SPARSE,
            self.sel_content_aware,
            self.sel_bbox_optimization,
            self.sel_bbox_refinement,
            self.sel_bbox_strict_line_merge,
        )

        bboxes_img = visualize_bboxes(
            image,
            boxes,
            format="xywh",
            # blackout_color=(100, 100, 200, 100),
            # blackout=True,
        )

        name = str(uuid.uuid4())
        store_json_object(boxes, f"/tmp/boxes/boxes-{name}.json")
        lines_img = visualize_bboxes(image, lines_bboxes, format="xywh")

        return bboxes_img, lines_img, len(boxes), len(lines_bboxes)


processor = GradioBoxProcessor()


def interface():
    article = """
         # DiT for Text Detection
         [DiT: Self-supervised Pre-training for Document Image Transformer](https://github.com/microsoft/unilm/tree/master/dit/text_detection)
        """

    with gr.Blocks() as iface:
        gr.Markdown(article)

        with gr.Row():
            with gr.Column():
                src = gr.Image(type="pil", sources=["upload"])
            with gr.Column():
                chk_apply_bbox_optimization = gr.Checkbox(
                    label="Bounding Box optimization",
                    value=processor.sel_bbox_optimization,
                    interactive=True,
                )

                chk_apply_content_aware = gr.Checkbox(
                    label="Content aware transformation",
                    value=processor.sel_content_aware,
                    interactive=True,
                )

                chk_apply_bbox_refinement = gr.Checkbox(
                    label="Bounding box refinement",
                    value=processor.sel_bbox_refinement,
                    interactive=True,
                )

                chk_apply_bbox_strict_line_merge = gr.Checkbox(
                    label="Strict line merge",
                    value=processor.sel_bbox_strict_line_merge,
                    interactive=True,
                )

                chk_apply_bbox_optimization.change(
                    lambda x: processor.update_bbox_optimization(x),
                    inputs=[chk_apply_bbox_optimization],
                    outputs=[],
                )

                chk_apply_content_aware.change(
                    lambda x: processor.update_content_aware(x),
                    inputs=[chk_apply_content_aware],
                    outputs=[],
                )

                chk_apply_bbox_refinement.change(
                    lambda x: processor.update_bbox_refinement(x),
                    inputs=[chk_apply_bbox_refinement],
                    outputs=[],
                )

                chk_apply_bbox_strict_line_merge.change(
                    lambda x: processor.update_bbox_strict_line_merge(x),
                    inputs=[chk_apply_bbox_strict_line_merge],
                    outputs=[],
                )

        with gr.Row():
            btn_reset = gr.Button("Clear")
            btn_submit = gr.Button("Submit", variant="primary")

        with gr.Row():
            with gr.Column():
                txt_bboxes = gr.components.Textbox(label="Bounding Boxes", value="0")
            with gr.Column():
                txt_lines = gr.components.Textbox(label="Lines", value="0")

        with gr.Row():
            with gr.Column():
                boxes = gr.components.Image(type="pil", label="boxes")
            with gr.Column():
                lines = gr.components.Image(type="pil", label="lines")

        btn_submit.click(
            processor.process_image,
            inputs=[src],
            outputs=[boxes, lines, txt_bboxes, txt_lines],
        )

    iface.launch(debug=True, share=True, server_name="0.0.0.0")


if __name__ == "__main__":
    setup_torch_optimizations()
    interface()
