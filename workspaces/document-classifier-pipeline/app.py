import argparse
import os
import tempfile
from functools import partial
from typing import List, Union

import gradio as gr
import numpy as np
import torch as torch

from marie.conf.helper import load_yaml
from marie.helper import colored
from marie.logging.mdc import MDC
from marie.logging.profile import TimeContext
from marie.pipe.classification_pipeline import ClassificationPipeline
from marie.utils.docs import frames_from_file
from marie.utils.json import deserialize_value, to_json

use_cuda = torch.cuda.is_available()


def cleanup_json(mydict: dict):
    return deserialize_value(to_json(mydict))


def process_frames(
    frames: Union[np.ndarray, List[np.ndarray]],
    pipeline: ClassificationPipeline,
):
    MDC.put("request_id", "1")

    if not isinstance(frames, list):
        frames = [frames]
    filename = "test-gradio"
    with TimeContext(f"### ClassificationPipeline info"):
        results = pipeline.execute(
            ref_id=filename, ref_type="pid", frames=frames, runtime_conf=None
        )
    val = cleanup_json(results)
    print('val', val)
    return val


gallery_selection = None


def process_all_frames(pipeline: ClassificationPipeline, image_src):
    MDC.put("request_id", "2")
    frames = gradio_src_to_frames(image_src)
    results = process_frames(frames, pipeline)
    return results


def process_selection(pipeline: ClassificationPipeline, gallery_selection):
    print("process_selection")
    MDC.put("request_id", "3")
    filename = gallery_selection["name"]
    frame = frames_from_file(filename)[0]
    results = process_frames(frame, pipeline)

    return results


def gradio_src_to_frames(image_src):
    if image_src is None:
        return None
    if not isinstance(image_src, tempfile._TemporaryFileWrapper):
        raise Exception(
            "Expected image_src to be of type tempfile._TemporaryFileWrapper, "
            "ensure that the source is set to 'upload' in the gr.File component."
        )
    return frames_from_file(image_src.name)


def interface(classifier: ClassificationPipeline):
    def gallery_click_handler(src_gallery, evt: gr.SelectData):
        global gallery_selection
        gallery_selection = src_gallery[evt.index]

    with gr.Blocks() as iface:
        gr.HTML(
            """
            <div style="text-align: center; max-width: 1200px; margin: 20px auto;">
                <h1 style="font-weight: 900; font-size: 3rem; margin: 0rem">
                    Document Classification Pipeline
                </h1>        
            </div>
            """
        )
        with gr.Row():
            src = gr.File(type="file", source="upload")

        with gr.Row():
            btn_reset = gr.Button("Clear")
            # btn_submit = gr.Butt
            # .on("Classify All", variant="primary")
            # btn_submit_selected = gr.Button("Classify Selected", variant="primary")
            btn_grid = gr.Button("Build-Grid", variant="primary")

        with gr.Row(live=True):
            gallery = gr.Gallery(
                label="Image frames",
                show_label=False,
                elem_id="gallery",
                interactive=True,
            ).style(columns=4, object_fit="contain", height="auto")

        with gr.Row():
            btn_submit_all = gr.Button("Classify All", variant="primary")
            btn_submit_selected = gr.Button("Classify Selected", variant="primary")

        with gr.Row():
            with gr.Column():
                json_output = gr.outputs.JSON()

        btn_submit_all.click(
            partial(process_all_frames, classifier),
            inputs=[src],
            outputs=[json_output],
        )

        btn_submit_selected.click(
            partial(process_selection, classifier),
            inputs=[gallery],
            outputs=[json_output],
        )

        btn_grid.click(gradio_src_to_frames, inputs=[src], outputs=gallery)
        btn_reset.click(lambda: src.clear())

        gallery.select(gallery_click_handler, inputs=[gallery])

    iface.launch(debug=True, share=False, server_name="0.0.0.0")


def parse_args():
    parser = argparse.ArgumentParser(description="Document classification pipeline.")
    parser.add_argument(
        "--pipeline_path",
        type=str,
        help="Path to the pipeline configuration file",
    )
    args = parser.parse_args()
    print(f'{colored("[√]", "green")} Arguments are loaded.')
    print(args)
    return args


if __name__ == "__main__":
    import torch

    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = False
    # torch._dynamo.config.suppress_errors = False
    os.environ["MARIE_SUPPRESS_WARNINGS"] = "true"
    args = parse_args()
    config = load_yaml(
        os.path.expanduser(
            "~/dev/marieai/marie-ai/config/tests-integration/pipeline-classify-006.partial.yml"
        )
    )

    pipelines_config = config["pipelines"]
    pipeline = ClassificationPipeline(pipelines_config=pipelines_config)

    interface(classifier=pipeline)

# python ./app.py --pretrained_model_name_or_path  marie/lmv3-medical-document-classification
