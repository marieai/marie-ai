import os
import tempfile
from functools import partial
from typing import List, Union

import gradio as gr
import numpy as np
import torch as torch

from marie.components import TransformersDocumentClassifier
from marie.helper import colored
from marie.logging.mdc import MDC
from marie.ocr import DefaultOcrEngine, MockOcrEngine
from marie.ocr.util import get_words_and_boxes
from marie.utils.docs import frames_from_file, docs_from_image
from marie.logging.predefined import default_logger as logger
import argparse

use_cuda = torch.cuda.is_available()

# # TODO : add support for dependency injection
# MDC.put("request_id", "0")

mock_ocr = False
if mock_ocr:
    ocr_engine = MockOcrEngine(cuda=use_cuda)
else:
    ocr_engine = DefaultOcrEngine(cuda=use_cuda)


def process_frames(
    frames: Union[np.ndarray, List[np.ndarray]],
    model_name_or_path: str,
    classifier: TransformersDocumentClassifier,
):
    MDC.put("request_id", "1")

    if not isinstance(frames, list):
        frames = [frames]

    ocr_results = ocr_engine.extract(frames)
    # classifier = TransformersDocumentClassifier(model_name_or_path=model_name_or_path)
    documents = docs_from_image(frames)

    words = []
    boxes = []

    for page_idx in range(len(frames)):
        page_words, page_boxes = get_words_and_boxes(ocr_results, page_idx)
        words.append(page_words)
        boxes.append(page_boxes)

    classified_docs = classifier.run(documents=documents, words=words, boxes=boxes)
    results = []

    for page_idx, document in enumerate(classified_docs):
        results.append(
            {
                "page": page_idx,
                "classification": document.tags['classification'],
            }
        )

    return results


gallery_selection = None


def process_all_frames(
    model_name_or_path: str, classifier: TransformersDocumentClassifier, image_src
):
    MDC.put("request_id", "2")
    frames = gradio_src_to_frames(image_src)
    results = process_frames(
        frames, model_name_or_path=model_name_or_path, classifier=classifier
    )
    return results


def process_selection(model_name_or_path: str, evt):
    print("process_selection", evt)
    print("model_name_or_path", model_name_or_path)
    MDC.put("request_id", "3")
    filename = gallery_selection["name"]
    frame = frames_from_file(filename)[0]
    results = process_frames(frame, model_name_or_path=model_name_or_path)

    return results[0]


def gradio_src_to_frames(image_src):
    if image_src is None:
        return None
    if not isinstance(image_src, tempfile._TemporaryFileWrapper):
        raise Exception(
            "Expected image_src to be of type tempfile._TemporaryFileWrapper, "
            "ensure that the source is set to 'upload' in the gr.File component."
        )
    return frames_from_file(image_src.name)


def interface(model_name_or_path: str, classifier: TransformersDocumentClassifier):
    def gallery_click_handler(src_gallery, evt: gr.SelectData):
        global gallery_selection
        gallery_selection = src_gallery[evt.index]

    with gr.Blocks() as iface:
        gr.HTML(
            """
            <div style="text-align: center; max-width: 1200px; margin: 20px auto;">
            <h1 style="font-weight: 900; font-size: 3rem; margin: 0rem">
                Document Classification : LayoutLMv3
            </h1>        
            <h3 style="font-weight: 450; font-size: 1rem; margin: 0rem"> 
            [<a href="https://arxiv.org/abs/2204.08387" style="color:blue;">arXiv</a>] 
            </h3> 
            <h2 style="text-align: left; font-weight: 450; font-size: 1rem; margin-top: 0.5rem; margin-bottom: 0.5rem">
             LayoutLMv3 is capable of recognizing and encoding both the textual content and the visual layout of a document, allowing it to provide superior performance on document analysis tasks.
            </h2>

            </div>
            """
        )
        with gr.Row():
            src = gr.File(type="file", source="upload")

        with gr.Row():
            btn_reset = gr.Button("Clear")
            # btn_submit = gr.Button("Classify All", variant="primary")
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
            partial(process_all_frames, model_name_or_path, classifier),
            inputs=[src],
            outputs=[json_output],
        )
        btn_submit_selected.click(
            partial(process_selection, model_name_or_path, classifier),
            inputs=[gallery],
            outputs=[json_output],
        )

        btn_grid.click(gradio_src_to_frames, inputs=[src], outputs=gallery)
        btn_reset.click(lambda: src.clear())

        gallery.select(gallery_click_handler, inputs=[gallery])

    iface.launch(debug=True, share=False, server_name="0.0.0.0")


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default='marie/lmv3-document-classification',
        help="Path to pretrained model or model identifier from Model Hub",
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
    model_name_or_path = args.pretrained_model_name_or_path

    logger.info(f"Using model : {model_name_or_path}")
    classifier = TransformersDocumentClassifier(model_name_or_path=model_name_or_path)

    interface(model_name_or_path=model_name_or_path, classifier=classifier)

# python ./app.py --pretrained_model_name_or_path  marie/lmv3-medical-document-classification
